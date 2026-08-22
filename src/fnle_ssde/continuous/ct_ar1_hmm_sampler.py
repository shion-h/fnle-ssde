"""
Continuous-time AR(1)-HMM Gibbs sampler in a single research-oriented file.

This implementation combines:
1. Uniformization / candidate jumps for the continuous-time discrete state path z(t)
2. Conditional NUTS updates for the continuous latent trajectory y on an augmented grid
3. FFBS updates for the discrete state skeleton on the same augmented grid
4. Conditional NUTS updates for the NLE transition parameters
5. Conjugate Gibbs updates for the diagonal observation noise
6. Conjugate Gibbs updates for the CTMC generator Q

Model summary
-------------
z(t) in {0, ..., K-1} follows a continuous-time Markov jump process with generator Q.

The observation model is diagonal Gaussian:

    x_i | y(t_i) ~ Normal(y(t_i), diag(tau_obs^2))

The latent transition density is evaluated by an NLEEstimator:

    p(y_{t+1} | y_t, z=k) = flow.log_prob(y_{t+1}, context=[theta_k, y_t, interval_length / dt])

The code keeps the implementation intentionally explicit and modular inside one class so
that each Gibbs step remains easy to inspect and modify.
"""

from __future__ import annotations

import heapq
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch
from torch.distributions import Distribution, constraints

class ContinuousTimeAR1HMMSampler:
    """
    Gibbs sampler for a continuous-time AR(1)-HMM with NLE transition density.

    Shapes
    ------
    x_obs: (N, obs_dim)
    T_obs: (N,)
    y_aug: (L+1, D)
    z_aug: (L+1,)
    theta: (K, theta_dim)       real-valued NLE parameter coordinates
    log_tau: (obs_dim,)         observation-noise log scale, not part of theta
    Q: (K, K)                   CTMC generator, sampled by Gamma conjugacy

    Conventions
    -----------
    - States are indexed from 0 to K-1.
    - `z_aug[j+1]` denotes the state on the interval [T_all[j], T_all[j+1]].
    - Therefore the NLE transition from y_aug[j] to y_aug[j+1] uses state z_aug[j+1].
    - `z_aug[0]` duplicates the first interval state so that `z_aug` has the same
      length as `T_all`.
    """

    def __init__(
        self,
        Q: torch.Tensor,
        x_obs: torch.Tensor | Sequence[torch.Tensor],
        obs_times: torch.Tensor | Sequence[torch.Tensor],
        T: float | Sequence[float],
        *,
        omega_scale: float = 1.5,
        nle_estimator: Any,
        y_nuts_config: Optional[Dict[str, Any]] = None,
        theta_nuts_config: Optional[Dict[str, Any]] = None,
        sir_config: Optional[Dict[str, Any]] = None,
        prior_config: Optional[Dict[str, Any]] = None,
        theta_prior: Optional[Distribution] = None,
        switching_parameter_mask: Optional[torch.Tensor] = None,
        fixed_parameter_mask: Optional[torch.Tensor] = None,
        observed_dims: Optional[torch.Tensor] = None,
        y0_prior_loc: float | torch.Tensor = 0.0,
        y0_prior_scale: float | torch.Tensor = 5.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        time_dtype: torch.dtype = torch.float64,
        seed: int = 0,
        latest_sample_path: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        Q:
            Generator matrix of shape (K, K). Rows must sum to zero.
        x_obs:
            Observation matrix of shape (N, D).
        obs_times:
            Observation times T_obs of shape (N,), with
            0 = T_obs[0] < ... < T_obs[N-1] = T.
        T:
            End time of the latent process.
        omega_scale:
            Uniformization rate factor. Omega = omega_scale * max_k(-Q_kk).
        nle_estimator:
            Trained/loaded NLEEstimator. Transition log densities are evaluated by
            `nle_estimator.transition_log_prob`.
        y_nuts_config / theta_nuts_config:
            Pyro NUTS settings used for the conditional updates.
        sir_config:
            Settings for SIR initialization of y at newly inserted candidate times.
        prior_config:
            Prior hyperparameters for theta and observation-noise variance.
            `theta_loc` and `theta_scale` define the legacy Normal prior in the
            NLE theta coordinate when `theta_prior` is omitted.
        theta_prior:
            Optional component-wise prior in the real-valued NLE theta coordinate.
            It must have batch shape `(theta_dim,)`, scalar event shape, and real
            support, and its log density must evaluate on the sampler's device
            and dtype. A physical-scale prior can be converted with
            `nle_estimator.dynamics.pullback_theta_prior()`.
        switching_parameter_mask:
            Boolean tensor of shape (theta_dim,). True dimensions have one
            parameter per regime; False dimensions are shared across regimes.
            If omitted, every parameter is regime-specific as before.
        fixed_parameter_mask:
            Boolean tensor of shape (theta_dim,). True dimensions are held at
            their `initial_theta` values and excluded from both NUTS and the
            theta prior. Fixed dimensions take precedence over switching/shared
            classification.
        observed_dims:
            Latent-state dimension indices represented by the columns of x_obs.
            Must have shape (obs_dim,). If omitted, all latent dimensions must be
            observed and x_obs must have D columns.
        y0_prior_loc / y0_prior_scale:
            Gaussian prior location and scale for y at the initial time. Scalars
            are broadcast across latent dimensions; vectors must have shape (D,).
        time_dtype:
            Floating-point dtype used for observation, jump, candidate, and
            augmented-grid times. Time differences are computed in this dtype and
            converted to the NLE dtype only after subtraction. Must be float32 or
            float64; float64 is the default to make exact time collisions negligible.
        latest_sample_path:
            Optional monitoring file overwritten atomically after every completed
            Gibbs sweep. The complete in-memory history is unchanged.
        """
        first_x_obs = x_obs[0] if isinstance(x_obs, (list, tuple)) else x_obs
        self.device = device or first_x_obs.device
        self.dtype = dtype
        if time_dtype not in (torch.float32, torch.float64):
            raise ValueError("time_dtype must be torch.float32 or torch.float64.")
        self.time_dtype = time_dtype
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)
        pyro.set_rng_seed(seed)
        self.latest_sample_path = (
            Path(latest_sample_path) if latest_sample_path is not None else None
        )

        self.Q = Q.to(device=self.device, dtype=self.dtype)
        self.omega_scale = float(omega_scale)
        if isinstance(x_obs, (list, tuple)):
            if not isinstance(obs_times, (list, tuple)):
                raise ValueError("obs_times must be a list/tuple when x_obs is a list/tuple.")
            if not isinstance(T, (list, tuple)):
                T = [float(times[-1].item()) for times in obs_times]
            if len(x_obs) != len(obs_times) or len(x_obs) != len(T):
                raise ValueError("x_obs, obs_times, and T must have the same number of series.")
            self.x_obs_list = [
                x.to(device=self.device, dtype=self.dtype) for x in x_obs
            ]
            self.T_obs_list = [
                times.to(device=self.device, dtype=self.time_dtype)
                for times in obs_times
            ]
            self.T_list = [
                torch.tensor(float(t), device=self.device, dtype=self.time_dtype)
                for t in T
            ]
        else:
            if isinstance(obs_times, (list, tuple)):
                raise ValueError("obs_times must be a tensor when x_obs is a tensor.")
            if isinstance(T, (list, tuple)):
                raise ValueError("T must be a scalar when x_obs is a tensor.")
            self.x_obs_list = [x_obs.to(device=self.device, dtype=self.dtype)]
            self.T_obs_list = [
                obs_times.to(device=self.device, dtype=self.time_dtype)
            ]
            self.T_list = [
                torch.tensor(float(T), device=self.device, dtype=self.time_dtype)
            ]

        self.S = len(self.x_obs_list)
        self.K = int(self.Q.shape[0])
        self.obs_dim = int(self.x_obs_list[0].shape[1])
        self.N_list = [int(x.shape[0]) for x in self.x_obs_list]
        self.nle_estimator = nle_estimator
        self.flow_model = nle_estimator.estimator
        if self.flow_model is None:
            raise ValueError("nle_estimator.estimator must be trained/loaded before sampling.")
        self.theta_dim = int(nle_estimator.dynamics.theta_dim)
        self.D = int(nle_estimator.dynamics.x_dim)
        self.y0_prior_loc = torch.as_tensor(
            y0_prior_loc, dtype=self.dtype, device=self.device
        ).broadcast_to((self.D,))
        self.y0_prior_scale = torch.as_tensor(
            y0_prior_scale, dtype=self.dtype, device=self.device
        ).broadcast_to((self.D,))
        if torch.any(self.y0_prior_scale <= 0):
            raise ValueError("y0_prior_scale entries must be positive.")
        if observed_dims is None:
            if self.obs_dim != self.D:
                raise ValueError(
                    "observed_dims is required when x_obs does not contain every "
                    f"latent dimension: obs_dim={self.obs_dim}, latent_dim={self.D}."
                )
            observed_dims = torch.arange(
                self.D, dtype=torch.long, device=self.device
            )
        else:
            observed_dims = torch.as_tensor(
                observed_dims, dtype=torch.long, device=self.device
            )
        for series_idx, x_s in enumerate(self.x_obs_list):
            if x_s.ndim != 2:
                raise ValueError(f"x_obs for series {series_idx} must be two-dimensional.")
            if x_s.shape[1] != self.obs_dim:
                raise ValueError("All series must have the same observed dimension.")
        if observed_dims.shape != (self.obs_dim,):
            raise ValueError(
                f"observed_dims must have shape ({self.obs_dim},), "
                f"got {tuple(observed_dims.shape)}."
            )
        if torch.any(observed_dims < 0) or torch.any(observed_dims >= self.D):
            raise ValueError(
                f"observed_dims entries must be in [0, {self.D - 1}]."
            )
        if torch.unique(observed_dims).numel() != observed_dims.numel():
            raise ValueError("observed_dims entries must be unique.")
        self.observed_dims = observed_dims
        if switching_parameter_mask is None:
            switching_parameter_mask = torch.ones(
                self.theta_dim, dtype=torch.bool, device=self.device
            )
        else:
            switching_parameter_mask = torch.as_tensor(
                switching_parameter_mask, dtype=torch.bool, device=self.device
            )
        if switching_parameter_mask.shape != (self.theta_dim,):
            raise ValueError(
                "switching_parameter_mask must have shape "
                f"({self.theta_dim},), got {tuple(switching_parameter_mask.shape)}."
            )
        if fixed_parameter_mask is None:
            fixed_parameter_mask = torch.zeros(
                self.theta_dim, dtype=torch.bool, device=self.device
            )
        else:
            fixed_parameter_mask = torch.as_tensor(
                fixed_parameter_mask, dtype=torch.bool, device=self.device
            )
        if fixed_parameter_mask.shape != (self.theta_dim,):
            raise ValueError(
                "fixed_parameter_mask must have shape "
                f"({self.theta_dim},), got {tuple(fixed_parameter_mask.shape)}."
            )
        self.fixed_parameter_mask = fixed_parameter_mask
        self.switching_parameter_mask = (
            switching_parameter_mask & ~fixed_parameter_mask
        )
        self.shared_parameter_mask = (
            ~switching_parameter_mask & ~fixed_parameter_mask
        )
        self.dynamics_dt = float(nle_estimator.dynamics.dt)
        if self.dynamics_dt <= 0.0:
            raise ValueError("nle_estimator.dynamics.dt must be positive.")
        if hasattr(self.flow_model, "to"):
            self.flow_model.to(self.device)
        if hasattr(self.flow_model, "eval"):
            self.flow_model.eval()
        if hasattr(self.flow_model, "requires_grad_"):
            # The trained NLE is a fixed density inside Gibbs inference.  Keeping
            # its weights differentiable makes PyTorch retain unnecessary
            # autograd tensors across repeated one-step NUTS runs.  Freezing the
            # weights still permits gradients with respect to y and theta, which
            # are the variables sampled by NUTS.
            self.flow_model.requires_grad_(False)

        if self.Q.shape != (self.K, self.K):
            raise ValueError("Q must be square with shape (K, K).")
        if not torch.allclose(self.Q.sum(dim=1), torch.zeros(self.K, dtype=self.dtype, device=self.device), atol=1e-8):
            raise ValueError("Each row of Q must sum to zero.")
        for series_idx, (T_obs_s, T_s, N_s) in enumerate(
            zip(self.T_obs_list, self.T_list, self.N_list)
        ):
            if T_obs_s.ndim != 1 or T_obs_s.shape[0] != N_s:
                raise ValueError(f"obs_times for series {series_idx} must have shape (N_s,).")
            if N_s < 2:
                raise ValueError("Each series must contain at least the endpoints 0 and T_s.")
            if not torch.all(T_obs_s[1:] > T_obs_s[:-1]):
                raise ValueError(f"obs_times for series {series_idx} must be strictly increasing.")
            if abs(float(T_obs_s[0].item())) > 1e-10:
                raise ValueError(f"obs_times for series {series_idx} must start at 0.")
            if abs(float(T_obs_s[-1].item()) - float(T_s.item())) > 1e-10:
                raise ValueError(f"obs_times for series {series_idx} must end at T_s.")

        self.omega = 0.0
        self.B = torch.empty_like(self.Q)
        self._refresh_uniformization()

        self.y_nuts_config = {
            "max_tree_depth": 4,
            "target_accept_prob": 0.8,
        }
        if y_nuts_config is not None:
            self.y_nuts_config.update(y_nuts_config)

        self.theta_nuts_config = {
            "max_tree_depth": 4,
            "target_accept_prob": 0.8,
            "warmup_steps": 0,
            "num_samples": 1,
        }
        if theta_nuts_config is not None:
            self.theta_nuts_config.update(theta_nuts_config)

        self.sir_config = {
            "num_particles": 64,
        }
        if sir_config is not None:
            self.sir_config.update(sir_config)

        self.prior_config = {
            "theta_loc": 0.0,
            "theta_scale": 1.0,
            # tau_obs[d]^2 ~ InvGamma(tau2_alpha, tau2_beta), independently by d.
            "tau2_alpha": 2.0,
            "tau2_beta": 0.1,
            # q_ij ~ Gamma(q_alpha, q_beta) for i != j, using rate parameterization.
            "q_alpha": 1.0,
            "q_beta": 1.0,
        }
        if prior_config is not None:
            self.prior_config.update(prior_config)
        self.theta_prior = theta_prior
        if self.theta_prior is not None:
            self._validate_explicit_theta_prior()

        self.history: Dict[str, List[Any]] = {
            "y_aug": [],
            "z_aug": [],
            "T_all": [],
            "theta": [],
            "log_tau": [],
            "Q": [],
            "omega": [],
            "T_true": [],
            "z_true": [],
            "diagnostics": [],
        }

        self.theta: Optional[torch.Tensor] = None
        self.fixed_theta: Optional[torch.Tensor] = None
        self.log_tau: Optional[torch.Tensor] = None
        self.T_true_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.z_true_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.T_all_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.z_aug_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.is_event_time_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.obs_idx_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.true_idx_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.pseudo_idx_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.y_aug_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.initial_y_times_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.initial_y_values_list: List[Optional[torch.Tensor]] = [None for _ in range(self.S)]
        self.initial_state_probs = torch.full((self.K,), 1.0 / self.K, dtype=self.dtype, device=self.device)

    def initialize(
        self,
        *,
        initial_theta: Optional[torch.Tensor] = None,
        initial_log_tau: Optional[torch.Tensor] = None,
        initial_state_probs: Optional[torch.Tensor] = None,
        initial_y_times: Optional[torch.Tensor] = None,
        initial_y_values: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize theta, observation noise, and the true discrete path.

        `initial_theta` is expressed in the real-valued NLE parameter coordinate.
        Pass observation noise separately as `initial_log_tau`.

        The initial true path has no jumps. Its single state is sampled from
        `initial_state_probs`, unless that probability vector is overridden here.
        """
        if initial_state_probs is not None:
            probs = initial_state_probs.to(device=self.device, dtype=self.dtype)
            self.initial_state_probs = probs / probs.sum()

        if (initial_y_times is None) != (initial_y_values is None):
            raise ValueError(
                "initial_y_times and initial_y_values must be provided together."
            )
        if initial_y_times is not None and initial_y_values is not None:
            initial_y_times = initial_y_times.to(
                device=self.device, dtype=self.time_dtype
            )
            initial_y_values = initial_y_values.to(
                device=self.device, dtype=self.dtype
            )
            if initial_y_times.ndim != 1:
                raise ValueError("initial_y_times must be one-dimensional.")
            if initial_y_values.shape != (initial_y_times.shape[0], self.D):
                raise ValueError(
                    "initial_y_values must have shape "
                    f"({initial_y_times.shape[0]}, {self.D})."
                )
            if not torch.all(initial_y_times[1:] > initial_y_times[:-1]):
                raise ValueError("initial_y_times must be strictly increasing.")
            for series_idx, (T_obs_s, T_s) in enumerate(zip(self.T_obs_list, self.T_list)):
                if initial_y_times[0] > T_obs_s[0] or initial_y_times[-1] < T_s:
                    raise ValueError(
                        "initial_y_times must cover the complete inference interval "
                        f"for series {series_idx}."
                    )
        prior_config = self.prior_config
        if initial_theta is None:
            if self.theta_prior is None:
                theta_loc, theta_scale = self._theta_prior_parameters()
                theta_value = dist.Normal(theta_loc, theta_scale).sample((self.K,))
            else:
                theta_value = self.theta_prior.sample((self.K,))
            # Shared dimensions represent one random variable, not K independent draws.
            theta_value[:, self.shared_parameter_mask] = theta_value[
                0, self.shared_parameter_mask
            ]
        else:
            theta_value = initial_theta

        if initial_log_tau is None:
            tau2_alpha = torch.as_tensor(
                prior_config["tau2_alpha"], dtype=self.dtype, device=self.device
            ).broadcast_to((self.obs_dim,))
            tau2_beta = torch.as_tensor(
                prior_config["tau2_beta"], dtype=self.dtype, device=self.device
            ).broadcast_to((self.obs_dim,))
            precision = dist.Gamma(tau2_alpha, tau2_beta).sample()
            initial_log_tau = 0.5 * torch.log(precision.reciprocal().clamp_min(1e-16))

        self.theta = theta_value.to(device=self.device, dtype=self.dtype).clone()
        self.fixed_theta = self.theta[:, self.fixed_parameter_mask].clone()
        self.log_tau = initial_log_tau.to(device=self.device, dtype=self.dtype).clone()
        if self.theta.shape != (self.K, self.theta_dim):
            raise ValueError(f"theta must have shape ({self.K}, {self.theta_dim}).")
        self._validate_theta_structure(self.theta)
        if self.log_tau.shape != (self.obs_dim,):
            raise ValueError(
                f"log_tau must have shape ({self.obs_dim},)."
            )

        for s in range(self.S):
            initial_state = dist.Categorical(probs=self.initial_state_probs).sample()
            self.T_true_list[s] = torch.empty(
                0, dtype=self.time_dtype, device=self.device
            )
            self.z_true_list[s] = initial_state.reshape(1).to(
                dtype=torch.long, device=self.device
            )
            self.T_all_list[s] = None
            self.z_aug_list[s] = None
            self.is_event_time_list[s] = None
            self.obs_idx_list[s] = None
            self.true_idx_list[s] = None
            self.pseudo_idx_list[s] = None
            self.y_aug_list[s] = None
            # A tensor-valued initial path is shared across series by default.
            # Multi-series callers that need different initial paths can assign
            # initial_y_times_list / initial_y_values_list before run().
            self.initial_y_times_list[s] = initial_y_times
            self.initial_y_values_list[s] = initial_y_values

    def _theta_prior_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return legacy NLE-coordinate Normal parameters of shape (theta_dim,)."""
        theta_loc = torch.as_tensor(
            self.prior_config["theta_loc"], dtype=self.dtype, device=self.device
        ).broadcast_to((self.theta_dim,))
        theta_scale = torch.as_tensor(
            self.prior_config["theta_scale"], dtype=self.dtype, device=self.device
        ).broadcast_to((self.theta_dim,))
        if torch.any(theta_scale <= 0):
            raise ValueError("theta prior scales must be positive.")
        return theta_loc, theta_scale

    def _validate_explicit_theta_prior(self) -> None:
        """Validate the component-wise prior supplied in the NLE coordinate."""
        if not isinstance(self.theta_prior, Distribution):
            raise TypeError("theta_prior must be a torch Distribution.")
        expected_batch_shape = torch.Size((self.theta_dim,))
        if self.theta_prior.batch_shape != expected_batch_shape:
            raise ValueError(
                "theta_prior must have batch_shape "
                f"{tuple(expected_batch_shape)}, got "
                f"{tuple(self.theta_prior.batch_shape)}."
            )
        if self.theta_prior.event_shape != torch.Size():
            raise ValueError(
                "theta_prior must be component-wise with an empty event_shape; "
                "do not wrap it in Independent."
            )
        if self.theta_prior.support != constraints.real:
            raise ValueError(
                "theta_prior must have real support in the NLE theta coordinate. "
                "Use dynamics.pullback_theta_prior() for a constrained physical prior."
            )
        probe = torch.zeros(
            self.theta_dim,
            dtype=self.dtype,
            device=self.device,
        )
        try:
            probe_log_prob = self.theta_prior.log_prob(probe)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                "theta_prior must be constructed on a device compatible with the sampler."
            ) from error
        if probe_log_prob.shape != expected_batch_shape:
            raise ValueError(
                "theta_prior.log_prob(theta) must return one value per theta dimension."
            )
        if (
            probe_log_prob.device != self.device
            or probe_log_prob.dtype != self.dtype
        ):
            raise ValueError(
                "theta_prior.log_prob(theta) must use the sampler's device and dtype "
                f"({self.device}, {self.dtype}); got "
                f"({probe_log_prob.device}, {probe_log_prob.dtype})."
            )

    def _validate_theta_structure(self, theta: torch.Tensor) -> None:
        """Validate shared columns and the values retained for fixed columns."""
        if torch.any(self.shared_parameter_mask):
            shared = theta[:, self.shared_parameter_mask]
            if not torch.allclose(shared, shared[0].expand_as(shared)):
                raise ValueError(
                    "initial_theta must be identical across regimes for shared "
                    "parameter dimensions."
                )
        if (
            torch.any(self.fixed_parameter_mask)
            and self.fixed_theta is not None
            and not torch.allclose(
                theta[:, self.fixed_parameter_mask], self.fixed_theta
            )
        ):
            raise ValueError(
                "Fixed theta dimensions differ from their initialized values."
            )

    def _pack_theta(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pack expanded theta (K, P) into the non-redundant NUTS variables."""
        packed: Dict[str, torch.Tensor] = {}
        if torch.any(self.shared_parameter_mask):
            packed["theta_shared"] = theta[0, self.shared_parameter_mask].clone()
        if torch.any(self.switching_parameter_mask):
            packed["theta_switching"] = theta[
                :, self.switching_parameter_mask
            ].clone()
        return packed

    def _expand_theta(self, packed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Expand non-redundant NUTS variables to theta with shape (K, P)."""
        columns: List[torch.Tensor] = []
        shared_index = 0
        switching_index = 0
        fixed_index = 0
        for parameter_index in range(self.theta_dim):
            if self.fixed_parameter_mask[parameter_index]:
                if self.fixed_theta is None:
                    raise RuntimeError("Call initialize() before expanding theta.")
                column = self.fixed_theta[:, fixed_index]
                fixed_index += 1
            elif self.switching_parameter_mask[parameter_index]:
                column = packed["theta_switching"][:, switching_index]
                switching_index += 1
            else:
                column = packed["theta_shared"][shared_index].expand(self.K)
                shared_index += 1
            columns.append(column)
        return torch.stack(columns, dim=1)

    def _theta_log_prior(self, packed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate each unique theta prior exactly once."""
        if self.theta_prior is not None:
            theta = self._expand_theta(packed)
            elementwise_logp = self.theta_prior.log_prob(theta)
            expected_shape = torch.Size((self.K, self.theta_dim))
            if elementwise_logp.shape != expected_shape:
                raise RuntimeError(
                    "theta_prior.log_prob(theta) returned shape "
                    f"{tuple(elementwise_logp.shape)}, expected {tuple(expected_shape)}."
                )
            logp = torch.zeros((), dtype=self.dtype, device=self.device)
            if torch.any(self.shared_parameter_mask):
                logp = logp + elementwise_logp[
                    0, self.shared_parameter_mask
                ].sum()
            if torch.any(self.switching_parameter_mask):
                logp = logp + elementwise_logp[
                    :, self.switching_parameter_mask
                ].sum()
            return logp

        theta_loc, theta_scale = self._theta_prior_parameters()
        logp = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if torch.any(self.shared_parameter_mask):
            logp = logp + dist.Normal(
                theta_loc[self.shared_parameter_mask],
                theta_scale[self.shared_parameter_mask],
            ).log_prob(packed["theta_shared"]).sum()
        if torch.any(self.switching_parameter_mask):
            logp = logp + dist.Normal(
                theta_loc[self.switching_parameter_mask],
                theta_scale[self.switching_parameter_mask],
            ).log_prob(packed["theta_switching"]).sum()
        return logp

    def _iter_merged_time_sources(
        self,
        s: int,
        *,
        context: str,
        sources: Sequence[Tuple[str, torch.Tensor]],
    ) -> Iterator[Tuple[str, int, float]]:
        """Yield tagged, strictly increasing times from sorted source tensors."""

        def tagged_records(
            source_name: str,
            times: torch.Tensor,
        ) -> Iterator[Tuple[float, str, int]]:
            for position, time in enumerate(times):
                yield (
                    time.item(),
                    source_name,
                    position,
                )

        streams = []
        for source_name, source_times in sources:
            times = source_times.to(
                device=self.device,
                dtype=self.time_dtype,
            )
            if times.ndim != 1:
                raise ValueError(
                    f"{source_name} times for series {s + 1} must be one-dimensional."
                )
            if not torch.all(torch.isfinite(times)):
                raise ValueError(
                    f"{source_name} times for series {s + 1} must be finite."
                )
            if times.numel() > 1 and torch.any(times[1:] <= times[:-1]):
                bad_position = int(
                    torch.nonzero(
                        times[1:] <= times[:-1], as_tuple=False
                    )[0].item()
                )
                raise RuntimeError(
                    f"{source_name} times for series {s + 1} must be strictly "
                    f"increasing; positions {bad_position} and {bad_position + 1} "
                    f"contain {float(times[bad_position].item())} and "
                    f"{float(times[bad_position + 1].item())}."
                )
            streams.append(tagged_records(source_name, times))

        previous_time: Optional[float] = None
        previous_source: Optional[str] = None
        merged = heapq.merge(*streams, key=lambda record: record[0])
        for time_value, source_name, position in merged:
            if previous_time is not None and time_value == previous_time:
                raise RuntimeError(
                    f"Duplicate time {time_value} in {context} merge for series "
                    f"{s + 1}: sources {previous_source} and {source_name}."
                )
            previous_time = time_value
            previous_source = source_name
            yield source_name, position, time_value

    def build_augmented_grid(
        self,
        s: int,
        T_cand: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return augmented-grid arrays for T_all = T_true U T_cand U T_obs."""
        T_true = self.T_true_list[s]
        z_true = self.z_true_list[s]
        T_obs = self.T_obs_list[s]
        N = self.N_list[s]
        if T_true is None or z_true is None:
            raise RuntimeError("Call initialize() before build_augmented_grid().")

        T_true = T_true.to(device=self.device, dtype=self.time_dtype)
        T_cand = T_cand.to(device=self.device, dtype=self.time_dtype)
        self.T_true_list[s] = T_true

        T_all_values: List[float] = []
        is_event_values: List[bool] = []
        obs_idx_values: List[int] = []
        cand_idx_values: List[int] = []

        sources = (
            ("T_true", T_true),
            ("T_cand", T_cand),
            ("T_obs", T_obs),
        )
        for idx_all, (source_name, _, time_value) in enumerate(
            self._iter_merged_time_sources(
                s,
                context="augmented-grid",
                sources=sources,
            )
        ):
            T_all_values.append(time_value)
            is_event_values.append(source_name in ("T_true", "T_cand"))
            if source_name == "T_cand":
                cand_idx_values.append(idx_all)
            elif source_name == "T_obs":
                obs_idx_values.append(idx_all)

        T_all = torch.tensor(
            T_all_values,
            dtype=self.time_dtype,
            device=self.device,
        )
        is_event_time = torch.tensor(is_event_values, dtype=torch.bool, device=self.device)
        obs_idx = torch.tensor(obs_idx_values, dtype=torch.long, device=self.device)
        cand_idx = torch.tensor(cand_idx_values, dtype=torch.long, device=self.device)
        if obs_idx.shape[0] != N:
            raise RuntimeError("Observation time was not found on the merged grid.")
        if cand_idx.shape[0] != T_cand.shape[0]:
            raise RuntimeError("Candidate jump time was not found on the merged grid.")

        z_aug = self._expand_true_path_onto_grid(s, T_all)
        return T_all, z_aug, is_event_time, obs_idx, cand_idx

    def _expand_true_path_onto_grid(
        self,
        s: int,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Return point states with z[j] assigned to interval [times[j-1], times[j]]."""
        T_true = self.T_true_list[s]
        z_true = self.z_true_list[s]
        if T_true is None or z_true is None:
            raise RuntimeError("Call initialize() before assigning states to a grid.")
        if times.ndim != 1 or times.shape[0] < 2:
            raise ValueError("times must be one-dimensional and include 0 and T.")

        # Point-state representation on T_all.
        # z_aug[j+1] is the regime on [T_all[j], T_all[j+1]], and z_aug[0]
        # duplicates the first interval state. z_true is not padded: z_true[r] is
        # the regime on true-path segment r.
        interval_states = torch.empty(
            times.shape[0] - 1, dtype=torch.long, device=self.device
        )
        T_true_idx = 0
        for j in range(times.shape[0] - 1):
            # time at the left of the interval
            left = times[j]
            # T_true_idx is the number of true jumps at or before `left`.
            # The matching true-path state is z_true[T_true_idx].
            while T_true_idx < T_true.shape[0] and T_true[T_true_idx] <= left:
                T_true_idx += 1
            interval_states[j] = z_true[T_true_idx]

        z_aug = torch.empty(times.shape[0], dtype=torch.long, device=self.device)
        # To fit z_aug indices with y_aug indices, duplicate the first interval state
        z_aug[0] = interval_states[0]
        z_aug[1:] = interval_states
        return z_aug

    def _build_sir_work_grid(
        self,
        s: int,
        T_cand: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Four-way merge T_obs, T_true, old T_pseudo, and new T_cand.

        Returns the temporary union grid, a mask identifying old pseudo points,
        and a mask identifying new candidate points that require SIR sampling.
        """
        old_T_all = self.T_all_list[s]
        old_pseudo_idx = self.pseudo_idx_list[s]
        T_true = self.T_true_list[s]
        if any(
            value is None
            for value in (
                old_T_all,
                old_pseudo_idx,
                T_true,
            )
        ):
            raise RuntimeError("The previous augmented grid is incomplete.")

        T_obs = self.T_obs_list[s]
        old_T_all = old_T_all.to(device=self.device, dtype=self.time_dtype)
        T_true = T_true.to(device=self.device, dtype=self.time_dtype)
        T_cand = T_cand.to(device=self.device, dtype=self.time_dtype)
        self.T_all_list[s] = old_T_all
        self.T_true_list[s] = T_true
        T_pseudo = old_T_all[old_pseudo_idx]
        source_times = (
            ("T_obs", T_obs),
            ("T_true", T_true),
            ("T_pseudo", T_pseudo),
            ("T_cand", T_cand),
        )

        times_for_sir: List[float] = []
        is_old_pseudo: List[bool] = []
        is_new_candidate: List[bool] = []

        for source_name, _, time_value in self._iter_merged_time_sources(
            s,
            context="SIR work-grid",
            sources=source_times,
        ):
            times_for_sir.append(time_value)
            is_old_pseudo.append(source_name == "T_pseudo")
            is_new_candidate.append(source_name == "T_cand")

        return (
            torch.tensor(
                times_for_sir,
                dtype=self.time_dtype,
                device=self.device,
            ),
            torch.tensor(is_old_pseudo, dtype=torch.bool, device=self.device),
            torch.tensor(is_new_candidate, dtype=torch.bool, device=self.device),
        )

    def add_candidate_jumps(self, s: int) -> torch.Tensor:
        """
        Add candidate jumps to the current true path using uniformization.

        On an interval with state k and duration dt, candidate jumps are sampled from:
            Poisson((Omega - q_k) dt) = Poisson((Omega + Q_kk) dt)
        because q_k = -Q_kk.
        """
        T_true = self.T_true_list[s]
        z_true = self.z_true_list[s]
        T = self.T_list[s]
        if T_true is None or z_true is None:
            raise RuntimeError("Call initialize() before add_candidate_jumps().")

        cand_times: List[torch.Tensor] = []
        for j, state in enumerate(z_true.tolist()):
            # state is the regime on the interval between t0 and t1
            t0 = 0.0 if j == 0 else T_true[j - 1].item()
            t1 = T.item() if j == z_true.shape[0] - 1 else T_true[j].item()
            dt = t1 - t0
            if dt <= 0.0:
                raise RuntimeError("dt <= 0 encountered when adding candidate jumps.")

            rate = self.omega + float(self.Q[state, state].item())
            if rate <= 0.0:
                raise RuntimeError("omega must be greater than max_k(-Q_kk) to ensure a positive candidate jump rate.")

            num_candidate = int(
                torch.poisson(
                    torch.tensor(rate * dt, dtype=self.time_dtype),
                    generator=self.rng,
                ).item()
            )
            if num_candidate == 0:
                continue

            u = torch.rand(
                num_candidate,
                generator=self.rng,
                dtype=self.time_dtype,
            )
            times = t0 + dt * u
            cand_times.append(times.to(device=self.device))
        if not cand_times:
            return torch.empty(
                0, dtype=self.time_dtype, device=self.device
            )

        # Do not silently remove exact duplicates. The source-aware merge checks
        # them as invariant violations and reports both colliding sources.
        return torch.sort(torch.cat(cand_times)).values

    def _delta_to_n_steps(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous elapsed time to the discrete NLE conditioning step count.

        The NLE was trained on simulator steps of size `dynamics.dt`, so an
        interval Delta is mapped to the continuous step count Delta / dt.
        """
        if not torch.all(torch.isfinite(delta)):
            raise ValueError("Transition time differences must be finite.")
        if torch.any(delta <= 0):
            smallest = float(delta.min().detach().cpu().item())
            raise RuntimeError(
                "Transition time differences must be strictly positive; "
                f"smallest delta is {smallest}."
            )
        return delta / self.dynamics_dt

    def observation_logprob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        log_tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diagonal Gaussian log density on the selected observed dimensions."""
        if log_tau is None:
            log_tau = self.log_tau
        tau = torch.exp(log_tau)
        return dist.Normal(y[..., self.observed_dims], tau).log_prob(x).sum()

    def logprob_y_given_z_theta(
        self,
        y_aug: torch.Tensor,
        z_aug: torch.Tensor,
        T_all: torch.Tensor,
        obs_idx: torch.Tensor,
        x_obs: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        log_tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Conditional log density:

            log p(y_aug | z_aug, theta, log_tau, x)
              = log p(y_0 | z_0, theta)
              + sum_l log p(y_l | y_{l-1}, z_l, Delta_l, theta)
              + sum_i log p(x_i | y_{m(i)}, log_tau)
        """
        theta = self.theta if theta is None else theta
        if theta is None:
            raise RuntimeError("Sampler theta has not been initialized.")
        if log_tau is None:
            log_tau = self.log_tau
        if x_obs is None:
            x_obs = self.x_obs_list[0]
        logp = dist.Normal(
            self.y0_prior_loc,
            self.y0_prior_scale,
        ).log_prob(y_aug[0]).sum()
        logp = logp + self._compute_log_emission_given_z(y_aug, z_aug, T_all, theta)

        y_at_obs = y_aug[obs_idx][:, self.observed_dims]  # (N, obs_dim)
        tau = torch.exp(log_tau)
        logp = logp + dist.Normal(y_at_obs, tau).log_prob(x_obs).sum()
        return logp

    def logprob_theta_given_y_z(
        self,
        theta: torch.Tensor,
        y_aug: torch.Tensor,
        z_aug: torch.Tensor,
        T_all: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional log density:

            log p(theta | y, z)
              = log p(theta)
              + sum_l log p(y_l | y_{l-1}, z_l, Delta_l, theta)

        The observation model x | y, log_tau is constant with respect to
        theta, so it is intentionally omitted here. Observation noise is updated
        separately by sample_log_tau().
        """
        self._validate_theta_structure(theta)
        logp = self._theta_log_prior(self._pack_theta(theta))
        logp = logp + self._compute_log_emission_given_z(y_aug, z_aug, T_all, theta)
        return logp

    def _logprob_theta_all_series(self, theta: torch.Tensor) -> torch.Tensor:
        """Shared-theta conditional log density summed over series s=1,...,S."""
        self._validate_theta_structure(theta)
        logp = self._theta_log_prior(self._pack_theta(theta))
        for s in range(self.S):
            y_aug_s = self.y_aug_list[s]
            z_aug_s = self.z_aug_list[s]
            T_all_s = self.T_all_list[s]
            if y_aug_s is None or z_aug_s is None or T_all_s is None:
                raise RuntimeError("All series must have y/z/T_all before theta update.")
            logp = logp + self._compute_log_emission_given_z(
                y_aug_s,
                z_aug_s,
                T_all_s,
                theta,
            )
        return logp

    @torch.no_grad()
    def complete_data_log_joint(self) -> Dict[str, Any]:
        """Evaluate a component-wise complete-data log joint for diagnostics.

        The latent path is scored on the canonical grid formed by observation
        times and retained CTMC jump times.  Virtual uniformization candidates
        are deliberately excluded, so this diagnostic does not change merely
        because a sweep happened to introduce more auxiliary grid points.

        The density is with respect to ``tau**2`` (the model parameter having an
        inverse-gamma prior), rather than with respect to ``log_tau``.  Hence no
        change-of-variables Jacobian for ``log_tau`` is included.

        The theta-prior component is with respect to the real-valued NLE
        coordinate. If ``theta_prior`` was pulled back from physical scale, its
        ``log_prob`` already contains the corresponding Jacobian.
        """
        if self.theta is None or self.log_tau is None:
            raise RuntimeError(
                "Call initialize() and complete a sweep before evaluating the joint."
            )

        theta_prior = self._theta_log_prior(self._pack_theta(self.theta))

        tau2 = torch.exp(2.0 * self.log_tau)
        tau2_alpha = torch.as_tensor(
            self.prior_config["tau2_alpha"],
            dtype=self.dtype,
            device=self.device,
        ).broadcast_to(tau2.shape)
        tau2_beta = torch.as_tensor(
            self.prior_config["tau2_beta"],
            dtype=self.dtype,
            device=self.device,
        ).broadcast_to(tau2.shape)
        tau2_prior = dist.InverseGamma(tau2_alpha, tau2_beta).log_prob(tau2).sum()

        q_alpha = torch.as_tensor(
            self.prior_config["q_alpha"],
            dtype=self.dtype,
            device=self.device,
        )
        q_beta = torch.as_tensor(
            self.prior_config["q_beta"],
            dtype=self.dtype,
            device=self.device,
        )
        if q_alpha.ndim != 0 or q_beta.ndim != 0:
            raise ValueError("q_alpha and q_beta must be scalars.")
        off_diagonal = ~torch.eye(
            self.K, dtype=torch.bool, device=self.device
        )
        Q_prior = dist.Gamma(q_alpha, q_beta).log_prob(
            self.Q[off_diagonal]
        ).sum()

        initial_state = torch.zeros((), dtype=self.dtype, device=self.device)
        ctmc_path = torch.zeros((), dtype=self.dtype, device=self.device)
        latent_initial = torch.zeros((), dtype=self.dtype, device=self.device)
        nle_transition = torch.zeros((), dtype=self.dtype, device=self.device)
        observation = torch.zeros((), dtype=self.dtype, device=self.device)
        num_latent_transitions = 0
        num_observation_rows = 0
        num_observation_values = 0
        num_true_jumps = 0

        for s in range(self.S):
            T_all_s = self.T_all_list[s]
            z_aug_s = self.z_aug_list[s]
            y_aug_s = self.y_aug_list[s]
            obs_idx_s = self.obs_idx_list[s]
            true_idx_s = self.true_idx_list[s]
            T_true_s = self.T_true_list[s]
            z_true_s = self.z_true_list[s]
            if any(
                value is None
                for value in (
                    T_all_s,
                    z_aug_s,
                    y_aug_s,
                    obs_idx_s,
                    true_idx_s,
                    T_true_s,
                    z_true_s,
                )
            ):
                raise RuntimeError(
                    "Complete a Gibbs sweep before evaluating the complete-data joint."
                )

            canonical_idx = torch.unique(
                torch.cat([obs_idx_s, true_idx_s]), sorted=True
            )
            if canonical_idx.numel() < 2:
                raise RuntimeError(
                    "The canonical grid must contain at least the interval endpoints."
                )
            canonical_times = T_all_s[canonical_idx]
            canonical_y = y_aug_s[canonical_idx]
            right_idx = canonical_idx[1:]
            interval_states = z_aug_s[right_idx]
            delta = canonical_times[1:] - canonical_times[:-1]

            latent_initial = latent_initial + dist.Normal(
                self.y0_prior_loc,
                self.y0_prior_scale,
            ).log_prob(canonical_y[0]).sum()
            nle_transition = nle_transition + self._evaluate_batched_transition_logprobs(
                y_prev_batch=canonical_y[:-1],
                y_curr_batch=canonical_y[1:],
                theta_batch=self.theta[interval_states],
                delta_batch=delta,
            ).sum()
            observation = observation + self.observation_logprob(
                self.x_obs_list[s],
                y_aug_s[obs_idx_s],
                self.log_tau,
            )

            initial_state = initial_state + torch.log(
                self.initial_state_probs[z_true_s[0]]
            )
            path_boundaries = torch.cat(
                [
                    torch.zeros(
                        1, dtype=self.time_dtype, device=self.device
                    ),
                    T_true_s,
                    self.T_list[s].reshape(1),
                ]
            )
            dwell_times = path_boundaries[1:] - path_boundaries[:-1]
            ctmc_path = ctmc_path + (
                self.Q[z_true_s, z_true_s] * dwell_times
            ).sum()
            if T_true_s.numel() > 0:
                ctmc_path = ctmc_path + torch.log(
                    self.Q[z_true_s[:-1], z_true_s[1:]]
                ).sum()

            num_latent_transitions += int(canonical_idx.numel() - 1)
            num_observation_rows += int(self.x_obs_list[s].shape[0])
            num_observation_values += int(self.x_obs_list[s].numel())
            num_true_jumps += int(T_true_s.numel())

        components = {
            "theta_prior": float(theta_prior.item()),
            "tau2_prior": float(tau2_prior.item()),
            "Q_prior": float(Q_prior.item()),
            "initial_state": float(initial_state.item()),
            "ctmc_path": float(ctmc_path.item()),
            "latent_initial": float(latent_initial.item()),
            "nle_transition": float(nle_transition.item()),
            "observation": float(observation.item()),
        }
        total = sum(components.values())
        return {
            "definition": "complete_data_on_observation_and_true_jump_grid",
            "theta_measure": "nle_coordinate",
            "tau_measure": "tau_squared",
            "components": components,
            "total": total,
            "counts": {
                "num_series": self.S,
                "num_latent_transitions": num_latent_transitions,
                "num_observation_rows": num_observation_rows,
                "num_observation_values": num_observation_values,
                "num_true_jumps": num_true_jumps,
            },
            "nle_transition_mean": (
                components["nle_transition"] / num_latent_transitions
                if num_latent_transitions
                else float("nan")
            ),
            "observation_mean": (
                components["observation"] / num_observation_values
                if num_observation_values
                else float("nan")
            ),
        }

    def _latest_sample_payload(self) -> Dict[str, Any]:
        """Build the monitoring payload from the most recently stored sweep."""
        if not self.history["diagnostics"]:
            raise RuntimeError("No completed sweep is available to save.")
        sample_keys = (
            "y_aug",
            "z_aug",
            "T_all",
            "T_true",
            "z_true",
            "theta",
            "log_tau",
            "Q",
            "omega",
        )
        return {
            "schema_version": 1,
            "completed_sweeps": len(self.history["diagnostics"]),
            "sample": {key: self.history[key][-1] for key in sample_keys},
            "diagnostics": self.history["diagnostics"][-1],
        }

    def _save_latest_sample(self) -> None:
        """Atomically overwrite the externally readable latest-sample file."""
        target = self.latest_sample_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("wb") as handle:
                torch.save(self._latest_sample_payload(), handle)
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)

    def sample_y_nuts(
        self,
        s: int,
        y_init: torch.Tensor,
        max_tree_depth: Optional[int] = None,
        target_accept_prob: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample the full augmented latent path y_aug with one-step Pyro NUTS."""
        T_all = self.T_all_list[s]
        z_aug = self.z_aug_list[s]
        obs_idx = self.obs_idx_list[s]
        x_obs = self.x_obs_list[s]
        if T_all is None or z_aug is None or obs_idx is None:
            raise RuntimeError("Sampler must be initialized before sample_y_nuts().")
        if self.theta is None or self.log_tau is None:
            raise RuntimeError("Sampler parameters have not been initialized.")

        y_nuts_config = dict(self.y_nuts_config)
        if max_tree_depth is not None:
            y_nuts_config["max_tree_depth"] = max_tree_depth
        if target_accept_prob is not None:
            y_nuts_config["target_accept_prob"] = target_accept_prob

        def y_potential_fn(params: Dict[str, torch.Tensor]) -> torch.Tensor:
            return -self.logprob_y_given_z_theta(
                y_aug=params["y_aug"],
                z_aug=z_aug,
                T_all=T_all,
                obs_idx=obs_idx,
                x_obs=x_obs,
                theta=self.theta,
                log_tau=self.log_tau,
            )

        pyro.clear_param_store()
        kernel = NUTS(
            potential_fn=y_potential_fn,
            max_tree_depth=y_nuts_config["max_tree_depth"],
            target_accept_prob=y_nuts_config["target_accept_prob"],
        )
        mcmc = MCMC(
            kernel,
            warmup_steps=0,
            num_samples=1,
            initial_params={"y_aug": y_init},
            disable_progbar=True,
        )
        mcmc.run()
        samples = mcmc.get_samples()["y_aug"]
        return samples[-1].detach()

    def sample_theta_nuts(
        self,
        max_tree_depth: Optional[int] = None,
        target_accept_prob: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample theta with one-step Pyro NUTS."""
        for s in range(self.S):
            if self.T_all_list[s] is None or self.z_aug_list[s] is None or self.y_aug_list[s] is None:
                raise RuntimeError("Sampler must be initialized before sample_theta_nuts().")
        if self.theta is None:
            raise RuntimeError("Sampler theta has not been initialized.")
        if not (
            torch.any(self.shared_parameter_mask)
            or torch.any(self.switching_parameter_mask)
        ):
            return self.theta.clone()

        theta_nuts_config = dict(self.theta_nuts_config)
        if max_tree_depth is not None:
            theta_nuts_config["max_tree_depth"] = max_tree_depth
        if target_accept_prob is not None:
            theta_nuts_config["target_accept_prob"] = target_accept_prob

        def theta_potential_fn(params: Dict[str, torch.Tensor]) -> torch.Tensor:
            theta = self._expand_theta(params)
            return -self._logprob_theta_all_series(theta)

        pyro.clear_param_store()
        kernel = NUTS(
            potential_fn=theta_potential_fn,
            max_tree_depth=theta_nuts_config["max_tree_depth"],
            target_accept_prob=theta_nuts_config["target_accept_prob"],
        )
        mcmc = MCMC(
            kernel,
            warmup_steps=theta_nuts_config["warmup_steps"],
            num_samples=theta_nuts_config["num_samples"],
            initial_params=self._pack_theta(self.theta),
            disable_progbar=True,
        )
        mcmc.run()
        samples = mcmc.get_samples()
        packed_sample = {
            name: values[-1] for name, values in samples.items()
        }
        return self._expand_theta(packed_sample).detach()

    def sample_log_tau(self) -> torch.Tensor:
        """
        Gibbs update for diagonal observation noise.

        With x_i[d] | y_i[d], tau_d^2 ~ Normal(y_i[d], tau_d^2) and
        tau_d^2 ~ InvGamma(alpha0, beta0), the conditional posterior is:

            tau_d^2 | x, y ~ InvGamma(alpha0 + N/2,
                                      beta0 + 0.5 * sum_i (x_i[d] - y_i[d])^2)

        The stored parameter is log_tau[d] = 0.5 * log(tau_d^2).
        """
        prior_config = self.prior_config
        ssr = torch.zeros(self.obs_dim, dtype=self.dtype, device=self.device)
        total_N = 0
        for s in range(self.S):
            y_aug_s = self.y_aug_list[s]
            obs_idx_s = self.obs_idx_list[s]
            x_obs_s = self.x_obs_list[s]
            if y_aug_s is None or obs_idx_s is None:
                raise RuntimeError("Sampler must be initialized before sample_log_tau().")
            y_at_obs = y_aug_s[obs_idx_s][:, self.observed_dims]
            residual = x_obs_s - y_at_obs
            ssr = ssr + (residual**2).sum(dim=0)
            total_N += int(x_obs_s.shape[0])

        alpha = torch.as_tensor(prior_config["tau2_alpha"], dtype=self.dtype, device=self.device) + 0.5 * total_N
        beta = torch.as_tensor(prior_config["tau2_beta"], dtype=self.dtype, device=self.device) + 0.5 * ssr

        # If tau^2 ~ InvGamma(alpha, beta), then precision 1/tau^2 ~ Gamma(alpha, beta).
        precision = dist.Gamma(alpha.expand_as(beta), beta).sample()
        tau2 = precision.reciprocal().clamp_min(1e-16)
        return 0.5 * torch.log(tau2)

    def sample_Q(self) -> torch.Tensor:
        """
        Sample a new CTMC generator Q from the current true jump path.

        For i != j, use independent Gamma priors:

            q_ij ~ Gamma(a_ij, b_ij)      # rate parameterization

        Given the true path, let n_ij be the number of jumps i -> j and let
        S_i be total dwell time in state i. The conditional posterior is:

            q_ij | z(t) ~ Gamma(a_ij + n_ij, b_ij + S_i)

        After sampling off-diagonal rates, diagonals are set to
        q_ii = -sum_{j != i} q_ij.

        This method has no side effects: assignment to self.Q and the required
        Omega/B refresh are handled by one_sweep().
        """
        prior_config = self.prior_config
        q_alpha = torch.as_tensor(prior_config["q_alpha"], dtype=self.dtype, device=self.device)
        q_beta = torch.as_tensor(prior_config["q_beta"], dtype=self.dtype, device=self.device)
        if q_alpha.ndim != 0 or q_beta.ndim != 0:
            raise ValueError("q_alpha and q_beta must be scalars.")

        dwell_time_k = torch.zeros(self.K, dtype=self.dtype, device=self.device)
        n_jumps_kk = torch.zeros(self.K, self.K, dtype=self.dtype, device=self.device)

        for s in range(self.S):
            T_true_s = self.T_true_list[s]
            z_true_s = self.z_true_list[s]
            T_s = self.T_list[s]
            if T_true_s is None or z_true_s is None:
                raise RuntimeError("Call initialize() before sample_Q().")
            if z_true_s.shape[0] != T_true_s.shape[0] + 1:
                raise RuntimeError("z_true must have len(T_true)+1 entries.")

            for i, state in enumerate(z_true_s.tolist()):
                t_left = 0.0 if i == 0 else T_true_s[i - 1]
                t_right = T_s if i == z_true_s.shape[0] - 1 else T_true_s[i]
                dwell_time_k[state] = dwell_time_k[state] + (t_right - t_left)

            for i in range(T_true_s.shape[0]):
                src = z_true_s[i].item()
                dst = z_true_s[i + 1].item()
                if src != dst:
                    n_jumps_kk[src, dst] = n_jumps_kk[src, dst] + 1.0

        Q = torch.zeros(self.K, self.K, dtype=self.dtype, device=self.device)
        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                    continue
                posterior_alpha = q_alpha + n_jumps_kk[i, j]
                posterior_beta = q_beta + dwell_time_k[i]
                Q[i, j] = dist.Gamma(posterior_alpha, posterior_beta).sample()
            Q[i, i] = -Q[i].sum()

        return Q.detach()

    def _evaluate_batched_transition_logprobs(
        self,
        y_prev_batch: torch.Tensor,
        y_curr_batch: torch.Tensor,
        theta_batch: torch.Tensor,
        delta_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate batched NLE transition log probabilities."""
        delta = torch.as_tensor(
            delta_batch,
            device=self.device,
            dtype=y_prev_batch.dtype,
        ).reshape(-1)
        if y_prev_batch.shape != y_curr_batch.shape:
            raise ValueError("y_prev_batch and y_curr_batch must have the same shape.")
        if y_prev_batch.shape[0] != theta_batch.shape[0] or y_prev_batch.shape[0] != delta.shape[0]:
            raise ValueError("Batch dimensions of y, theta, and delta_batch must match.")

        context = self._build_transition_context(
            theta_batch=theta_batch,
            y_prev_batch=y_prev_batch,
            delta_batch=delta,
        )
        n_steps = self._delta_to_n_steps(delta)
        return self.nle_estimator.transition_log_prob(
            x_next=y_curr_batch,
            context=context,
            x_prev=y_prev_batch,
            n_steps=n_steps.to(device=y_prev_batch.device, dtype=y_prev_batch.dtype),
            include_jacobian=True,
        )

    def _build_transition_context(
        self,
        theta_batch: torch.Tensor,
        y_prev_batch: torch.Tensor,
        delta_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Build NLE context [theta, y_prev, delta / dynamics.dt]."""
        n_steps = self._delta_to_n_steps(delta_batch)
        context = torch.cat([theta_batch, y_prev_batch], dim=-1)
        n_step_batch = n_steps.unsqueeze(-1).to(device=context.device, dtype=context.dtype)
        return torch.cat([context, n_step_batch], dim=-1)

    def _sample_transition_one_per_context(
        self,
        context: torch.Tensor,
        y_prev: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Draw one transition sample per context row in y-space."""
        n_steps = self._delta_to_n_steps(delta)
        samples = self.nle_estimator.sample_transition(
            context=context,
            x_prev=y_prev,
            n_steps=n_steps.to(device=y_prev.device, dtype=y_prev.dtype),
        )
        return samples.to(device=self.device, dtype=self.dtype)

    def _sample_forward_block_particles(
        self,
        left_y: torch.Tensor,
        left_time: torch.Tensor,
        block_times: torch.Tensor,
        block_states: torch.Tensor,
        num_particles: int,
    ) -> torch.Tensor:
        """
        Sample particles from q(block) = p(block | y_left) by simulating forward
        with the NLE transition sampler.

        Returns a tensor with shape (num_particles, block_len, D).
        """
        particles: List[torch.Tensor] = []
        y_prev = left_y.unsqueeze(0).expand(num_particles, self.D)
        previous_time = left_time

        with torch.no_grad():
            for block_time, state in zip(block_times, block_states):
                delta = block_time - previous_time
                theta_batch = self.theta[state].unsqueeze(0).expand(
                    num_particles, self.theta_dim
                )
                context = self._build_transition_context(
                    theta_batch=theta_batch,
                    y_prev_batch=y_prev,
                    delta_batch=delta.expand(num_particles),
                )
                # batch size is num_particle
                y_curr = self._sample_transition_one_per_context(
                    context=context,
                    y_prev=y_prev,
                    delta=delta.expand(num_particles),
                )
                particles.append(y_curr)
                y_prev = y_curr
                previous_time = block_time

        return torch.stack(particles, dim=1)

    def _compute_log_emission_matrix(
        self,
        y_aug: torch.Tensor,
        theta: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute FFBS emission logits for all intervals and states in one NLE call.

        `delta[j] = T_all[j+1] - T_all[j]` is converted to the NLE conditioning
        step count by delta[j] / dynamics.dt.

        Returns
        -------
        log_emission:
            Tensor with shape (L-1, K), where
            log_emission[j, k] = log p(y_aug[j+1] | y_aug[j], z_aug[j+1]=k, theta).
        """
        L = y_aug.shape[0]
        n_intervals = L - 1
        if n_intervals <= 0:
            return torch.empty(0, self.K, dtype=self.dtype, device=self.device)

        delta = torch.as_tensor(delta, device=self.device, dtype=y_aug.dtype).reshape(-1)
        if delta.shape[0] != n_intervals:
            raise ValueError("delta must have length len(y_aug) - 1.")

        # ((L-1)*K, y_dim)
        y_prev_batch = y_aug[:-1].repeat_interleave(self.K, dim=0).contiguous()
        y_curr_batch = y_aug[1:].repeat_interleave(self.K, dim=0).contiguous()

        # (K, theta_dim) -> (1, K, theta_dim) -> ((L-1), K, theta_dim)
        theta_batch = theta.unsqueeze(0).expand(n_intervals, self.K, self.theta_dim)
        # ((L-1), K, theta_dim) -> ((L-1)*K, theta_dim)
        theta_batch = theta_batch.reshape(n_intervals * self.K, self.theta_dim).contiguous()
        delta_batch = delta.repeat_interleave(self.K)

        log_probs = self._evaluate_batched_transition_logprobs(
            y_prev_batch=y_prev_batch,
            y_curr_batch=y_curr_batch,
            theta_batch=theta_batch,
            delta_batch=delta_batch,
        )
        return log_probs.reshape(n_intervals, self.K)

    def _compute_log_emission_given_z(
        self,
        y_aug: torch.Tensor,
        z_aug: torch.Tensor,
        T_all: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the total transition log density along a fixed z path using a
        single batched NLE evaluation over the selected states only.
        """
        delta = T_all[1:] - T_all[:-1]
        n_intervals = y_aug.shape[0] - 1
        if n_intervals <= 0:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)

        delta = torch.as_tensor(delta, device=self.device, dtype=y_aug.dtype).reshape(-1)
        if delta.shape[0] != n_intervals:
            raise ValueError("T_all must have length len(y_aug).")

        y_prev_batch = y_aug[:-1].contiguous()
        y_curr_batch = y_aug[1:].contiguous()
        theta_batch = theta[z_aug[1:].to(dtype=torch.long, device=self.device)].contiguous()
        log_probs = self._evaluate_batched_transition_logprobs(
            y_prev_batch=y_prev_batch,
            y_curr_batch=y_curr_batch,
            theta_batch=theta_batch,
            delta_batch=delta,
        )
        return log_probs.sum()

    @torch.no_grad()
    def sample_z_ffbs(self, s: int) -> torch.Tensor:
        """
        Sample z on the augmented grid with FFBS in log-space.

        Forward recursion:
            log_alpha_j(h)
                = log g_j(h) + logsumexp_i(log_alpha_{j-1}(i) + log A_j(i, h))

        Then we normalize at each step:
            log_alpha_j(h)
                <- log_alpha_j(h) - logsumexp_h(log_alpha_j(h))

        Hence `log_alpha[j]` in the code is a scaled forward message for the state
        on interval [T_all[j-1], T_all[j]], defined only up to an additive constant
        shared across states at time index j. This scaling does not change the FFBS
        conditional distributions because only within-time differences across states
        matter.

        where:
            g_j(h) = p(y_j | y_{j-1}, z_j = h, Delta_j; theta)
            A_j(i, h) is the transition kernel at the right endpoint T_all[j]
            A_j = B if T_all[j] is in T_true ∪ T_cand, else I

        This matches the rest of the codebase: the latent regime attached to an NLE
        transition from y_aug[j-1] to y_aug[j] is z_aug[j].
        """
        T_all = self.T_all_list[s]
        z_aug = self.z_aug_list[s]
        is_event_time = self.is_event_time_list[s]
        y_aug = self.y_aug_list[s]
        if T_all is None or z_aug is None or is_event_time is None or y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_z_ffbs().")
        if self.theta is None:
            raise RuntimeError("Sampler theta has not been initialized.")

        L = T_all.shape[0]
        log_alpha = torch.empty(L, self.K, dtype=self.dtype, device=self.device)
        log_alpha[0] = torch.log(self.initial_state_probs.clamp_min(1e-32))

        log_B = torch.log(self.B.clamp_min(1e-32))
        delta = T_all[1:] - T_all[:-1]
        emission_logits = self._compute_log_emission_matrix(y_aug, self.theta, delta)

        for j in range(L - 1):
            if is_event_time[j]:
                scores = log_alpha[j].unsqueeze(1) + log_B
                log_alpha[j + 1] = emission_logits[j] + torch.logsumexp(scores, dim=0)
            else:
                # At observation-only times the transition kernel is I, so the state
                # does not change. Only the NLE transition likelihood contributes.
                log_alpha[j + 1] = log_alpha[j] + emission_logits[j]
            log_alpha[j + 1] = log_alpha[j + 1] - torch.logsumexp(log_alpha[j + 1], dim=0)

        z = torch.empty(L, dtype=torch.long, device=self.device)
        z[L - 1] = dist.Categorical(logits=log_alpha[L - 1]).sample()

        for j in range(L - 2, -1, -1):
            if is_event_time[j]:
                logits_prev = log_alpha[j] + log_B[:, z[j + 1]]
                z[j] = dist.Categorical(logits=logits_prev).sample()
            else:
                z[j] = z[j + 1]

        z[0] = z[1]
        return z

    def prune_self_transitions(
        self,
        T_all: torch.Tensor,
        z_aug: torch.Tensor,
        is_event_time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Partition sampled event points into true and pseudo jumps.

        Event points with a state change form T_true; self-transition event points
        form T_pseudo and are represented by `pseudo_idx` into T_all. Returned
        z_true is not padded: z_true[r] is the state on true segment r.
        """
        interior_event_idx = torch.nonzero(
            is_event_time[1:-1], as_tuple=False
        ).squeeze(-1) + 1
        state_changed = (
            z_aug[interior_event_idx + 1] != z_aug[interior_event_idx]
        )
        true_idx = interior_event_idx[state_changed]
        pseudo_idx = interior_event_idx[~state_changed]

        # Index T_all directly so true and pseudo times retain their exact stored
        # floating-point representations for the next source-aware merge.
        T_true = T_all[true_idx].clone()
        z_true = torch.cat(
            [z_aug[1].reshape(1), z_aug[true_idx + 1]]
        ).to(dtype=torch.long, device=self.device)
        return T_true, z_true, true_idx, pseudo_idx

    def one_sweep(self) -> Dict[str, Any]:
        """Run one Gibbs sweep: augment grid, sample y, z, theta, tau, and Q."""
        if self.theta is None or self.log_tau is None:
            raise RuntimeError("Call initialize() before one_sweep().")
        for s in range(self.S):
            if self.T_true_list[s] is None or self.z_true_list[s] is None:
                raise RuntimeError("Call initialize() before one_sweep().")

        per_series_info: List[Dict[str, Any]] = []
        for s in range(self.S):
            T_cand = self.add_candidate_jumps(s)
            T_all, z_aug, is_event_time, obs_idx, cand_idx = self.build_augmented_grid(s, T_cand)
            y_nuts_init, sir_info = self._sample_inserted_y_by_forward_sir(
                new_grid_times=T_all,
                new_z_aug=z_aug,
                new_cand_idx=cand_idx,
                s=s,
            )
            self.T_all_list[s] = T_all
            self.z_aug_list[s] = z_aug
            self.is_event_time_list[s] = is_event_time
            self.obs_idx_list[s] = obs_idx
            self.y_aug_list[s] = self.sample_y_nuts(s, y_nuts_init)
            self.z_aug_list[s] = self.sample_z_ffbs(s)
            (
                self.T_true_list[s],
                self.z_true_list[s],
                self.true_idx_list[s],
                self.pseudo_idx_list[s],
            ) = self.prune_self_transitions(
                T_all=T_all,
                z_aug=self.z_aug_list[s],
                is_event_time=is_event_time,
            )
            per_series_info.append(
                {
                    "series": s + 1,
                    "grid_size": int(T_all.shape[0]),
                    "num_candidate_events": int(is_event_time.sum().item()),
                    "num_true_segments": int(self.z_true_list[s].shape[0]),
                    "sir_num_particles": sir_info["num_particles"],
                    "sir_min_ess": sir_info["min_ess"],
                    "sir_mean_ess": sir_info["mean_ess"],
                }
            )

        self.theta = self.sample_theta_nuts()
        self.log_tau = self.sample_log_tau()
        self.Q = self.sample_Q()
        self._refresh_uniformization()
        log_joint = self.complete_data_log_joint()

        finite_min_ess = [
            info["sir_min_ess"]
            for info in per_series_info
            if info["sir_min_ess"] == info["sir_min_ess"]
        ]
        finite_mean_ess = [
            info["sir_mean_ess"]
            for info in per_series_info
            if info["sir_mean_ess"] == info["sir_mean_ess"]
        ]
        sweep_info = {
            "num_series": self.S,
            "grid_size": sum(info["grid_size"] for info in per_series_info),
            "num_candidate_events": sum(info["num_candidate_events"] for info in per_series_info),
            "num_true_segments": sum(info["num_true_segments"] for info in per_series_info),
            "per_series": per_series_info,
            "sir_num_particles": self.sir_config["num_particles"],
            "sir_min_ess": min(finite_min_ess) if finite_min_ess else float("nan"),
            "sir_mean_ess": (
                sum(finite_mean_ess) / len(finite_mean_ess)
                if finite_mean_ess
                else float("nan")
            ),
            "log_tau_update": "conjugate_inverse_gamma_gibbs",
            "Q_update": "conjugate_gamma_gibbs",
            "omega": float(self.omega),
            "log_joint": log_joint,
        }

        self.history["y_aug"].append([
            y.detach().cpu() if y is not None else None for y in self.y_aug_list
        ])
        self.history["z_aug"].append([
            z.detach().cpu() if z is not None else None for z in self.z_aug_list
        ])
        self.history["T_all"].append([
            t.detach().cpu() if t is not None else None for t in self.T_all_list
        ])
        self.history["T_true"].append([
            t.detach().cpu() if t is not None else None for t in self.T_true_list
        ])
        self.history["z_true"].append([
            z.detach().cpu() if z is not None else None for z in self.z_true_list
        ])
        self.history["theta"].append(self.theta.detach().cpu())
        self.history["log_tau"].append(self.log_tau.detach().cpu())
        self.history["Q"].append(self.Q.detach().cpu())
        self.history["omega"].append(float(self.omega))
        self.history["diagnostics"].append(sweep_info)
        self._save_latest_sample()
        return sweep_info

    def run(self, num_sweeps: int, verbose: bool = True) -> Dict[str, List[Any]]:
        """Run multiple Gibbs sweeps and return the stored history."""
        if self.theta is None or self.log_tau is None or any(
            T_true is None or z_true is None
            for T_true, z_true in zip(self.T_true_list, self.z_true_list)
        ):
            self.initialize()

        for sweep in range(num_sweeps):
            info = self.one_sweep()
            if verbose:
                print(
                    f"[sweep {sweep + 1:03d}] "
                    f"grid={info['grid_size']}, "
                    f"candidate_events={info['num_candidate_events']}, "
                    f"true_segments={info['num_true_segments']}"
                )
        return self.history

    def _build_uniformized_transition_matrix(self) -> torch.Tensor:
        """Return B = I + Q / Omega, then clamp/renormalize rows for numeric stability."""
        B = torch.eye(self.K, dtype=self.dtype, device=self.device) + self.Q / self.omega
        B = B.clamp_min(0.0)
        B = B / B.sum(dim=1, keepdim=True)
        return B

    def _refresh_uniformization(self) -> None:
        """
        Recompute Omega and B after Q changes.

        Uniformization requires Omega >= max_i -Q_ii.  We keep a strict margin so
        candidate-jump rates Omega + Q_ii remain positive even for the largest exit
        rate state.
        """
        max_exit = torch.max(-torch.diag(self.Q)).item()
        self.omega = float(max(self.omega_scale * max_exit, max_exit + 1e-6, 1e-6))
        self.B = self._build_uniformized_transition_matrix()

    def _initialize_y_on_grid(self, s: int, T_all: torch.Tensor) -> torch.Tensor:
        """
        Initialize observed dimensions by interpolation and unobserved dimensions
        at their initial-prior locations.

        This is only used for the first sweep or when the user does not provide y.
        """
        T_obs = self.T_obs_list[s]
        x_obs = self.x_obs_list[s]
        initial_y_times = self.initial_y_times_list[s]
        initial_y_values = self.initial_y_values_list[s]
        obs_t = T_obs.detach().cpu()
        grid_t = T_all.detach().cpu()
        y_np = self.y0_prior_loc.detach().cpu().broadcast_to(
            (T_all.shape[0], self.D)
        ).clone()
        if initial_y_times is not None and initial_y_values is not None:
            reference_times = initial_y_times.detach().cpu().numpy()
            reference_values = initial_y_values.detach().cpu()
            for d in range(self.D):
                y_np[:, d] = torch.from_numpy(
                    __import__("numpy").interp(
                        grid_t.numpy(),
                        reference_times,
                        reference_values[:, d].numpy(),
                    )
                ).to(dtype=self.dtype)
        for idx_in_x, d in enumerate(
            self.observed_dims.detach().cpu().tolist()
        ):
            x_j = x_obs[:, idx_in_x].detach().cpu()
            interp = torch.from_numpy(
                __import__("numpy").interp(
                    grid_t.numpy(),
                    obs_t.numpy(),
                    x_j.numpy(),
                    left=x_j[0].item(),
                    right=x_j[-1].item()
                )
            ).to(dtype=self.dtype)
            y_np[:, d] = interp
        return y_np.to(device=self.device)

    def _sample_inserted_y_by_forward_sir(
        self,
        new_grid_times: torch.Tensor,
        new_z_aug: torch.Tensor,
        new_cand_idx: torch.Tensor,
        s: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Sample newly inserted y values by forward SIR.

        T_obs, T_true, old T_pseudo, and new T_cand are merged while retaining
        source indices. Every old-grid point temporarily keeps its previous y
        value. New candidate times between adjacent old-grid points form inserted
        blocks. For each such block, use

            q(block) = p(block | y_left)

        as the proposal. The bridge target is proportional to

            p(block, y_right | y_left)
              = p(block | y_left) p(y_right | block_last),

        so SIR weights only require the right-boundary likelihood. After all new
        values are sampled, only values on `new_grid_times` are returned; old-only
        candidate points have served as bridge endpoints and are then marginalized.
        """
        num_particles = int(self.sir_config["num_particles"])
        if num_particles < 1:
            raise ValueError("sir_config['num_particles'] must be positive.")

        if self.theta is None or self.log_tau is None:
            raise RuntimeError("Current parameters must be available before resampling y on a new grid.")

        self.last_sir_cand_particles = []

        old_y_aug = self.y_aug_list[s]

        if self.T_all_list[s] is None or old_y_aug is None:
            return self._initialize_y_on_grid(s, new_grid_times), {
                "num_particles": float(num_particles),
                "min_ess": float("nan"),
                "mean_ess": float("nan"),
            }

        if new_z_aug.shape[0] != new_grid_times.shape[0]:
            raise ValueError("new_z_aug and new_grid_times must have equal length.")

        T_cand = new_grid_times[new_cand_idx]
        times_for_sir, is_old_pseudo, is_new_candidate = (
            self._build_sir_work_grid(s, T_cand)
        )
        work_z_aug = self._expand_true_path_onto_grid(s, times_for_sir)
        y_work = torch.empty(
            times_for_sir.shape[0], self.D, dtype=self.dtype, device=self.device
        )
        has_old_y = ~is_new_candidate
        y_work[has_old_y] = old_y_aug.to(
            device=self.device, dtype=self.dtype
        )
        if torch.any(~has_old_y & ~is_new_candidate):
            raise RuntimeError(
                "A temporary-grid point had neither an old y nor a new candidate source."
            )

        inserted_idx = torch.nonzero(is_new_candidate, as_tuple=False).squeeze(-1)
        if inserted_idx.numel() == 0:
            y_sample = y_work[~is_old_pseudo]
            if y_sample.shape[0] != new_grid_times.shape[0]:
                raise RuntimeError("Four-way merge produced the wrong new-grid size.")
            return y_sample, {
                "num_particles": float(num_particles),
                "min_ess": float("nan"),
                "mean_ess": float("nan"),
            }

        ess_values: List[float] = []
        sir_cand_particles: List[Dict[str, torch.Tensor]] = []

        block_start = 0
        while block_start < inserted_idx.shape[0]:
            block_end = block_start + 1
            while (
                block_end < inserted_idx.shape[0]
                and int(inserted_idx[block_end].item())
                == int(inserted_idx[block_end - 1].item()) + 1
            ):
                block_end += 1

            block_work_idx = inserted_idx[block_start:block_end]
            left_idx = int(block_work_idx[0].item()) - 1
            right_idx = int(block_work_idx[-1].item()) + 1
            if left_idx < 0 or right_idx >= times_for_sir.shape[0]:
                raise RuntimeError("Inserted block must be bracketed by old-grid points.")
            if not bool(has_old_y[left_idx]) or not bool(has_old_y[right_idx]):
                raise RuntimeError("Inserted block endpoints must carry old y values.")

            block_times = times_for_sir[block_work_idx]
            block_states = work_z_aug[block_work_idx]

            # particles: (num_particles, block_size, D)
            particles = self._sample_forward_block_particles(
                left_y=y_work[left_idx],
                left_time=times_for_sir[left_idx],
                block_times=block_times,
                block_states=block_states,
                num_particles=num_particles,
            )
            sir_cand_particles.append(
                {
                    "times": block_times.detach().cpu(),
                    "particles": particles.detach().cpu(),
                }
            )

            right_y = y_work[right_idx].unsqueeze(0).expand(
                num_particles, self.D
            )
            last_particle = particles[:, -1, :]
            right_delta = times_for_sir[right_idx] - block_times[-1]
            right_theta = self.theta[work_z_aug[right_idx]].unsqueeze(0).expand(
                num_particles, self.theta_dim
            )
            with torch.no_grad():
                log_weights = self._evaluate_batched_transition_logprobs(
                    y_prev_batch=last_particle,
                    y_curr_batch=right_y,
                    theta_batch=right_theta,
                    delta_batch=right_delta.expand(num_particles),
                ).reshape(-1)
            normalized_log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
            weights = normalized_log_weights.exp()
            ess = weights.square().sum().reciprocal()
            ess_values.append(float(ess.detach().cpu().item()))

            chosen = dist.Categorical(probs=weights).sample()
            y_work[block_work_idx] = particles[chosen]
            sir_cand_particles[-1]["weights"] = weights.detach().cpu()
            sir_cand_particles[-1]["chosen"] = chosen.detach().cpu().reshape(())
            block_start = block_end

        ess_tensor = torch.tensor(ess_values, dtype=self.dtype)
        min_ess = float(ess_tensor.min().item())
        mean_ess = float(ess_tensor.mean().item())
        self.last_sir_cand_particles = sir_cand_particles

        y_sample = y_work[~is_old_pseudo]
        if y_sample.shape[0] != new_grid_times.shape[0]:
            raise RuntimeError("Four-way merge produced the wrong new-grid size.")

        return y_sample, {
            "num_particles": float(num_particles),
            "min_ess": min_ess,
            "mean_ess": mean_ess,
        }
