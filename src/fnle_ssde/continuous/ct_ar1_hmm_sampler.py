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

from typing import Any, Dict, List, Optional, Tuple

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch

class ContinuousTimeAR1HMMSampler:
    """
    Gibbs sampler for a continuous-time AR(1)-HMM with NLE transition density.

    Shapes
    ------
    x_obs: (N, D)
    T_obs: (N,)
    y_aug: (L+1, D)
    z_aug: (L+1,)
    theta: (K, theta_dim)       NLE transition/emission-density parameters only
    log_tau: (D,)               observation-noise log scale, not part of theta
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
        x_obs: torch.Tensor,
        obs_times: torch.Tensor,
        T: float,
        *,
        omega_scale: float = 1.5,
        nle_estimator: Any,
        y_nuts_config: Optional[Dict[str, Any]] = None,
        theta_nuts_config: Optional[Dict[str, Any]] = None,
        sir_config: Optional[Dict[str, Any]] = None,
        prior_config: Optional[Dict[str, Any]] = None,
        switching_parameter_mask: Optional[torch.Tensor] = None,
        y0_prior_scale: float = 5.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        seed: int = 0,
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
        switching_parameter_mask:
            Boolean tensor of shape (theta_dim,). True dimensions have one
            parameter per regime; False dimensions are shared across regimes.
            If omitted, every parameter is regime-specific as before.
        y0_prior_scale:
            Optional fallback scale for initialization of y.
        """
        self.device = device or x_obs.device
        self.dtype = dtype
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)
        pyro.set_rng_seed(seed)

        self.Q = Q.to(device=self.device, dtype=self.dtype)
        self.omega_scale = float(omega_scale)
        self.x_obs = x_obs.to(device=self.device, dtype=self.dtype)
        self.T_obs = obs_times.to(device=self.device, dtype=self.dtype)
        self.T = torch.tensor(float(T), device=self.device, dtype=self.dtype)
        self.K = int(self.Q.shape[0])
        self.N, self.D = self.x_obs.shape
        self.y0_prior_scale = float(y0_prior_scale)
        self.nle_estimator = nle_estimator
        self.flow_model = nle_estimator.estimator
        if self.flow_model is None:
            raise ValueError("nle_estimator.estimator must be trained/loaded before sampling.")
        self.theta_dim = int(nle_estimator.dynamics.theta_dim)
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
        self.switching_parameter_mask = switching_parameter_mask
        self.shared_parameter_mask = ~switching_parameter_mask
        self.dynamics_dt = float(nle_estimator.dynamics.dt)
        if self.dynamics_dt <= 0.0:
            raise ValueError("nle_estimator.dynamics.dt must be positive.")
        if hasattr(self.flow_model, "to"):
            self.flow_model.to(self.device)
        if hasattr(self.flow_model, "eval"):
            self.flow_model.eval()

        if self.Q.shape != (self.K, self.K):
            raise ValueError("Q must be square with shape (K, K).")
        if not torch.allclose(self.Q.sum(dim=1), torch.zeros(self.K, dtype=self.dtype, device=self.device), atol=1e-8):
            raise ValueError("Each row of Q must sum to zero.")
        if self.T_obs.ndim != 1 or self.T_obs.shape[0] != self.N:
            raise ValueError("obs_times must have shape (N,).")
        if self.N < 2:
            raise ValueError("obs_times must contain at least the endpoints 0 and T.")
        if not torch.all(self.T_obs[1:] > self.T_obs[:-1]):
            raise ValueError("obs_times must be strictly increasing.")
        if abs(float(self.T_obs[0].item())) > 1e-10:
            raise ValueError("obs_times must start at 0.")
        if abs(float(self.T_obs[-1].item()) - float(self.T.item())) > 1e-10:
            raise ValueError("obs_times must end at T.")

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
        self.log_tau: Optional[torch.Tensor] = None
        self.T_true: Optional[torch.Tensor] = None
        self.z_true: Optional[torch.Tensor] = None
        self.T_all: Optional[torch.Tensor] = None
        self.z_aug: Optional[torch.Tensor] = None
        # True at times in T_true ∪ T_cand, i.e. times where FFBS uses B instead of I.
        self.is_event_time: Optional[torch.Tensor] = None
        self.obs_idx: Optional[torch.Tensor] = None
        self.true_idx: Optional[torch.Tensor] = None
        self.y_aug: Optional[torch.Tensor] = None
        self.initial_state_probs = torch.full((self.K,), 1.0 / self.K, dtype=self.dtype, device=self.device)

    def initialize(
        self,
        *,
        initial_theta: Optional[torch.Tensor] = None,
        initial_log_tau: Optional[torch.Tensor] = None,
        initial_state_probs: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize theta, observation noise, and the true discrete path.

        `initial_theta` is the NLE transition/emission-density parameter only. Pass
        observation noise separately as `initial_log_tau`.

        The initial true path has no jumps. Its single state is sampled from
        `initial_state_probs`, unless that probability vector is overridden here.
        """
        if initial_state_probs is not None:
            probs = initial_state_probs.to(device=self.device, dtype=self.dtype)
            self.initial_state_probs = probs / probs.sum()

        prior_config = self.prior_config
        if initial_theta is None:
            theta_loc, theta_scale = self._theta_prior_parameters()
            theta_value = dist.Normal(theta_loc, theta_scale).sample((self.K,))
            # Shared dimensions represent one random variable, not K independent draws.
            theta_value[:, self.shared_parameter_mask] = theta_value[
                0, self.shared_parameter_mask
            ]
        else:
            theta_value = initial_theta

        if initial_log_tau is None:
            tau2_alpha = torch.as_tensor(prior_config["tau2_alpha"], dtype=self.dtype, device=self.device)
            tau2_beta = torch.as_tensor(prior_config["tau2_beta"], dtype=self.dtype, device=self.device)
            precision = dist.Gamma(tau2_alpha, tau2_beta).sample((self.D,))
            initial_log_tau = 0.5 * torch.log(precision.reciprocal().clamp_min(1e-16))

        self.theta = theta_value.to(device=self.device, dtype=self.dtype).clone()
        self.log_tau = initial_log_tau.to(device=self.device, dtype=self.dtype).clone()
        if self.theta.shape != (self.K, self.theta_dim):
            raise ValueError(f"theta must have shape ({self.K}, {self.theta_dim}).")
        self._validate_shared_theta(self.theta)
        if self.log_tau.shape != (self.D,):
            raise ValueError(f"log_tau must have shape ({self.D},).")

        self.T_true = torch.empty(0, dtype=self.dtype, device=self.device)
        initial_state = dist.Categorical(probs=self.initial_state_probs).sample()
        self.z_true = initial_state.reshape(1).to(dtype=torch.long, device=self.device)

        self.T_all = None
        self.z_aug = None
        self.is_event_time = None
        self.obs_idx = None
        self.true_idx = None
        self.y_aug = None

    def _theta_prior_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return theta prior location and scale as vectors of shape (theta_dim,)."""
        theta_loc = torch.as_tensor(
            self.prior_config["theta_loc"], dtype=self.dtype, device=self.device
        ).broadcast_to((self.theta_dim,))
        theta_scale = torch.as_tensor(
            self.prior_config["theta_scale"], dtype=self.dtype, device=self.device
        ).broadcast_to((self.theta_dim,))
        if torch.any(theta_scale <= 0):
            raise ValueError("theta prior scales must be positive.")
        return theta_loc, theta_scale

    def _validate_shared_theta(self, theta: torch.Tensor) -> None:
        """Ensure expanded theta contains identical values in shared dimensions."""
        if not torch.any(self.shared_parameter_mask):
            return
        shared = theta[:, self.shared_parameter_mask]
        if not torch.allclose(shared, shared[0].expand_as(shared)):
            raise ValueError(
                "initial_theta must be identical across regimes for dimensions "
                "marked False in switching_parameter_mask."
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
        for parameter_index in range(self.theta_dim):
            if self.switching_parameter_mask[parameter_index]:
                column = packed["theta_switching"][:, switching_index]
                switching_index += 1
            else:
                column = packed["theta_shared"][shared_index].expand(self.K)
                shared_index += 1
            columns.append(column)
        return torch.stack(columns, dim=1)

    def _theta_log_prior(self, packed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate each unique theta prior exactly once."""
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

    def build_augmented_grid(
        self,
        T_cand: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return augmented-grid arrays for T_all = T_true U T_cand U T_obs."""
        if self.T_true is None or self.z_true is None:
            raise RuntimeError("Call initialize() before build_augmented_grid().")

        T_all_values: List[float] = []
        is_event_values: List[bool] = []
        obs_idx_values: List[int] = []
        cand_idx_values: List[int] = []

        # K-way merge of the sorted sets T_true, T_cand, and T_obs.
        # This avoids building T_all by concatenate+unique and avoids searching
        # T_all again to mark event times.  Exact ties are collapsed by recording
        # which source sequences attained the selected next_time.
        idx_true = 0
        idx_cand = 0
        idx_obs = 0
        idx_all = 0
        while idx_true < self.T_true.shape[0] or idx_cand < T_cand.shape[0] or idx_obs < self.N:
            t_true = float(self.T_true[idx_true].item()) if idx_true < self.T_true.shape[0] else float("inf")
            t_cand = float(T_cand[idx_cand].item()) if idx_cand < T_cand.shape[0] else float("inf")
            t_obs = float(self.T_obs[idx_obs].item()) if idx_obs < self.N else float("inf")

            next_time = min(t_true, t_cand, t_obs)
            from_true = t_true == next_time
            from_cand = t_cand == next_time
            from_obs = t_obs == next_time

            T_all_values.append(next_time)
            is_event_values.append(from_true or from_cand)

            if from_true:
                idx_true += 1
            if from_cand:
                cand_idx_values.append(idx_all)
                idx_cand += 1
            if from_obs:
                obs_idx_values.append(idx_all)
                idx_obs += 1
            idx_all += 1

        T_all = torch.tensor(T_all_values, dtype=self.dtype, device=self.device)
        is_event_time = torch.tensor(is_event_values, dtype=torch.bool, device=self.device)
        obs_idx = torch.tensor(obs_idx_values, dtype=torch.long, device=self.device)
        cand_idx = torch.tensor(cand_idx_values, dtype=torch.long, device=self.device)
        if obs_idx.shape[0] != self.N:
            raise RuntimeError("Observation time was not found on the merged grid.")
        if cand_idx.shape[0] != T_cand.shape[0]:
            raise RuntimeError("Candidate jump time was not found on the merged grid.")

        # Point-state representation on T_all.
        # z_aug[j+1] is the regime on [T_all[j], T_all[j+1]], and z_aug[0]
        # duplicates the first interval state. z_true is not padded: z_true[r] is
        # the regime on true-path segment r.
        interval_states = torch.empty(T_all.shape[0] - 1, dtype=torch.long, device=self.device)
        T_true_idx = 0
        for j in range(T_all.shape[0] - 1):
            # time at the left of the interval
            left = T_all[j]
            # T_true_idx is the number of true jumps at or before `left`.
            # The matching true-path state is z_true[T_true_idx].
            while T_true_idx < self.T_true.shape[0] and self.T_true[T_true_idx] <= left:
                T_true_idx += 1
            interval_states[j] = self.z_true[T_true_idx]

        z_aug = torch.empty(T_all.shape[0], dtype=torch.long, device=self.device)
        # To fit z_aug indices with y_aug indices, duplicate the first interval state
        z_aug[0] = interval_states[0]
        z_aug[1:] = interval_states

        return T_all, z_aug, is_event_time, obs_idx, cand_idx

    def add_candidate_jumps(self) -> torch.Tensor:
        """
        Add candidate jumps to the current true path using uniformization.

        On an interval with state k and duration dt, candidate jumps are sampled from:
            Poisson((Omega - q_k) dt) = Poisson((Omega + Q_kk) dt)
        because q_k = -Q_kk.
        """
        if self.T_true is None or self.z_true is None:
            raise RuntimeError("Call initialize() before add_candidate_jumps().")

        cand_times: List[torch.Tensor] = []
        for j, state in enumerate(self.z_true.tolist()):
            # state is the regime on the interval between t0 and t1
            t0 = 0.0 if j == 0 else self.T_true[j - 1].item()
            t1 = self.T.item() if j == self.z_true.shape[0] - 1 else self.T_true[j].item()
            dt = t1 - t0
            if dt <= 0.0:
                raise RuntimeError("dt <= 0 encountered when adding candidate jumps.")

            rate = self.omega + float(self.Q[state, state].item())
            if rate <= 0.0:
                raise RuntimeError("omega must be greater than max_k(-Q_kk) to ensure a positive candidate jump rate.")

            num_candidate = torch.poisson(torch.tensor(rate * dt, dtype=self.dtype)).to(torch.long).item()
            if num_candidate == 0:
                continue

            u = torch.rand(num_candidate, generator=self.rng, dtype=self.dtype)
            times = t0 + dt * u
            cand_times.append(times.to(device=self.device))
        if not cand_times:
            return torch.empty(0, dtype=self.dtype, device=self.device)

        T_cand = torch.unique(torch.cat(cand_times), sorted=True)

        # Candidate times that exactly coincide with retained grid points are not
        # newly inserted points.  Keeping them in cand_idx would make SIR skip a
        # y value that should be copied from the old grid.
        if T_cand.numel() > 0:
            retained_times = torch.cat([self.T_obs.to(device=self.device), self.T_true.to(device=self.device)])
            if retained_times.numel() > 0:
                T_cand = T_cand[~torch.isin(T_cand, retained_times)]
        return T_cand

    def _delta_to_n_steps(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous elapsed time to the discrete NLE conditioning step count.

        The NLE was trained on simulator steps of size `dynamics.dt`, so an
        interval Delta is mapped to the continuous step count Delta / dt.
        """
        return delta / self.dynamics_dt

    def observation_logprob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        log_tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diagonal Gaussian observation log density."""
        if log_tau is None:
            log_tau = self.log_tau
        tau = torch.exp(log_tau)
        return dist.Normal(y, tau).log_prob(x).sum()

    def logprob_y_given_z_theta(
        self,
        y_aug: torch.Tensor,
        z_aug: torch.Tensor,
        T_all: torch.Tensor,
        obs_idx: torch.Tensor,
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
        logp = dist.Normal(
            torch.zeros(self.D, dtype=self.dtype, device=self.device),
            self.y0_prior_scale,
        ).log_prob(y_aug[0]).sum()
        logp = logp + self._compute_log_emission_given_z(y_aug, z_aug, T_all, theta)

        y_at_obs = y_aug[obs_idx]  # (N, D)
        tau = torch.exp(log_tau)
        logp = logp + dist.Normal(y_at_obs, tau).log_prob(self.x_obs).sum()
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
        self._validate_shared_theta(theta)
        logp = self._theta_log_prior(self._pack_theta(theta))
        logp = logp + self._compute_log_emission_given_z(y_aug, z_aug, T_all, theta)
        return logp

    def sample_y_nuts(
        self,
        y_init: torch.Tensor,
        max_tree_depth: Optional[int] = None,
        target_accept_prob: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample the full augmented latent path y_aug with one-step Pyro NUTS."""
        if self.T_all is None or self.z_aug is None or self.obs_idx is None:
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
                z_aug=self.z_aug,
                T_all=self.T_all,
                obs_idx=self.obs_idx,
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
        if self.T_all is None or self.z_aug is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_theta_nuts().")
        if self.theta is None:
            raise RuntimeError("Sampler theta has not been initialized.")

        theta_nuts_config = dict(self.theta_nuts_config)
        if max_tree_depth is not None:
            theta_nuts_config["max_tree_depth"] = max_tree_depth
        if target_accept_prob is not None:
            theta_nuts_config["target_accept_prob"] = target_accept_prob

        def theta_potential_fn(params: Dict[str, torch.Tensor]) -> torch.Tensor:
            theta = self._expand_theta(params)
            return -self.logprob_theta_given_y_z(
                theta=theta,
                y_aug=self.y_aug,
                z_aug=self.z_aug,
                T_all=self.T_all,
            )

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
        if self.obs_idx is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_log_tau().")

        prior_config = self.prior_config
        y_at_obs = self.y_aug[self.obs_idx]  # (N, D)
        residual = self.x_obs - y_at_obs
        ssr = (residual**2).sum(dim=0)  # (D,)

        alpha = torch.as_tensor(prior_config["tau2_alpha"], dtype=self.dtype, device=self.device) + 0.5 * self.N
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
        if self.T_true is None or self.z_true is None:
            raise RuntimeError("Call initialize() before sample_Q().")
        if self.z_true.shape[0] != self.T_true.shape[0] + 1:
            raise RuntimeError("z_true must have len(T_true)+1 entries.")

        prior_config = self.prior_config
        q_alpha = torch.as_tensor(prior_config["q_alpha"], dtype=self.dtype, device=self.device)
        q_beta = torch.as_tensor(prior_config["q_beta"], dtype=self.dtype, device=self.device)
        if q_alpha.ndim != 0 or q_beta.ndim != 0:
            raise ValueError("q_alpha and q_beta must be scalars.")

        dwell_time_k = torch.zeros(self.K, dtype=self.dtype, device=self.device)
        n_jumps_kk = torch.zeros(self.K, self.K, dtype=self.dtype, device=self.device)

        for i, state in enumerate(self.z_true.tolist()):
            t_left = 0.0 if i == 0 else self.T_true[i - 1]
            t_right = self.T if i == self.z_true.shape[0] - 1 else self.T_true[i]
            dwell_time_k[state] = dwell_time_k[state] + (t_right - t_left)

        for i in range(self.T_true.shape[0]):
            src = self.z_true[i].item()
            dst = self.z_true[i + 1].item()
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
        block_indices: List[int],
        new_grid_times: torch.Tensor,
        new_z_aug: torch.Tensor,
        num_particles: int,
    ) -> torch.Tensor:
        """
        Sample particles from q(block) = p(block | y_left) by simulating forward
        with the NLE transition sampler.

        Returns a tensor with shape (num_particles, block_len, D).
        """
        particles: List[torch.Tensor] = []
        y_prev = left_y.unsqueeze(0).expand(num_particles, self.D)
        prev_idx = block_indices[0] - 1

        with torch.no_grad():
            for idx in block_indices:
                delta = new_grid_times[idx] - new_grid_times[prev_idx]
                theta_batch = self.theta[new_z_aug[idx]].unsqueeze(0).expand(num_particles, self.theta_dim)
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
                prev_idx = idx

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

    def sample_z_ffbs(self) -> torch.Tensor:
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
        if self.T_all is None or self.z_aug is None or self.is_event_time is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_z_ffbs().")
        if self.theta is None:
            raise RuntimeError("Sampler theta has not been initialized.")

        L = self.T_all.shape[0]
        log_alpha = torch.empty(L, self.K, dtype=self.dtype, device=self.device)
        log_alpha[0] = torch.log(self.initial_state_probs.clamp_min(1e-32))

        log_B = torch.log(self.B.clamp_min(1e-32))
        delta = self.T_all[1:] - self.T_all[:-1]
        emission_logits = self._compute_log_emission_matrix(self.y_aug, self.theta, delta)

        for j in range(L - 1):
            if self.is_event_time[j]:
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
            if self.is_event_time[j]:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the sampled augmented skeleton back to a true jump path.

        Only candidate times where the state actually changes are retained as true jumps.
        Returned z_true is not padded: z_true[r] is the state on true segment r.
        z_true[j]: state in [T_true[j], T_true[j+1]]
        """
        T_true_value: List[float] = []
        true_idx_value: List[int] = []
        # z_true[j] is the regime on [T_true[j-1], T_true[j]],
        # so we start with z_aug[1](equals z_aug[0]).
        z_true_value: List[int] = [int(z_aug[1].item())]

        # T_all[1], ... T_all[T_all.shape[0] - 2]
        # z_aug[2], ... z_aug[T_all.shape[0] - 1]
        for j in range(1, T_all.shape[0] - 1):
            # not obs, and state changes across this interval
            if bool(is_event_time[j]):
                if int(z_aug[j + 1].item()) != int(z_aug[j].item()):
                    T_true_value.append(float(T_all[j].item()))
                    true_idx_value.append(j)
                    z_true_value.append(int(z_aug[j + 1].item()))

        T_true = torch.tensor(T_true_value, dtype=self.dtype, device=self.device)
        z_true = torch.tensor(z_true_value, dtype=torch.long, device=self.device)
        true_idx = torch.tensor(
            true_idx_value,
            dtype=torch.long,
            device=self.device,
        )
        return T_true, z_true, true_idx

    def one_sweep(self) -> Dict[str, Any]:
        """Run one Gibbs sweep: augment grid, sample y, z, theta, tau, and Q."""
        if (
            self.T_true is None
            or self.z_true is None
            or self.theta is None
            or self.log_tau is None
        ):
            raise RuntimeError("Call initialize() before one_sweep().")

        T_cand = self.add_candidate_jumps()
        T_all, z_aug, is_event_time, obs_idx, cand_idx = self.build_augmented_grid(T_cand)
        y_nuts_init, sir_info = self._sample_inserted_y_by_forward_sir(
            new_grid_times=T_all,
            new_z_aug=z_aug,
            new_cand_idx=cand_idx,
        )
        self.T_all = T_all
        self.z_aug = z_aug
        self.is_event_time = is_event_time
        self.obs_idx = obs_idx
        self.y_aug = self.sample_y_nuts(y_nuts_init)
        self.z_aug = self.sample_z_ffbs()
        self.T_true, self.z_true, self.true_idx = self.prune_self_transitions(
            T_all=self.T_all,
            z_aug=self.z_aug,
            is_event_time=self.is_event_time,
        )

        self.theta = self.sample_theta_nuts()
        self.log_tau = self.sample_log_tau()
        self.Q = self.sample_Q()
        self._refresh_uniformization()

        sweep_info = {
            "grid_size": int(self.T_all.shape[0]),
            "num_candidate_events": int(self.is_event_time.sum().item()),
            "num_true_segments": int(self.z_true.shape[0]),
            "sir_num_particles": sir_info["num_particles"],
            "sir_min_ess": sir_info["min_ess"],
            "sir_mean_ess": sir_info["mean_ess"],
            "log_tau_update": "conjugate_inverse_gamma_gibbs",
            "Q_update": "conjugate_gamma_gibbs",
            "omega": float(self.omega),
        }

        self.history["y_aug"].append(self.y_aug.detach().cpu())
        self.history["z_aug"].append(self.z_aug.detach().cpu())
        self.history["T_all"].append(self.T_all.detach().cpu())
        self.history["theta"].append(self.theta.detach().cpu())
        self.history["log_tau"].append(self.log_tau.detach().cpu())
        self.history["Q"].append(self.Q.detach().cpu())
        self.history["omega"].append(float(self.omega))
        self.history["T_true"].append(self.T_true.detach().cpu())
        self.history["z_true"].append(self.z_true.detach().cpu())
        self.history["diagnostics"].append(sweep_info)
        return sweep_info

    def run(self, num_sweeps: int, verbose: bool = True) -> Dict[str, List[Any]]:
        """Run multiple Gibbs sweeps and return the stored history."""
        if self.theta is None or self.log_tau is None or self.T_true is None or self.z_true is None:
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

    def _initialize_y_on_grid(self, T_all: torch.Tensor) -> torch.Tensor:
        """
        Initialize y on a grid by linear interpolation of observations.

        This is only used for the first sweep or when the user does not provide y.
        """
        if self.N == 1:
            return self.x_obs[0].unsqueeze(0).repeat(T_all.shape[0], 1)

        obs_t = self.T_obs.detach().cpu()
        grid_t = T_all.detach().cpu()
        y_np = torch.empty(T_all.shape[0], self.D, dtype=self.dtype)
        for d in range(self.D):
            x_d = self.x_obs[:, d].detach().cpu()
            interp = torch.from_numpy(
                __import__("numpy").interp(grid_t.numpy(), obs_t.numpy(), x_d.numpy(), left=x_d[0].item(), right=x_d[-1].item())
            ).to(dtype=self.dtype)
            y_np[:, d] = interp
        return y_np.to(device=self.device)

    def _sample_inserted_y_by_forward_sir(
        self,
        new_grid_times: torch.Tensor,
        new_z_aug: torch.Tensor,
        new_cand_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Sample newly inserted y values by forward SIR.

        Points corresponding to old observations and old retained true jumps keep
        their previous y values. For each contiguous block of inserted candidate
        times, use

            q(block) = p(block | y_left)

        as the proposal. The bridge target is proportional to

            p(block, y_right | y_left)
              = p(block | y_left) p(y_right | block_last),

        so SIR weights only require the right-boundary likelihood.
        """
        num_particles = int(self.sir_config["num_particles"])
        if num_particles < 1:
            raise ValueError("sir_config['num_particles'] must be positive.")

        if self.theta is None or self.log_tau is None:
            raise RuntimeError("Current parameters must be available before resampling y on a new grid.")

        self.last_sir_cand_particles = []

        if self.T_all is None or self.y_aug is None or self.obs_idx is None:
            return self._initialize_y_on_grid(new_grid_times), {
                "num_particles": float(num_particles),
                "min_ess": float("nan"),
                "mean_ess": float("nan"),
            }

        inserted_idx = new_cand_idx
        inserted_idx_set = set(inserted_idx.tolist())

        retained_old_idx_list = self.obs_idx.tolist()
        if self.true_idx is not None and self.true_idx.numel() > 0:
            retained_old_idx_list.extend(self.true_idx.tolist())
        retained_old_idx_list = sorted(set(retained_old_idx_list))
        retained_old_idx_tensor = torch.tensor(retained_old_idx_list, dtype=torch.long, device=self.device)
        retained_old_times = self.T_all[retained_old_idx_tensor]

        y_sample = torch.empty(new_grid_times.shape[0], self.D, dtype=self.dtype, device=self.device)
        retained_old_ptr = 0
        for new_idx in range(new_grid_times.shape[0]):
            # Candidate times are filled by SIR below.
            if new_idx in inserted_idx_set:
                continue
            if retained_old_ptr >= retained_old_times.shape[0]:
                raise RuntimeError("Ran out of retained old grid points while matching the new grid.")
            if retained_old_times[retained_old_ptr].item() != new_grid_times[new_idx].item():
                raise RuntimeError("Non-candidate point on the new grid did not match the retained old grid.")
            y_sample[new_idx] = self.y_aug[retained_old_idx_tensor[retained_old_ptr]].to(device=self.device, dtype=self.dtype)
            retained_old_ptr += 1

        if inserted_idx.numel() == 0:
            return y_sample, {
                "num_particles": float(num_particles),
                "min_ess": float("nan"),
                "mean_ess": float("nan"),
            }

        ess_values: List[float] = []
        sir_cand_particles: List[Dict[str, torch.Tensor]] = []

        inserted_idx_values = inserted_idx.tolist()
        block_start = inserted_idx_values[0]
        block_end = inserted_idx_values[0]
        inserted_blocks: List[Tuple[int, int]] = []
        # Making inserted_blocks
        for idx in inserted_idx_values[1:]:
            if idx == block_end + 1:
                block_end = idx
            else:
                inserted_blocks.append((block_start, block_end))
                block_start = idx
                block_end = idx
        inserted_blocks.append((block_start, block_end))

        for block_start, block_end in inserted_blocks:
            left_idx = block_start - 1
            right_idx = block_end + 1
            if left_idx < 0 or right_idx >= new_grid_times.shape[0]:
                raise RuntimeError("Inserted block must be bracketed by retained grid points.")

            # particles: (num_particles, block_size, D)
            particles = self._sample_forward_block_particles(
                left_y=y_sample[left_idx],
                block_indices=list(range(block_start, block_end + 1)),
                new_grid_times=new_grid_times,
                new_z_aug=new_z_aug,
                num_particles=num_particles,
            )
            block_time = new_grid_times[block_start : block_end + 1].detach().cpu()
            sir_cand_particles.append(
                {
                    "times": block_time,
                    "particles": particles.detach().cpu(),
                }
            )

            right_y = y_sample[right_idx].unsqueeze(0).expand(num_particles, self.D)
            last_particle = particles[:, -1, :]
            right_delta = new_grid_times[right_idx] - new_grid_times[block_end]
            right_theta = self.theta[new_z_aug[right_idx]].unsqueeze(0).expand(num_particles, self.theta_dim)
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
            y_sample[block_start : block_end + 1] = particles[chosen]
            sir_cand_particles[-1]["weights"] = weights.detach().cpu()
            sir_cand_particles[-1]["chosen"] = chosen.detach().cpu().reshape(())

        ess_tensor = torch.tensor(ess_values, dtype=self.dtype)
        min_ess = float(ess_tensor.min().item())
        mean_ess = float(ess_tensor.mean().item())
        self.last_sir_cand_particles = sir_cand_particles

        return y_sample, {
            "num_particles": float(num_particles),
            "min_ess": min_ess,
            "mean_ess": mean_ess,
        }
