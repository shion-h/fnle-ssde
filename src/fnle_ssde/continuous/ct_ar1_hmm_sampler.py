"""
Continuous-time AR(1)-HMM Gibbs sampler in a single research-oriented file.

This implementation combines:
1. Uniformization / virtual jumps for the continuous-time discrete state path z(t)
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_value
import torch


@dataclass
class _AugmentedGrid:
    """Container for one augmented grid realization."""

    T_all: torch.Tensor  # (L+1,)
    z_aug: torch.Tensor  # (L+1,) point-state representation
    is_jump_or_virtual_time: torch.Tensor  # (L+1,) True at T_true or T_pseudo times
    obs_idx_in_T_all: torch.Tensor  # (N,) maps each observation time in T_obs to an index in T_all
    T_pseudo: torch.Tensor  # (M,) pseudo jump times only, excludes 0 and T


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
    log_tau_obs: (D,)           observation-noise log scale, not part of theta
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
        prior_config: Optional[Dict[str, Any]] = None,
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
            `nle_estimator.estimator.log_prob`.
        y_nuts_config / theta_nuts_config:
            Pyro NUTS settings used for the conditional updates.
        prior_config:
            Prior hyperparameters for theta and observation-noise variance.
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
            "warmup_steps": 32,
            "num_samples": 1,
            "max_tree_depth": 4,
            "target_accept_prob": 0.8,
        }
        if y_nuts_config is not None:
            self.y_nuts_config.update(y_nuts_config)

        self.theta_nuts_config = {
            "warmup_steps": 48,
            "num_samples": 1,
            "max_tree_depth": 4,
            "target_accept_prob": 0.8,
        }
        if theta_nuts_config is not None:
            self.theta_nuts_config.update(theta_nuts_config)

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
            # Accept old names, but internally keep theta as the NLE parameter only.
            if "theta_nle_loc" in prior_config:
                prior_config = {**prior_config, "theta_loc": prior_config["theta_nle_loc"]}
            if "theta_nle_scale" in prior_config:
                prior_config = {**prior_config, "theta_scale": prior_config["theta_nle_scale"]}
            self.prior_config.update(prior_config)

        self.history: Dict[str, List[Any]] = {
            "y_aug": [],
            "z_aug": [],
            "T_all": [],
            "theta": [],
            "log_tau_obs": [],
            "Q": [],
            "omega": [],
            "T_true": [],
            "z_true": [],
            "diagnostics": [],
        }

        self.theta: Optional[torch.Tensor] = None
        self.log_tau_obs: Optional[torch.Tensor] = None
        self.T_true: Optional[torch.Tensor] = None
        self.z_true: Optional[torch.Tensor] = None
        self.grid: Optional[_AugmentedGrid] = None
        self.y_aug: Optional[torch.Tensor] = None
        self.initial_state_probs = torch.full((self.K,), 1.0 / self.K, dtype=self.dtype, device=self.device)

    def initialize(
        self,
        *,
        initial_theta: Optional[torch.Tensor | Dict[str, torch.Tensor]] = None,
        initial_log_tau_obs: Optional[torch.Tensor] = None,
        initial_path_times: Optional[Sequence[float]] = None,
        initial_path_states: Optional[Sequence[int]] = None,
        initial_y_aug: Optional[torch.Tensor] = None,
        initial_state_probs: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize theta, observation noise, the true discrete path, and y.

        `initial_theta` is the NLE transition/emission-density parameter only. Pass
        observation noise separately as `initial_log_tau_obs`.

        Path convention
        ---------------
        - `initial_path_times` stores only true jump times in the open interval (0, T).
        - `initial_path_states` stores the regime on each true-path segment, so its
          length must be `len(initial_path_times) + 1`.
        """
        if initial_state_probs is not None:
            probs = initial_state_probs.to(device=self.device, dtype=self.dtype)
            self.initial_state_probs = probs / probs.sum()

        if initial_theta is None:
            empirical_scale = self.x_obs.std(dim=0).clamp_min(0.25)
            theta_value = torch.zeros(self.K, self.theta_dim, device=self.device, dtype=self.dtype)
            if initial_log_tau_obs is None:
                initial_log_tau_obs = torch.log(empirical_scale * 0.3)
        else:
            if isinstance(initial_theta, dict):
                if initial_log_tau_obs is None:
                    # Legacy input only. New callers should pass initial_log_tau_obs separately.
                    initial_log_tau_obs = initial_theta.get("log_tau_obs")
                if "theta" in initial_theta:
                    theta_value = initial_theta["theta"]
                elif "theta_nle" in initial_theta:
                    # Backward-compatible input name for callers that still pass theta_nle.
                    theta_value = initial_theta["theta_nle"]
                else:
                    raise ValueError("initial_theta dict must contain key 'theta'.")
            else:
                theta_value = initial_theta
            if initial_log_tau_obs is None:
                empirical_scale = self.x_obs.std(dim=0).clamp_min(0.25)
                initial_log_tau_obs = torch.log(empirical_scale * 0.3)

        self.theta = theta_value.to(device=self.device, dtype=self.dtype).clone()
        self.log_tau_obs = initial_log_tau_obs.to(device=self.device, dtype=self.dtype).clone()
        if self.theta.shape != (self.K, self.theta_dim):
            raise ValueError(f"theta must have shape ({self.K}, {self.theta_dim}).")
        if self.log_tau_obs.shape != (self.D,):
            raise ValueError(f"log_tau_obs must have shape ({self.D},).")

        if initial_path_times is None or initial_path_states is None:
            self.T_true = torch.empty(0, dtype=self.dtype, device=self.device)
            self.z_true = torch.zeros(1, dtype=torch.long, device=self.device)
        else:
            path_times = torch.tensor(list(initial_path_times), dtype=self.dtype, device=self.device)
            path_states = torch.tensor(list(initial_path_states), dtype=torch.long, device=self.device)
            if path_times.numel() > 0:
                if not torch.all(path_times[1:] > path_times[:-1]):
                    raise ValueError("initial_path_times must be strictly increasing.")
                if torch.any(path_times <= 0.0) or torch.any(path_times >= self.T):
                    raise ValueError("initial_path_times must lie in the open interval (0, T).")
            if path_states.numel() != path_times.numel() + 1:
                raise ValueError("initial_path_states must have len(initial_path_times)+1 entries.")
            self.T_true = path_times
            self.z_true = path_states

        self.grid = self.build_augmented_grid()
        if initial_y_aug is None:
            self.y_aug = self._initialize_y_on_grid(self.grid.T_all)
        else:
            y_aug = initial_y_aug.to(device=self.device, dtype=self.dtype)
            if y_aug.shape != (self.grid.T_all.shape[0], self.D):
                raise ValueError("initial_y_aug shape does not match the current augmented grid.")
            self.y_aug = y_aug.clone()

    def build_augmented_grid(self) -> _AugmentedGrid:
        """
        Build T_all = T_true U T_pseudo U T_obs.

        Under the convention used here, T_obs already contains the endpoints
        0 and T, so they are not added separately.
        """
        if self.T_true is None or self.z_true is None:
            raise RuntimeError("Call initialize() before build_augmented_grid().")

        T_pseudo = self.add_virtual_jumps()
        T_all_values: List[float] = []
        is_event_values: List[bool] = []
        obs_idx_values: List[int] = []

        # K-way merge of the sorted sets T_true, T_pseudo, and T_obs.
        # This avoids building T_all by concatenate+unique and avoids searching
        # T_all again to mark event times.  Exact ties are collapsed by recording
        # which source sequences attained the selected next_time.
        idx_true = 0
        idx_pseudo = 0
        idx_obs = 0
        idx_all = 0
        while idx_true < self.T_true.shape[0] or idx_pseudo < T_pseudo.shape[0] or idx_obs < self.N:
            t_true = float(self.T_true[idx_true].item()) if idx_true < self.T_true.shape[0] else float("inf")
            t_pseudo = float(T_pseudo[idx_pseudo].item()) if idx_pseudo < T_pseudo.shape[0] else float("inf")
            t_obs = float(self.T_obs[idx_obs].item()) if idx_obs < self.N else float("inf")

            next_time = min(t_true, t_pseudo, t_obs)
            from_true = t_true == next_time
            from_pseudo = t_pseudo == next_time
            from_obs = t_obs == next_time

            T_all_values.append(next_time)
            is_event_values.append(from_true or from_pseudo)

            if from_true:
                idx_true += 1
            if from_pseudo:
                idx_pseudo += 1
            if from_obs:
                obs_idx_values.append(idx_all)
                idx_obs += 1
            idx_all += 1

        T_all = torch.tensor(T_all_values, dtype=self.dtype, device=self.device)
        is_jump_or_virtual_time = torch.tensor(is_event_values, dtype=torch.bool, device=self.device)
        obs_idx_in_T_all = torch.tensor(obs_idx_values, dtype=torch.long, device=self.device)
        if obs_idx_in_T_all.shape[0] != self.N:
            raise RuntimeError("Observation time was not found on the merged grid.")

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

        return _AugmentedGrid(
            T_all=T_all,
            z_aug=z_aug,
            is_jump_or_virtual_time=is_jump_or_virtual_time,
            obs_idx_in_T_all=obs_idx_in_T_all,
            T_pseudo=T_pseudo,
        )

    def add_virtual_jumps(self) -> torch.Tensor:
        """
        Add virtual jumps to the current true path using uniformization.

        On an interval with state k and duration dt, virtual jumps are sampled from:
            Poisson((Omega - q_k) dt) = Poisson((Omega + Q_kk) dt)
        because q_k = -Q_kk.
        """
        if self.T_true is None or self.z_true is None:
            raise RuntimeError("Call initialize() before add_virtual_jumps().")

        pseudo_times: List[torch.Tensor] = []
        for j, state in enumerate(self.z_true.tolist()):
            # state is the regime on the interval between t0 and t1
            t0 = 0.0 if j == 0 else self.T_true[j - 1].item()
            t1 = self.T.item() if j == self.z_true.shape[0] - 1 else self.T_true[j].item()
            dt = t1 - t0
            if dt <= 0.0:
                raise RuntimeError("dt <= 0 encountered when adding virtual jumps.")

            rate = self.omega + float(self.Q[state, state].item())
            if rate <= 0.0:
                raise RuntimeError("omega must be greater than max_k(-Q_kk) to ensure a positive virtual jump rate.")

            num_virtual = torch.poisson(torch.tensor(rate * dt, dtype=self.dtype)).to(torch.long).item()
            if num_virtual == 0:
                continue

            u = torch.rand(num_virtual, generator=self.rng, dtype=self.dtype)
            times = t0 + dt * u
            pseudo_times.append(times.to(device=self.device))
        if not pseudo_times:
            return torch.empty(0, dtype=self.dtype, device=self.device)

        T_pseudo = torch.cat(pseudo_times)
        return torch.unique(T_pseudo, sorted=True)

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
        log_tau_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diagonal Gaussian observation log density."""
        if log_tau_obs is None:
            log_tau_obs = self.log_tau_obs
        if log_tau_obs is None:
            raise RuntimeError("Observation noise has not been initialized.")
        tau = torch.exp(log_tau_obs).clamp_min(1e-8)
        return dist.Normal(y, tau).log_prob(x).sum()

    def logprob_y_given_z_theta(
        self,
        y_aug: torch.Tensor,
        z_aug: torch.Tensor,
        T_all: torch.Tensor,
        obs_idx_in_T_all: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        log_tau_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Conditional log density:

            log p(y_aug | z_aug, theta, log_tau_obs, x)
              = log p(y_0 | z_0, theta)
              + sum_l log p(y_l | y_{l-1}, z_l, Delta_l, theta)
              + sum_i log p(x_i | y_{m(i)}, log_tau_obs)
        """
        theta = self.theta if theta is None else theta
        if theta is None:
            raise RuntimeError("Sampler theta has not been initialized.")
        if log_tau_obs is None:
            log_tau_obs = self.log_tau_obs
        if log_tau_obs is None:
            raise RuntimeError("Observation noise has not been initialized.")
        logp = dist.Normal(
            torch.zeros(self.D, dtype=self.dtype, device=self.device),
            self.y0_prior_scale,
        ).log_prob(y_aug[0]).sum()
        logp = logp + self._compute_log_emission_given_z(y_aug, z_aug, T_all, theta)

        y_at_obs = y_aug[obs_idx_in_T_all]  # (N, D)
        tau = torch.exp(log_tau_obs).clamp_min(1e-8)
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

        The observation model x | y, log_tau_obs is constant with respect to
        theta, so it is intentionally omitted here. Observation noise is updated
        separately by sample_log_tau_obs().
        """
        cfg = self.prior_config
        logp = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        logp = logp + dist.Normal(cfg["theta_loc"], cfg["theta_scale"]).log_prob(theta).sum()
        logp = logp + self._compute_log_emission_given_z(y_aug, z_aug, T_all, theta)
        return logp

    def sample_y_nuts(
        self,
        *,
        warmup_steps: Optional[int] = None,
        num_samples: Optional[int] = None,
        max_tree_depth: Optional[int] = None,
        target_accept_prob: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sample the full augmented latent path y_aug with Pyro NUTS."""
        if self.grid is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_y_nuts().")
        if self.theta is None or self.log_tau_obs is None:
            raise RuntimeError("Sampler parameters have not been initialized.")

        grid = self.grid
        y_init = self.y_aug.clone()
        config = dict(self.y_nuts_config)
        if warmup_steps is not None:
            config["warmup_steps"] = warmup_steps
        if num_samples is not None:
            config["num_samples"] = num_samples
        if max_tree_depth is not None:
            config["max_tree_depth"] = max_tree_depth
        if target_accept_prob is not None:
            config["target_accept_prob"] = target_accept_prob

        base_dist = dist.Normal(
            torch.zeros_like(y_init),
            torch.ones_like(y_init),
        ).to_event(2)

        def y_model() -> None:
            y = pyro.sample("y_aug", base_dist)
            target = self.logprob_y_given_z_theta(
                y_aug=y,
                z_aug=grid.z_aug,
                T_all=grid.T_all,
                obs_idx_in_T_all=grid.obs_idx_in_T_all,
                theta=self.theta,
                log_tau_obs=self.log_tau_obs,
            )
            pyro.factor("target_log_density", target - base_dist.log_prob(y))

        pyro.clear_param_store()
        kernel = NUTS(
            y_model,
            init_strategy=init_to_value(values={"y_aug": y_init}),
            max_tree_depth=config["max_tree_depth"],
            target_accept_prob=config["target_accept_prob"],
        )
        mcmc = MCMC(
            kernel,
            warmup_steps=config["warmup_steps"],
            num_samples=config["num_samples"],
            disable_progbar=True,
        )
        mcmc.run()
        samples = mcmc.get_samples()["y_aug"]
        diagnostics = self._extract_mcmc_diagnostics(mcmc)
        return samples[-1].detach(), diagnostics

    def sample_theta_nuts(
        self,
        *,
        warmup_steps: Optional[int] = None,
        num_samples: Optional[int] = None,
        max_tree_depth: Optional[int] = None,
        target_accept_prob: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sample only the NLE transition/emission-density parameter theta with NUTS."""
        if self.grid is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_theta_nuts().")
        if self.theta is None:
            raise RuntimeError("Sampler theta has not been initialized.")

        grid = self.grid
        config = dict(self.theta_nuts_config)
        if warmup_steps is not None:
            config["warmup_steps"] = warmup_steps
        if num_samples is not None:
            config["num_samples"] = num_samples
        if max_tree_depth is not None:
            config["max_tree_depth"] = max_tree_depth
        if target_accept_prob is not None:
            config["target_accept_prob"] = target_accept_prob

        cfg = self.prior_config
        init_values = {
            "theta": self.theta.clone(),
        }

        def theta_model() -> None:
            theta = pyro.sample(
                "theta",
                dist.Normal(cfg["theta_loc"], cfg["theta_scale"]).expand([self.K, self.theta_dim]).to_event(2),
            )
            log_lik = self.logprob_theta_given_y_z(
                theta=theta,
                y_aug=self.y_aug,
                z_aug=grid.z_aug,
                T_all=grid.T_all,
            )
            # logprob_theta_given_y_z already includes the priors, so subtract them once.
            log_prior = dist.Normal(cfg["theta_loc"], cfg["theta_scale"]).log_prob(theta).sum()
            pyro.factor("likelihood_plus_prior_correction", log_lik - log_prior)

        pyro.clear_param_store()
        kernel = NUTS(
            theta_model,
            init_strategy=init_to_value(values=init_values),
            max_tree_depth=config["max_tree_depth"],
            target_accept_prob=config["target_accept_prob"],
        )
        mcmc = MCMC(
            kernel,
            warmup_steps=config["warmup_steps"],
            num_samples=config["num_samples"],
            disable_progbar=True,
        )
        mcmc.run()
        samples = mcmc.get_samples()
        diagnostics = self._extract_mcmc_diagnostics(mcmc)
        theta_sample = samples["theta"][-1].detach()
        return theta_sample, diagnostics

    def sample_log_tau_obs(self) -> torch.Tensor:
        """
        Gibbs update for diagonal observation noise.

        With x_i[d] | y_i[d], tau_d^2 ~ Normal(y_i[d], tau_d^2) and
        tau_d^2 ~ InvGamma(alpha0, beta0), the conditional posterior is:

            tau_d^2 | x, y ~ InvGamma(alpha0 + N/2,
                                      beta0 + 0.5 * sum_i (x_i[d] - y_i[d])^2)

        The stored parameter is log_tau_obs[d] = 0.5 * log(tau_d^2).
        """
        if self.grid is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_log_tau_obs().")

        cfg = self.prior_config
        y_at_obs = self.y_aug[self.grid.obs_idx_in_T_all]  # (N, D)
        residual = self.x_obs - y_at_obs
        ssr = (residual**2).sum(dim=0)  # (D,)

        alpha = torch.as_tensor(cfg["tau2_alpha"], dtype=self.dtype, device=self.device) + 0.5 * self.N
        beta = torch.as_tensor(cfg["tau2_beta"], dtype=self.dtype, device=self.device) + 0.5 * ssr

        # If tau^2 ~ InvGamma(alpha, beta), then precision 1/tau^2 ~ Gamma(alpha, beta).
        precision = dist.Gamma(alpha.expand_as(beta), beta).sample()
        tau2 = precision.reciprocal().clamp_min(1e-16)
        return 0.5 * torch.log(tau2).detach()

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

        cfg = self.prior_config
        q_alpha = torch.as_tensor(cfg["q_alpha"], dtype=self.dtype, device=self.device)
        q_beta = torch.as_tensor(cfg["q_beta"], dtype=self.dtype, device=self.device)
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
        if self.flow_model is None:
            raise RuntimeError("NLE transition requested but no flow_model is available.")
        delta = torch.as_tensor(
            delta_batch,
            device=self.device,
            dtype=y_prev_batch.dtype,
        ).reshape(-1)
        if y_prev_batch.shape != y_curr_batch.shape:
            raise ValueError("y_prev_batch and y_curr_batch must have the same shape.")
        if y_prev_batch.shape[0] != theta_batch.shape[0] or y_prev_batch.shape[0] != delta.shape[0]:
            raise ValueError("Batch dimensions of y, theta, and delta_batch must match.")

        n_steps = self._delta_to_n_steps(delta)
        context = torch.cat([theta_batch, y_prev_batch], dim=-1)
        # change it to a column vector
        n_step_batch = n_steps.unsqueeze(-1).to(
            device=context.device,
            dtype=context.dtype,
        )
        context = torch.cat([context, n_step_batch], dim=-1)
        # first dimension is not a batch, needed to be 1
        return self.flow_model.log_prob(y_curr_batch.unsqueeze(0), condition=context)

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
            A_j = B if T_all[j] is in T_true ∪ T_pseudo, else I

        This matches the rest of the codebase: the latent regime attached to an NLE
        transition from y_aug[j-1] to y_aug[j] is z_aug[j].
        """
        if self.grid is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_z_ffbs().")
        if self.theta is None:
            raise RuntimeError("Sampler theta has not been initialized.")

        grid = self.grid
        L = grid.T_all.shape[0]
        log_alpha = torch.empty(L, self.K, dtype=self.dtype, device=self.device)
        log_alpha[0] = torch.log(self.initial_state_probs.clamp_min(1e-32))

        log_B = torch.log(self.B.clamp_min(1e-32))
        delta = grid.T_all[1:] - grid.T_all[:-1]
        emission_logits = self._compute_log_emission_matrix(self.y_aug, self.theta, delta)

        for j in range(L - 1):
            if grid.is_jump_or_virtual_time[j]:
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
            if grid.is_jump_or_virtual_time[j]:
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
        is_jump_or_virtual_time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the sampled augmented skeleton back to a true jump path.

        Only candidate times where the state actually changes are retained as true jumps.
        Returned z_true is not padded: z_true[r] is the state on true segment r.
        z_true[j]: state in [T_true[j], T_true[j+1]]
        """
        T_true_value: List[float] = []
        # z_true[j] is the regime on [T_true[j-1], T_true[j]],
        # so we start with z_aug[1](equals z_aug[0]).
        z_true_value: List[int] = [int(z_aug[1].item())]

        # T_all[1], ... T_all[T_all.shape[0] - 2]
        # z_aug[2], ... z_aug[T_all.shape[0] - 1]
        for j in range(1, T_all.shape[0] - 1):
            # not obs, and state changes across this interval
            if bool(is_jump_or_virtual_time[j]) and int(z_aug[j + 1].item()) != int(z_aug[j].item()):
                T_true_value.append(float(T_all[j].item()))
                z_true_value.append(int(z_aug[j + 1].item()))

        T_true = torch.tensor(T_true_value, dtype=self.dtype, device=self.device)
        z_true = torch.tensor(z_true_value, dtype=torch.long, device=self.device)
        return T_true, z_true

    def one_sweep(self) -> Dict[str, Any]:
        """Run one Gibbs sweep: augment grid, sample y, z, theta, tau, and Q."""
        if self.T_true is None or self.z_true is None:
            raise RuntimeError("Call initialize() before one_sweep().")

        previous_grid = self.grid
        previous_y = None if self.y_aug is None else self.y_aug.clone()
        self.grid = self.build_augmented_grid()
        self.y_aug = self._project_previous_y_to_new_grid(
            new_grid_times=self.grid.T_all,
            old_grid_times=None if previous_grid is None else previous_grid.T_all,
            old_y=previous_y,
        )

        y_sample, y_diag = self.sample_y_nuts()
        self.y_aug = y_sample

        self.grid = _AugmentedGrid(
            T_all=self.grid.T_all,
            z_aug=self.sample_z_ffbs(),
            is_jump_or_virtual_time=self.grid.is_jump_or_virtual_time,
            obs_idx_in_T_all=self.grid.obs_idx_in_T_all,
            T_pseudo=self.grid.T_pseudo,
        )

        theta_sample, theta_diag = self.sample_theta_nuts()
        self.theta = theta_sample
        self.log_tau_obs = self.sample_log_tau_obs()

        self.T_true, self.z_true = self.prune_self_transitions(
            T_all=self.grid.T_all,
            z_aug=self.grid.z_aug,
            is_jump_or_virtual_time=self.grid.is_jump_or_virtual_time,
        )
        self.Q = self.sample_Q()
        self._refresh_uniformization()

        sweep_info = {
            "grid_size": int(self.grid.T_all.shape[0]),
            "num_candidate_events": int(self.grid.is_jump_or_virtual_time.sum().item()),
            "num_true_segments": int(self.z_true.shape[0]),
            "y_nuts": y_diag,
            "theta_nuts": theta_diag,
            "log_tau_obs_update": "conjugate_inverse_gamma_gibbs",
            "Q_update": "conjugate_gamma_gibbs",
            "omega": float(self.omega),
        }

        self.history["y_aug"].append(self.y_aug.detach().cpu())
        self.history["z_aug"].append(self.grid.z_aug.detach().cpu())
        self.history["T_all"].append(self.grid.T_all.detach().cpu())
        self.history["theta"].append(self.theta.detach().cpu())
        self.history["log_tau_obs"].append(self.log_tau_obs.detach().cpu())
        self.history["Q"].append(self.Q.detach().cpu())
        self.history["omega"].append(float(self.omega))
        self.history["T_true"].append(self.T_true.detach().cpu())
        self.history["z_true"].append(self.z_true.detach().cpu())
        self.history["diagnostics"].append(sweep_info)
        return sweep_info

    def run(self, num_sweeps: int, verbose: bool = True) -> Dict[str, List[Any]]:
        """Run multiple Gibbs sweeps and return the stored history."""
        if self.y_aug is None:
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_uniformized_transition_matrix(self) -> torch.Tensor:
        """Return B = I + Q / Omega, then clamp/renormalize rows for numeric stability."""
        B = torch.eye(self.K, dtype=self.dtype, device=self.device) + self.Q / self.omega
        B = B.clamp_min(0.0)
        B = B / B.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return B

    def _refresh_uniformization(self) -> None:
        """
        Recompute Omega and B after Q changes.

        Uniformization requires Omega >= max_i -Q_ii.  We keep a strict margin so
        virtual-jump rates Omega + Q_ii remain positive even for the largest exit
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

    def _project_previous_y_to_new_grid(
        self,
        new_grid_times: torch.Tensor,
        old_grid_times: Optional[torch.Tensor] = None,
        old_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reuse the previous latent trajectory as the NUTS initial value on a new grid.

        If a previous y_aug exists, linearly interpolate it onto the new grid.
        Otherwise use observation-based initialization.
        """
        if old_y is None or old_grid_times is None:
            return self._initialize_y_on_grid(new_grid_times)

        old_t = old_grid_times.detach().cpu()
        new_t = new_grid_times.detach().cpu()
        y_new = torch.empty(new_grid_times.shape[0], self.D, dtype=self.dtype)
        for d in range(self.D):
            old_y_d = old_y[:, d].detach().cpu()
            interp = torch.from_numpy(
                __import__("numpy").interp(
                    new_t.numpy(),
                    old_t.numpy(),
                    old_y_d.numpy(),
                    left=old_y_d[0].item(),
                    right=old_y_d[-1].item(),
                )
            ).to(dtype=self.dtype)
            y_new[:, d] = interp
        return y_new.to(device=self.device)

    def _extract_mcmc_diagnostics(self, mcmc: MCMC) -> Dict[str, Any]:
        """Best-effort extraction of a few NUTS diagnostics from Pyro."""
        try:
            num_samples = int(getattr(mcmc, "num_samples", 0))
        except Exception:
            num_samples = 0
        if num_samples < 2:
            return {"note": "diagnostics skipped because num_samples < 2"}

        try:
            diagnostics = mcmc.diagnostics()
        except Exception:
            return {}

        summary: Dict[str, Any] = {}
        if isinstance(diagnostics, dict):
            for key in ("acceptance rate", "divergences", "step size"):
                if key in diagnostics:
                    summary[key] = diagnostics[key]
        return summary
