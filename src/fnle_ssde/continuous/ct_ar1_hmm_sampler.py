"""
Continuous-time AR(1)-HMM Gibbs sampler in a single research-oriented file.

This implementation combines:
1. Uniformization / virtual jumps for the continuous-time discrete state path z(t)
2. Conditional NUTS updates for the continuous latent trajectory y on an augmented grid
3. FFBS updates for the discrete state skeleton on the same augmented grid
4. Conditional NUTS updates for the OU / observation parameters

Model summary
-------------
z(t) in {0, ..., K-1} follows a continuous-time Markov jump process with generator Q.

Conditional on regime k, each coordinate d=1,...,D follows an independent OU process:

    dy_t^(d) = -lambda_{k,d} (y_t^(d) - mu_{k,d}) dt + sigma_{k,d} dW_t^(d)

The observation model is diagonal Gaussian:

    x_i | y(t_i) ~ Normal(y(t_i), diag(tau_obs^2))

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
    Gibbs sampler for a continuous-time AR(1)-HMM with diagonal multivariate OU dynamics.

    Shapes
    ------
    x_obs: (N, D)
    T_obs: (N,)
    y_aug: (L+1, D)
    z_aug: (L+1,)
    mu: (K, D)
    log_lambda: (K, D)
    log_sigma: (K, D)
    log_tau_obs: (D,)

    Conventions
    -----------
    - States are indexed from 0 to K-1.
    - `z_aug[j+1]` denotes the state on the interval [T_all[j], T_all[j+1]].
    - Therefore the OU transition from y_aug[j] to y_aug[j+1] uses state z_aug[j+1].
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
        y_nuts_config: Optional[Dict[str, Any]] = None,
        theta_nuts_config: Optional[Dict[str, Any]] = None,
        prior_config: Optional[Dict[str, float]] = None,
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
        y_nuts_config / theta_nuts_config:
            Pyro NUTS settings used for the conditional updates.
        prior_config:
            Prior hyperparameters for theta.
        y0_prior_scale:
            Optional fallback scale for initialization of y.
        """
        self.device = device or x_obs.device
        self.dtype = dtype
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)
        pyro.set_rng_seed(seed)

        self.Q = Q.to(device=self.device, dtype=self.dtype)
        self.x_obs = x_obs.to(device=self.device, dtype=self.dtype)
        self.T_obs = obs_times.to(device=self.device, dtype=self.dtype)
        self.T = torch.tensor(float(T), device=self.device, dtype=self.dtype)
        self.K = int(self.Q.shape[0])
        self.N, self.D = self.x_obs.shape
        self.y0_prior_scale = float(y0_prior_scale)

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

        max_exit = torch.max(-torch.diag(self.Q)).item()
        self.omega = float(max(omega_scale * max_exit, max_exit + 1e-6))
        self.B = self._build_uniformized_transition_matrix()

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
            "mu_loc": 0.0,
            "mu_scale": 3.0,
            "log_lambda_loc": 0.0,
            "log_lambda_scale": 0.75,
            "log_sigma_loc": -0.5,
            "log_sigma_scale": 0.75,
            "log_tau_loc": -1.0,
            "log_tau_scale": 0.75,
        }
        if prior_config is not None:
            self.prior_config.update(prior_config)

        self.history: Dict[str, List[Any]] = {
            "y_aug": [],
            "z_aug": [],
            "T_all": [],
            "theta": [],
            "T_true": [],
            "z_true": [],
            "diagnostics": [],
        }

        self.theta: Dict[str, torch.Tensor] = {}
        self.T_true: Optional[torch.Tensor] = None
        self.z_true: Optional[torch.Tensor] = None
        self.grid: Optional[_AugmentedGrid] = None
        self.y_aug: Optional[torch.Tensor] = None
        self.initial_state_probs = torch.full((self.K,), 1.0 / self.K, dtype=self.dtype, device=self.device)

    def initialize(
        self,
        *,
        initial_theta: Optional[Dict[str, torch.Tensor]] = None,
        initial_path_times: Optional[Sequence[float]] = None,
        initial_path_states: Optional[Sequence[int]] = None,
        initial_y_aug: Optional[torch.Tensor] = None,
        initial_state_probs: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize theta, the true discrete path, and an initial y trajectory.

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
            mu0 = self.x_obs.mean(dim=0)
            empirical_scale = self.x_obs.std(dim=0).clamp_min(0.25)
            initial_theta = {
                "mu": mu0.unsqueeze(0).repeat(self.K, 1),
                "log_lambda": torch.zeros(self.K, self.D, device=self.device, dtype=self.dtype),
                "log_sigma": torch.log(empirical_scale.unsqueeze(0).repeat(self.K, 1) * 0.7),
                "log_tau_obs": torch.log(empirical_scale * 0.3),
            }

        self.theta = {
            "mu": initial_theta["mu"].to(device=self.device, dtype=self.dtype).clone(),
            "log_lambda": initial_theta["log_lambda"].to(device=self.device, dtype=self.dtype).clone(),
            "log_sigma": initial_theta["log_sigma"].to(device=self.device, dtype=self.dtype).clone(),
            "log_tau_obs": initial_theta["log_tau_obs"].to(device=self.device, dtype=self.dtype).clone(),
        }

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
        path_idx = 0
        for j in range(T_all.shape[0] - 1):
            # time at the left of the interval
            left = T_all[j]
            # path_idx is the number of true jumps at or before `left`.
            # The matching true-path state is z_true[path_idx].
            while path_idx < self.T_true.shape[0] and self.T_true[path_idx] <= left:
                path_idx += 1
            interval_states[j] = self.z_true[path_idx]

        z_aug = torch.empty(T_all.shape[0], dtype=torch.long, device=self.device)
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
        segment_states = self.z_true
        for j, state in enumerate(segment_states.tolist()):
            # state is the regime on the interval between t0 and t1
            t0 = 0.0 if j == 0 else self.T_true[j - 1].item()
            t1 = self.T.item() if j == segment_states.shape[0] - 1 else self.T_true[j].item()
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

    def ou_transition_mean_var(
        self,
        y_prev: torch.Tensor,
        delta: torch.Tensor,
        mu_k: torch.Tensor,
        log_lambda_k: torch.Tensor,
        log_sigma_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diagonal OU transition moments.

        Shapes
        ------
        y_prev: (..., D)
        delta: scalar tensor or broadcastable to (..., 1)
        mu_k, log_lambda_k, log_sigma_k: (D,)
        """
        delta = torch.as_tensor(delta, dtype=self.dtype, device=self.device)
        if delta.ndim == 0:
            delta = delta.reshape(1)
        lam = torch.exp(log_lambda_k).clamp_min(1e-8)
        sig = torch.exp(log_sigma_k).clamp_min(1e-8)
        while delta.ndim < y_prev.ndim:
            delta = delta.unsqueeze(-1)

        exp_term = torch.exp(-lam * delta)
        mean = mu_k + exp_term * (y_prev - mu_k)
        # Use -expm1(-2 lambda delta) for small-delta stability.
        var = (sig**2) * (-torch.expm1(-2.0 * lam * delta)) / (2.0 * lam)
        var = var.clamp_min(1e-10)
        return mean, var

    def ou_transition_logprob(
        self,
        y_curr: torch.Tensor,
        y_prev: torch.Tensor,
        state: int | torch.Tensor,
        delta: torch.Tensor,
        theta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Log p(y_curr | y_prev, z=state, delta; theta) for diagonal OU transitions."""
        theta = self.theta if theta is None else theta
        k = int(state) if not torch.is_tensor(state) else int(state.item())
        mu_k = theta["mu"][k]
        log_lambda_k = theta["log_lambda"][k]
        log_sigma_k = theta["log_sigma"][k]
        mean, var = self.ou_transition_mean_var(y_prev, delta, mu_k, log_lambda_k, log_sigma_k)
        return dist.Normal(mean, torch.sqrt(var)).log_prob(y_curr).sum()

    def observation_logprob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        log_tau_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diagonal Gaussian observation log density."""
        if log_tau_obs is None:
            log_tau_obs = self.theta["log_tau_obs"]
        tau = torch.exp(log_tau_obs).clamp_min(1e-8)
        return dist.Normal(y, tau).log_prob(x).sum()

    def logprob_y_given_z_theta(
        self,
        y_aug: torch.Tensor,
        z_aug: torch.Tensor,
        T_all: torch.Tensor,
        obs_idx_in_T_all: torch.Tensor,
        theta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Conditional log density:

            log p(y_aug | z_aug, theta, x)
              = log p(y_0 | z_0, theta)
              + sum_l log p(y_l | y_{l-1}, z_l, Delta_l, theta)
              + sum_i log p(x_i | y_{m(i)}, theta)
        """
        theta = self.theta if theta is None else theta
        y0_state = int(z_aug[0].item())
        mu0 = theta["mu"][y0_state]
        lam0 = torch.exp(theta["log_lambda"][y0_state]).clamp_min(1e-8)
        sig0 = torch.exp(theta["log_sigma"][y0_state]).clamp_min(1e-8)
        stationary_var = (sig0**2) / (2.0 * lam0)
        logp = dist.Normal(mu0, torch.sqrt(stationary_var.clamp_min(1e-10))).log_prob(y_aug[0]).sum()

        deltas = T_all[1:] - T_all[:-1]
        for l in range(1, T_all.shape[0]):
            logp = logp + self.ou_transition_logprob(
                y_curr=y_aug[l],
                y_prev=y_aug[l - 1],
                state=z_aug[l],
                delta=deltas[l - 1],
                theta=theta,
            )

        y_at_obs = y_aug[obs_idx_in_T_all]  # (N, D)
        tau = torch.exp(theta["log_tau_obs"]).clamp_min(1e-8)
        logp = logp + dist.Normal(y_at_obs, tau).log_prob(self.x_obs).sum()
        return logp

    def logprob_theta_given_y_z(
        self,
        theta: Dict[str, torch.Tensor],
        y_aug: torch.Tensor,
        z_aug: torch.Tensor,
        T_all: torch.Tensor,
        obs_idx_in_T_all: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional log density:

            log p(theta | y, z, x)
              = log p(theta)
              + sum_l log p(y_l | y_{l-1}, z_l, Delta_l, theta)
              + sum_i log p(x_i | y_{m(i)}, theta)
        """
        cfg = self.prior_config
        logp = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        logp = logp + dist.Normal(cfg["mu_loc"], cfg["mu_scale"]).log_prob(theta["mu"]).sum()
        logp = logp + dist.Normal(cfg["log_lambda_loc"], cfg["log_lambda_scale"]).log_prob(theta["log_lambda"]).sum()
        logp = logp + dist.Normal(cfg["log_sigma_loc"], cfg["log_sigma_scale"]).log_prob(theta["log_sigma"]).sum()
        logp = logp + dist.Normal(cfg["log_tau_loc"], cfg["log_tau_scale"]).log_prob(theta["log_tau_obs"]).sum()

        deltas = T_all[1:] - T_all[:-1]
        for l in range(1, T_all.shape[0]):
            logp = logp + self.ou_transition_logprob(
                y_curr=y_aug[l],
                y_prev=y_aug[l - 1],
                state=z_aug[l],
                delta=deltas[l - 1],
                theta=theta,
            )

        y_at_obs = y_aug[obs_idx_in_T_all]
        tau = torch.exp(theta["log_tau_obs"]).clamp_min(1e-8)
        logp = logp + dist.Normal(y_at_obs, tau).log_prob(self.x_obs).sum()
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
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Sample theta = (mu, log_lambda, log_sigma, log_tau_obs) with Pyro NUTS."""
        if self.grid is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_theta_nuts().")

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
            "mu": self.theta["mu"].clone(),
            "log_lambda": self.theta["log_lambda"].clone(),
            "log_sigma": self.theta["log_sigma"].clone(),
            "log_tau_obs": self.theta["log_tau_obs"].clone(),
        }

        def theta_model() -> None:
            mu = pyro.sample(
                "mu",
                dist.Normal(cfg["mu_loc"], cfg["mu_scale"]).expand([self.K, self.D]).to_event(2),
            )
            log_lambda = pyro.sample(
                "log_lambda",
                dist.Normal(cfg["log_lambda_loc"], cfg["log_lambda_scale"]).expand([self.K, self.D]).to_event(2),
            )
            log_sigma = pyro.sample(
                "log_sigma",
                dist.Normal(cfg["log_sigma_loc"], cfg["log_sigma_scale"]).expand([self.K, self.D]).to_event(2),
            )
            log_tau_obs = pyro.sample(
                "log_tau_obs",
                dist.Normal(cfg["log_tau_loc"], cfg["log_tau_scale"]).expand([self.D]).to_event(1),
            )
            theta_now = {
                "mu": mu,
                "log_lambda": log_lambda,
                "log_sigma": log_sigma,
                "log_tau_obs": log_tau_obs,
            }
            log_lik = self.logprob_theta_given_y_z(
                theta=theta_now,
                y_aug=self.y_aug,
                z_aug=grid.z_aug,
                T_all=grid.T_all,
                obs_idx_in_T_all=grid.obs_idx_in_T_all,
            )
            # logprob_theta_given_y_z already includes the priors, so subtract them once.
            log_prior = (
                dist.Normal(cfg["mu_loc"], cfg["mu_scale"]).log_prob(mu).sum()
                + dist.Normal(cfg["log_lambda_loc"], cfg["log_lambda_scale"]).log_prob(log_lambda).sum()
                + dist.Normal(cfg["log_sigma_loc"], cfg["log_sigma_scale"]).log_prob(log_sigma).sum()
                + dist.Normal(cfg["log_tau_loc"], cfg["log_tau_scale"]).log_prob(log_tau_obs).sum()
            )
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
        theta_sample = {
            "mu": samples["mu"][-1].detach(),
            "log_lambda": samples["log_lambda"][-1].detach(),
            "log_sigma": samples["log_sigma"][-1].detach(),
            "log_tau_obs": samples["log_tau_obs"][-1].detach(),
        }
        return theta_sample, diagnostics

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

        This matches the rest of the codebase: the latent regime attached to an OU
        transition from y_aug[j-1] to y_aug[j] is z_aug[j].
        """
        if self.grid is None or self.y_aug is None:
            raise RuntimeError("Sampler must be initialized before sample_z_ffbs().")

        grid = self.grid
        L = grid.T_all.shape[0]
        log_alpha = torch.empty(L, self.K, dtype=self.dtype, device=self.device)
        log_alpha[0] = torch.log(self.initial_state_probs.clamp_min(1e-32))

        log_B = torch.log(self.B.clamp_min(1e-32))

        deltas = grid.T_all[1:] - grid.T_all[:-1]
        emission_logits = torch.empty(L - 1, self.K, dtype=self.dtype, device=self.device)

        for j in range(L - 1):
            dt = deltas[j]
            for k in range(self.K):
                emission_logits[j, k] = self.ou_transition_logprob(
                    y_curr=self.y_aug[j + 1],
                    y_prev=self.y_aug[j],
                    state=k,
                    delta=dt,
                    theta=self.theta,
                )

            if grid.is_jump_or_virtual_time[j + 1]:
                scores = log_alpha[j].unsqueeze(1) + log_B
                log_alpha[j + 1] = emission_logits[j] + torch.logsumexp(scores, dim=0)
            else:
                # At observation-only times the transition kernel is I, so the state
                # does not change. Only the OU transition likelihood contributes.
                log_alpha[j + 1] = log_alpha[j] + emission_logits[j]
            log_alpha[j + 1] = log_alpha[j + 1] - torch.logsumexp(log_alpha[j + 1], dim=0)

        z = torch.empty(L, dtype=torch.long, device=self.device)
        z[L - 1] = dist.Categorical(logits=log_alpha[L - 1]).sample()

        for j in range(L - 2, -1, -1):
            if grid.is_jump_or_virtual_time[j + 1]:
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
        """Run one Gibbs sweep: augment grid, sample y, sample z, sample theta."""
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

        self.T_true, self.z_true = self.prune_self_transitions(
            T_all=self.grid.T_all,
            z_aug=self.grid.z_aug,
            is_jump_or_virtual_time=self.grid.is_jump_or_virtual_time,
        )

        sweep_info = {
            "grid_size": int(self.grid.T_all.shape[0]),
            "num_candidate_events": int(self.grid.is_jump_or_virtual_time.sum().item()),
            "num_true_segments": int(self.z_true.shape[0]),
            "y_nuts": y_diag,
            "theta_nuts": theta_diag,
        }

        self.history["y_aug"].append(self.y_aug.detach().cpu())
        self.history["z_aug"].append(self.grid.z_aug.detach().cpu())
        self.history["T_all"].append(self.grid.T_all.detach().cpu())
        self.history["theta"].append({k: v.detach().cpu() for k, v in self.theta.items()})
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


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Small synthetic example with D=2.
    K = 2
    D = 2
    T = 8.0
    N = 25

    Q_true = torch.tensor(
        [
            [-0.6, 0.6],
            [0.45, -0.45],
        ],
        dtype=torch.float64,
    )

    theta_true = {
        "mu": torch.tensor([[0.0, -0.5], [2.0, 1.0]], dtype=torch.float64),
        "log_lambda": torch.log(torch.tensor([[1.0, 0.7], [1.4, 0.9]], dtype=torch.float64)),
        "log_sigma": torch.log(torch.tensor([[0.35, 0.25], [0.30, 0.40]], dtype=torch.float64)),
        "log_tau_obs": torch.log(torch.tensor([0.15, 0.20], dtype=torch.float64)),
    }

    # Simulate a simple continuous-time latent state path.
    # Here T_true stores only internal jump times, while z_true stores segment states.
    state = 0
    t = 0.0
    T_true = []
    z_true = []
    while t < T:
        rate = float(-Q_true[state, state].item())
        wait = dist.Exponential(rate).sample().item()
        next_t = t + wait
        z_true.append(state)
        if next_t >= T:
            break
        T_true.append(next_t)
        state = int(dist.Categorical(probs=(Q_true[state].clone().clamp_min(0.0) / rate)).sample().item())
        t = next_t

    T_true_t = torch.tensor(T_true, dtype=torch.float64)
    z_true_t = torch.tensor(z_true, dtype=torch.long)

    # Irregular observation times T_obs with 0 = t_1 and T = t_N.
    T_obs_interior = torch.sort(torch.rand(N - 2, dtype=torch.float64) * T).values
    T_obs = torch.cat(
        [
            torch.tensor([0.0], dtype=torch.float64),
            T_obs_interior,
            torch.tensor([T], dtype=torch.float64),
        ]
    )

    # Simulate y(T_obs) directly by propagating along the piecewise-constant regime path.
    y_obs = torch.zeros(N, D, dtype=torch.float64)
    current_y = theta_true["mu"][0] + 0.2 * torch.randn(D, dtype=torch.float64)
    current_time = 0.0
    path_idx = 0

    for i, t_obs in enumerate(T_obs.tolist()):
        while path_idx < len(T_true) and t_obs > T_true[path_idx]:
            state_k = int(z_true_t[path_idx].item())
            dt = T_true[path_idx] - current_time
            lam = torch.exp(theta_true["log_lambda"][state_k])
            sig = torch.exp(theta_true["log_sigma"][state_k])
            mu = theta_true["mu"][state_k]
            exp_term = torch.exp(-lam * dt)
            mean = mu + exp_term * (current_y - mu)
            var = (sig**2) * (-torch.expm1(-2.0 * lam * dt)) / (2.0 * lam)
            current_y = mean + torch.sqrt(var.clamp_min(1e-10)) * torch.randn(D, dtype=torch.float64)
            current_time = float(T_true[path_idx])
            path_idx += 1

        state_k = int(z_true_t[path_idx].item())
        dt = t_obs - current_time
        lam = torch.exp(theta_true["log_lambda"][state_k])
        sig = torch.exp(theta_true["log_sigma"][state_k])
        mu = theta_true["mu"][state_k]
        exp_term = torch.exp(-lam * dt)
        mean = mu + exp_term * (current_y - mu)
        var = (sig**2) * (-torch.expm1(-2.0 * lam * dt)) / (2.0 * lam)
        current_y = mean + torch.sqrt(var.clamp_min(1e-10)) * torch.randn(D, dtype=torch.float64)
        current_time = t_obs
        y_obs[i] = current_y

    x_obs = y_obs + torch.exp(theta_true["log_tau_obs"]) * torch.randn(N, D, dtype=torch.float64)

    sampler = ContinuousTimeAR1HMMSampler(
        Q=Q_true,
        x_obs=x_obs,
        obs_times=T_obs,
        T=T,
        y_nuts_config={"warmup_steps": 16, "num_samples": 1, "max_tree_depth": 3, "target_accept_prob": 0.75},
        theta_nuts_config={"warmup_steps": 24, "num_samples": 1, "max_tree_depth": 3, "target_accept_prob": 0.75},
        seed=123,
    )
    sampler.initialize()
    history = sampler.run(num_sweeps=5, verbose=True)

    print("\nFinal theta sample:")
    print("mu =", history["theta"][-1]["mu"])
    print("lambda =", torch.exp(history["theta"][-1]["log_lambda"]))
    print("sigma =", torch.exp(history["theta"][-1]["log_sigma"]))
    print("tau_obs =", torch.exp(history["theta"][-1]["log_tau_obs"]))

    print("\nFinal inferred true path:")
    print("T_true =", history["T_true"][-1])
    print("z_true =", history["z_true"][-1])
