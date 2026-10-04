"""OU transition densities and exact insertion bridges in the existing MCMC."""

from typing import Any

import torch

from .dynamics import ExactTransitionOUDynamics
from .sampler import SSDESampler


class ExactOUSSDESampler(SSDESampler):
    """Switching scalar OU with Gaussian observation noise.

    Y/theta use the parent's NUTS or MALA, regimes use FFBS, and Q/tau use
    the same conjugate updates. Only the transition backend and insertion
    bridge differ. The initial-state prior is the parent's explicit Gaussian,
    not an automatically selected OU stationary distribution.

    theta columns are (log kappa, mu, log sigma). ``sir_config`` retains its
    old-pseudo-boundary option; num_particles is unused in this subclass.
    """

    def __init__(self, Q, x_obs, obs_times, T, *, dynamics: ExactTransitionOUDynamics | None = None, **kwargs: Any):
        dynamics = ExactTransitionOUDynamics() if dynamics is None else dynamics
        if not isinstance(dynamics, ExactTransitionOUDynamics):
            raise TypeError("ExactOUSSDESampler requires ExactTransitionOUDynamics.")
        super().__init__(
            Q, x_obs, obs_times, T,
            dynamics=dynamics,
            transition_log_prob_fn=dynamics.transition_log_prob,
            transition_sample_fn=dynamics.sample_transition,
            **kwargs,
        )

    @torch.no_grad()
    def _sample_inserted_y_by_forward_sir(
        self, new_grid_times, new_z_aug, new_cand_idx, s=0,
    ):
        """Override the parent's insertion step with a joint exact OU bridge.

        Retained observation/true-jump states (and optionally old pseudo
        states) are fixed endpoints. Observations are not conditioned on a
        second time here: the subsequent Y update handles their likelihood.
        """
        info = {"num_particles": 0.0, "min_ess": float("nan"), "mean_ess": float("nan")}
        self.last_sir_cand_particles = []
        old_y = self.y_aug_list[s]
        if old_y is None or self.T_all_list[s] is None:
            return self._initialize_y_on_grid(s, new_grid_times), info
        if self.theta is None:
            raise RuntimeError("Initialize theta before sampling OU bridges.")
        times, is_old_pseudo, is_new = self._build_sir_work_grid(
            s, new_grid_times[new_cand_idx]
        )
        regimes = self._expand_true_path_onto_grid(s, times)
        y = torch.empty((len(times), self.D), dtype=self.dtype, device=self.device)
        if not self.sir_config["use_t_pseudo_in_sir"]:
            retain = torch.ones(len(old_y), dtype=torch.bool, device=self.device)
            retain[self.pseudo_idx_list[s]] = False
            old_y = old_y[retain]
        y[~is_new] = old_y
        boundaries = torch.nonzero(~is_new, as_tuple=False).flatten().tolist()
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right == left + 1:
                continue
            y[left + 1:right] = self.dynamics.sample_bridge(
                y[left], y[right], times[left:right + 1],
                self.theta[regimes[left + 1:right + 1]],
            )
        return y[~is_old_pseudo], info
