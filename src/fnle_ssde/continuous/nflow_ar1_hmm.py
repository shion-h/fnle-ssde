from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..common.nle import NLEEstimator


class ContinuousNFlowAR1HMMGibbs(nn.Module):
    """
    Continuous-time AR(1)-HMM with observation error.

    Notation:
      - X: observed variable
      - Y: continuous latent variable
      - Z: discrete latent variable

    Transition density p(Y_t | Y_{t-1}, Z_t=s) is obtained in the same way as
    `discrete/nflow_ar1_hmm.py`, i.e. via normalizing-flow log-probability:
      flow.log_prob(Y_t, condition=[theta_s, Y_{t-1}])

    Z is sampled with pseudo-jumps + uniformization.
    """

    def __init__(
        self,
        nle_estimator: NLEEstimator,
        n_states: int,
        theta_init_dist: torch.distributions.Distribution,
        sigma_obs: float = 0.1,
        learning_rate: float = 0.01,
        uniformization_rate: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_states = n_states
        self.flow_model = nle_estimator.estimator
        self.device = torch.device(device)
        self.learning_rate = learning_rate

        self.theta_dim = nle_estimator.dynamics.theta_dim
        theta_tensor = theta_init_dist.sample((self.n_states, self.theta_dim))
        self.theta = nn.Parameter(theta_tensor)

        self.log_sigma_obs = nn.Parameter(torch.tensor(np.log(sigma_obs), dtype=torch.float32))

        self._initialize_ctmc_parameters()
        self.uniformization_rate = uniformization_rate

        self.to(self.device)
        if hasattr(self.flow_model, "to"):
            self.flow_model.to(self.device)

    def _initialize_ctmc_parameters(self):
        self.register_buffer("pi", torch.ones(self.n_states) / self.n_states)
        q = torch.ones(self.n_states, self.n_states, dtype=torch.float32)
        q.fill_diagonal_(0.0)
        q = q / (self.n_states - 1)
        q = q * 1.0
        q.fill_diagonal_(-q.sum(dim=1))
        self.register_buffer("Q", q)

    @property
    def sigma_obs(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_obs)

    def _ensure_2d(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 1:
            return value.unsqueeze(-1)
        return value

    def _compute_log_emission_matrix(self, Y: torch.Tensor) -> torch.Tensor:
        Y = self._ensure_2d(Y)
        T = Y.shape[0]
        n_steps = T - 1

        y_prev_batch = Y[:n_steps].repeat_interleave(self.n_states, dim=0).contiguous()
        y_t_batch = Y[1:].repeat_interleave(self.n_states, dim=0).contiguous()

        ctx_list = []
        for s in range(self.n_states):
            ctx_s_batch = self.theta[s].unsqueeze(0).repeat(n_steps, 1).contiguous()
            ctx_list.append(ctx_s_batch)

        ctx_batch = torch.stack(ctx_list, dim=1).reshape(-1, self.theta_dim)
        ctx_batch = torch.cat([ctx_batch, y_prev_batch], dim=-1)

        log_probs = self.flow_model.log_prob(y_t_batch.unsqueeze(0), condition=ctx_batch)
        return log_probs.reshape(n_steps, self.n_states)

    def _log_obs_likelihood(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = self._ensure_2d(X)
        Y = self._ensure_2d(Y)
        diff = X - Y
        var = self.sigma_obs ** 2
        c = -0.5 * torch.log(2 * torch.tensor(np.pi, device=self.device) * var)
        ll = c - 0.5 * diff.pow(2) / var
        return ll.sum(dim=-1)

    def _uniformization_transition_mats(self, dt: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        exit_rates = -torch.diag(self.Q)
        omega = self.uniformization_rate
        min_omega = torch.max(exit_rates).item() + 1e-6
        if omega is None:
            omega = 1.2 * min_omega
        omega = max(float(omega), min_omega)

        B = torch.eye(self.n_states, device=self.device) + self.Q / omega
        B = torch.clamp(B, min=1e-12)
        B = B / B.sum(dim=1, keepdim=True)

        mats: List[torch.Tensor] = []
        n_pseudo = torch.zeros(len(dt), dtype=torch.long, device=self.device)
        for i, dti in enumerate(dt):
            m_i = torch.poisson(torch.tensor(float(omega * dti.item()), device=self.device)).long()
            n_i = m_i + 1
            n_pseudo[i] = m_i
            mats.append(torch.matrix_power(B, int(n_i.item())))

        return mats, n_pseudo

    def _sample_Z_uniformization(self, Y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        Y = self._ensure_2d(Y)
        T = Y.shape[0]
        log_B = self._compute_log_emission_matrix(Y)
        trans_mats, _ = self._uniformization_transition_mats(dt)

        log_alpha = torch.zeros(T - 1, self.n_states, device=self.device)
        log_alpha[0] = torch.log(self.pi + 1e-12) + log_B[0]

        for t in range(1, T - 1):
            trans = torch.log(trans_mats[t - 1] + 1e-12)
            log_alpha[t] = torch.logsumexp(log_alpha[t - 1].unsqueeze(1) + trans, dim=0) + log_B[t]

        Z = torch.zeros(T - 1, dtype=torch.long, device=self.device)
        probs_last = torch.softmax(log_alpha[-1], dim=0)
        Z[-1] = torch.multinomial(probs_last, num_samples=1)

        for t in range(T - 3, -1, -1):
            trans_t = trans_mats[t]
            log_p = log_alpha[t] + torch.log(trans_t[:, Z[t + 1]] + 1e-12)
            probs = torch.softmax(log_p, dim=0)
            Z[t] = torch.multinomial(probs, num_samples=1)

        return Z

    def _single_site_update_Y(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
        proposal_scale: float,
    ) -> torch.Tensor:
        X = self._ensure_2d(X)
        Y = self._ensure_2d(Y)
        T, d = Y.shape
        var = self.sigma_obs ** 2

        for t in range(T):
            y_old = Y[t].clone()
            y_new = y_old + proposal_scale * torch.randn(d, device=self.device)

            ll_obs_old = (-0.5 * (X[t] - y_old).pow(2) / var).sum()
            ll_obs_new = (-0.5 * (X[t] - y_new).pow(2) / var).sum()

            ll_tr_old = torch.tensor(0.0, device=self.device)
            ll_tr_new = torch.tensor(0.0, device=self.device)

            if t >= 1:
                s_t = int(Z[t - 1].item())
                ctx_old = torch.cat([self.theta[s_t], Y[t - 1]], dim=-1)
                ctx_new = ctx_old
                ll_tr_old = ll_tr_old + self.flow_model.log_prob(y_old.unsqueeze(0), condition=ctx_old.unsqueeze(0))
                ll_tr_new = ll_tr_new + self.flow_model.log_prob(y_new.unsqueeze(0), condition=ctx_new.unsqueeze(0))
            else:
                ll_tr_old = ll_tr_old - 0.5 * (y_old - X[0]).pow(2).sum() / (10.0**2)
                ll_tr_new = ll_tr_new - 0.5 * (y_new - X[0]).pow(2).sum() / (10.0**2)

            if t <= T - 2:
                s_next = int(Z[t].item())
                ctx_old = torch.cat([self.theta[s_next], y_old], dim=-1)
                ctx_new = torch.cat([self.theta[s_next], y_new], dim=-1)
                y_next = Y[t + 1]
                ll_tr_old = ll_tr_old + self.flow_model.log_prob(y_next.unsqueeze(0), condition=ctx_old.unsqueeze(0))
                ll_tr_new = ll_tr_new + self.flow_model.log_prob(y_next.unsqueeze(0), condition=ctx_new.unsqueeze(0))

            log_accept = (ll_obs_new + ll_tr_new) - (ll_obs_old + ll_tr_old)
            if torch.log(torch.rand(1, device=self.device)) < log_accept:
                Y[t] = y_new

        return Y

    def _sample_Q(self, Z: torch.Tensor, dt: torch.Tensor, alpha: float = 1.0, beta: float = 1.0):
        K = self.n_states
        N = torch.zeros(K, K, device=self.device)
        dwell = torch.zeros(K, device=self.device)

        for t in range(len(Z)):
            dwell[int(Z[t].item())] += dt[t]
        for t in range(len(Z) - 1):
            i = int(Z[t].item())
            j = int(Z[t + 1].item())
            if i != j:
                N[i, j] += 1

        Q_new = torch.zeros_like(self.Q)
        for i in range(K):
            rates = []
            for j in range(K):
                if i == j:
                    continue
                shape = alpha + N[i, j]
                rate = beta + dwell[i]
                q_ij = torch.distributions.Gamma(shape, rate).sample().to(self.device)
                rates.append((j, q_ij))
            row_sum = torch.tensor(0.0, device=self.device)
            for j, val in rates:
                Q_new[i, j] = val
                row_sum += val
            Q_new[i, i] = -row_sum

        with torch.no_grad():
            self.Q.copy_(Q_new)

    def _sample_sigma_obs(self, X: torch.Tensor, Y: torch.Tensor, a0: float = 2.0, b0: float = 0.1):
        X = self._ensure_2d(X)
        Y = self._ensure_2d(Y)
        n = X.numel()
        rss = (X - Y).pow(2).sum()
        a_post = a0 + 0.5 * n
        b_post = b0 + 0.5 * rss
        tau = torch.distributions.Gamma(a_post, b_post).sample().to(self.device)
        sigma2 = 1.0 / tau
        with torch.no_grad():
            self.log_sigma_obs.copy_(0.5 * torch.log(sigma2))

    def _sample_theta_mh(
        self,
        Y: torch.Tensor,
        Z: torch.Tensor,
        proposal_scale: float = 0.05,
    ):
        with torch.no_grad():
            for s in range(self.n_states):
                idx = (Z == s).nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue

                theta_old = self.theta[s].detach().clone()
                theta_new = theta_old + proposal_scale * torch.randn_like(theta_old)

                ll_old = torch.tensor(0.0, device=self.device)
                ll_new = torch.tensor(0.0, device=self.device)

                for t in idx:
                    y_prev = Y[t]
                    y_cur = Y[t + 1]
                    ctx_old = torch.cat([theta_old, y_prev], dim=-1)
                    ctx_new = torch.cat([theta_new, y_prev], dim=-1)
                    ll_old = ll_old + self.flow_model.log_prob(y_cur.unsqueeze(0), condition=ctx_old.unsqueeze(0))
                    ll_new = ll_new + self.flow_model.log_prob(y_cur.unsqueeze(0), condition=ctx_new.unsqueeze(0))

                log_accept = ll_new - ll_old
                if torch.log(torch.rand(1, device=self.device)) < log_accept:
                    self.theta[s].copy_(theta_new)

    def fit(
        self,
        X: torch.Tensor,
        times: torch.Tensor,
        n_iter: int = 1000,
        burn_in: int = 200,
        thin: int = 5,
        proposal_scale_y: float = 0.05,
        proposal_scale_theta: float = 0.02,
        sample_Q: bool = True,
        sample_theta: bool = True,
        verbose: bool = False,
    ) -> Dict[str, List[torch.Tensor]]:
        X = self._ensure_2d(X.to(self.device))
        times = times.to(self.device)
        dt = times[1:] - times[:-1]
        if torch.any(dt <= 0):
            raise ValueError("times must be strictly increasing")

        Y = X.clone()
        Z = torch.randint(0, self.n_states, (X.shape[0] - 1,), device=self.device)

        samples_Y: List[torch.Tensor] = []
        samples_Z: List[torch.Tensor] = []
        samples_Q: List[torch.Tensor] = []
        samples_theta: List[torch.Tensor] = []
        samples_sigma_obs: List[torch.Tensor] = []

        for it in range(n_iter):
            Y = self._single_site_update_Y(X, Y, Z, proposal_scale=proposal_scale_y)
            Z = self._sample_Z_uniformization(Y, dt)

            if sample_theta:
                self._sample_theta_mh(Y, Z, proposal_scale=proposal_scale_theta)
            if sample_Q:
                self._sample_Q(Z, dt)
            self._sample_sigma_obs(X, Y)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                samples_Y.append(Y.detach().clone().cpu())
                samples_Z.append(Z.detach().clone().cpu())
                samples_Q.append(self.Q.detach().clone().cpu())
                samples_theta.append(self.theta.detach().clone().cpu())
                samples_sigma_obs.append(self.sigma_obs.detach().clone().cpu())

            if verbose and ((it + 1) % max(1, n_iter // 10) == 0):
                print(
                    f"iter={it+1} sigma_obs={self.sigma_obs.item():.4f} "
                    f"mean|X-Y|={(X - Y).abs().mean().item():.4f}"
                )

        return {
            "Y": samples_Y,
            "Z": samples_Z,
            "Q": samples_Q,
            "theta": samples_theta,
            "sigma_obs": samples_sigma_obs,
        }

    def posterior_state_prob(self, Z_samples: List[torch.Tensor]) -> torch.Tensor:
        if len(Z_samples) == 0:
            raise ValueError("Z_samples is empty")

        z_stack = torch.stack([z.to(self.device) for z in Z_samples], dim=0)
        Tm1 = z_stack.shape[1]
        gamma = torch.zeros(Tm1, self.n_states, device=self.device)

        for s in range(self.n_states):
            gamma[:, s] = (z_stack == s).float().mean(dim=0)

        return gamma
