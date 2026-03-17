"""
Refactored ARHMM-NLE: Autoregressive Hidden Markov Model with Neural Likelihood Estimation
for Lotka-Volterra dynamics
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ..common.nle import NLEEstimator


class NFlowAR1HMMEstimator(nn.Module):
    """
    AR(1) Hidden Markov Model with Normalizing Flow emission model
    Supports multiple sequences of different lengths

    The emission probability is modeled by a normalizing flow:
    P(x_t | x_{t-1}, s_t) = flow.log_prob(x_t, context=[theta_{s_t}, x_{t-1}, elapsed_time])
    """

    def __init__(self,
                 nle_estimator: NLEEstimator,
                 n_states: int,
                 theta_init_dist: torch.distributions.Distribution,
                 learning_rate: float = 0.01,
                 elapsed_time_per_transition: float = 1.0,
                 device: str = 'cpu'):
        super(NFlowAR1HMMEstimator, self).__init__()

        self.n_states = n_states
        self.flow_model = nle_estimator.estimator
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.register_buffer(
            'elapsed_time_per_transition',
            torch.tensor(float(elapsed_time_per_transition), device=self.device),
        )

        # Initialize HMM parameters
        self._initialize_hmm_parameters()

        # Move to device
        self.to(self.device)
        if hasattr(self.flow_model, 'to'):
            self.flow_model.to(self.device)

        # Setup optimizer for all parameters
        self.theta_dim = nle_estimator.dynamics.theta_dim
        theta_tensor = theta_init_dist.sample((self.n_states, self.theta_dim))
        self.theta = nn.Parameter(theta_tensor)
        self.optimizer = optim.Adam([self.theta], lr=learning_rate)

    def _initialize_hmm_parameters(self):
        """Initialize HMM transition and initial state parameters."""
        self.register_buffer('log_pi', torch.log(torch.ones(self.n_states) / self.n_states))

        A = torch.distributions.Dirichlet(torch.ones(self.n_states)).sample((self.n_states,))
        self.register_buffer('log_A', torch.log(A))

    def _resolve_elapsed_time(
        self,
        elapsed_time: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return a scalar elapsed time tensor on the model device."""
        if elapsed_time is None:
            return self.elapsed_time_per_transition

        if isinstance(elapsed_time, torch.Tensor):
            elapsed_time = elapsed_time.to(self.device)
            if elapsed_time.numel() != 1:
                raise ValueError('elapsed_time must be a scalar value.')
            return elapsed_time.reshape(())

        return torch.tensor(float(elapsed_time), device=self.device)

    def _compute_log_emission_matrix(
        self,
        X: torch.Tensor,
        elapsed_time: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute emission log probabilities for all states and time steps."""
        T = X.shape[0]
        n_steps = T - 1
        elapsed_time_tensor = self._resolve_elapsed_time(elapsed_time).to(dtype=X.dtype)

        x_prev_batch = X[:n_steps].repeat_interleave(self.n_states, dim=0).contiguous()
        x_t_batch = X[1:].repeat_interleave(self.n_states, dim=0).contiguous()

        ctx_list = []
        for s in range(self.n_states):
            ctx_s_batch = self.theta[s].unsqueeze(0).repeat(n_steps, 1).contiguous()
            ctx_list.append(ctx_s_batch)

        ctx_batch = torch.stack(ctx_list, dim=1).reshape(-1, self.theta_dim)
        elapsed_column = elapsed_time_tensor.expand(ctx_batch.shape[0], 1)
        ctx_batch = torch.cat([ctx_batch, x_prev_batch, elapsed_column], dim=-1)

        log_probs = self.flow_model.log_prob(x_t_batch.unsqueeze(0), condition=ctx_batch)
        return log_probs.reshape(n_steps, self.n_states)

    def _forward_scaled(self, X: torch.Tensor, elapsed_time=None):
        T = X.shape[0]
        log_B = self._compute_log_emission_matrix(X, elapsed_time=elapsed_time)

        rows = []
        log_cs = []

        row0 = self.log_pi + log_B[0, :]
        c0 = torch.logsumexp(row0, dim=0)
        rows.append(row0 - c0)
        log_cs.append(c0)

        for t in range(1, T - 1):
            prev_hat = rows[-1]
            trans = prev_hat.unsqueeze(1) + self.log_A
            row_t = torch.logsumexp(trans, dim=0) + log_B[t, :]
            ct = torch.logsumexp(row_t, dim=0)
            rows.append(row_t - ct)
            log_cs.append(ct)

        return torch.stack(rows, dim=0), torch.stack(log_cs, dim=0)

    def _backward_scaled(self, X: torch.Tensor, log_c: torch.Tensor, elapsed_time=None):
        T = X.shape[0]
        nS = self.n_states
        log_B = self._compute_log_emission_matrix(X, elapsed_time=elapsed_time)

        rows_rev = []
        next_beta = torch.zeros(nS, device=self.device)
        rows_rev.append(next_beta)

        for t in range(T - 3, -1, -1):
            tmp = self.log_A + (log_B[t + 1, :] + next_beta).unsqueeze(0)
            beta_t = torch.logsumexp(tmp, dim=1) - log_c[t + 1]
            rows_rev.append(beta_t)
            next_beta = beta_t

        return torch.stack(list(reversed(rows_rev)), dim=0)

    def _compute_posteriors_single(self, X: torch.Tensor, elapsed_time=None):
        T = X.shape[0]
        log_B = self._compute_log_emission_matrix(X, elapsed_time=elapsed_time)
        log_alpha_hat, log_c = self._forward_scaled(X, elapsed_time=elapsed_time)
        log_beta_hat = self._backward_scaled(X, log_c, elapsed_time=elapsed_time)

        log_likelihood = torch.sum(log_c)

        log_gamma = log_alpha_hat + log_beta_hat
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)

        log_xi = torch.zeros(T - 2, self.n_states, self.n_states, device=self.device)
        for t in range(T - 2):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[t, i, j] = (
                        log_alpha_hat[t, i]
                        + self.log_A[i, j]
                        + log_B[t + 1, j]
                        + log_beta_hat[t + 1, j]
                        - log_c[t + 1]
                    )

            log_xi[t, :, :] = log_xi[t, :, :] - torch.logsumexp(log_xi[t, :, :].flatten(), dim=0)

        return log_gamma, log_xi, log_likelihood

    def _compute_posteriors_multiple(self, X_list, elapsed_time=None):
        log_gamma_list = []
        log_xi_list = []
        total_log_likelihood = 0

        for X in X_list:
            log_gamma, log_xi, log_likelihood = self._compute_posteriors_single(
                X, elapsed_time=elapsed_time)
            log_gamma_list.append(log_gamma)
            log_xi_list.append(log_xi)
            total_log_likelihood += log_likelihood

        self.log_gamma_list = log_gamma_list
        return log_gamma_list, log_xi_list, total_log_likelihood

    def _m_step_transitions(self, log_xi_list, log_gamma_list):
        with torch.no_grad():
            log_pi_acc = [log_gamma[0, :] for log_gamma in log_gamma_list]
            log_pi_stacked = torch.stack(log_pi_acc)
            self.log_pi.copy_(torch.logsumexp(log_pi_stacked, dim=0) - np.log(len(log_gamma_list)))

            for i in range(self.n_states):
                log_denominator_acc = [torch.logsumexp(log_gamma[:-1, i], dim=0) for log_gamma in log_gamma_list]
                log_denominator = torch.logsumexp(torch.stack(log_denominator_acc), dim=0)

                for j in range(self.n_states):
                    log_numerator_acc = []
                    for log_xi in log_xi_list:
                        if log_xi.shape[0] > 0:
                            log_numerator_acc.append(torch.logsumexp(log_xi[:, i, j], dim=0))

                    if log_numerator_acc:
                        log_numerator = torch.logsumexp(torch.stack(log_numerator_acc), dim=0)
                        self.log_A[i, j] = log_numerator - log_denominator

    def _m_step_flow_parameters(self, X_list, log_gamma_list, n_grad_steps=10, elapsed_time=None):
        _ = log_gamma_list
        for _ in range(n_grad_steps):
            self.optimizer.zero_grad()

            seq_ll_terms = []
            for X in X_list:
                _, log_c = self._forward_scaled(X, elapsed_time=elapsed_time)
                seq_ll_terms.append(log_c.sum())

            total_log_likelihood = torch.stack(seq_ll_terms).sum()
            loss = -total_log_likelihood
            loss.backward()
            self.optimizer.step()

    def fit(self,
            X_list: List[torch.Tensor],
            n_iter=100,
            n_grad_steps=10,
            tol=1e-4,
            verbose=False,
            elapsed_time: Optional[Union[float, torch.Tensor]] = None):

        X_list = [X.to(self.device) for X in X_list]
        elapsed_time_tensor = self._resolve_elapsed_time(elapsed_time)

        if verbose:
            print(f"Training on {len(X_list)} sequence(s)")
            print(f"Sequence lengths: {[X.shape[0] for X in X_list]}")

        log_likelihoods = []
        prev_log_likelihood = -float('inf')

        for iteration in range(n_iter):
            with torch.no_grad():
                log_gamma_list, log_xi_list, total_log_likelihood = self._compute_posteriors_multiple(
                    X_list, elapsed_time=elapsed_time_tensor)

            log_likelihood_val = total_log_likelihood.item()
            log_likelihoods.append(log_likelihood_val)

            if verbose:
                avg_ll = log_likelihood_val / sum(X.shape[0] - 1 for X in X_list)
                print(f"Iteration {iteration + 1}: Log-likelihood = {log_likelihood_val:.6f} (avg: {avg_ll:.6f})")

            if log_likelihood_val - prev_log_likelihood < tol:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            self._m_step_transitions(log_xi_list, log_gamma_list)
            self._m_step_flow_parameters(
                X_list, log_gamma_list, n_grad_steps, elapsed_time=elapsed_time_tensor)

            prev_log_likelihood = log_likelihood_val

        return log_likelihoods

    def predict(self,
                X_list: List[torch.Tensor],
                elapsed_time: Optional[Union[float, torch.Tensor]] = None):
        single_input = not isinstance(X_list, list)
        if single_input:
            X_list = [X_list]

        X_list = [X.to(self.device) for X in X_list]
        elapsed_time_tensor = self._resolve_elapsed_time(elapsed_time)

        states_list = []

        with torch.no_grad():
            for X in X_list:
                T = X.shape[0]
                log_B = self._compute_log_emission_matrix(X, elapsed_time=elapsed_time_tensor)

                log_delta = torch.zeros(T - 1, self.n_states, device=self.device)
                psi = torch.zeros(T - 1, self.n_states, dtype=torch.long, device=self.device)
                log_delta[0, :] = self.log_pi + log_B[0, :]

                for t in range(1, T - 1):
                    for j in range(self.n_states):
                        values = log_delta[t - 1, :] + self.log_A[:, j]
                        max_val, max_idx = torch.max(values, dim=0)
                        psi[t, j] = max_idx
                        log_delta[t, j] = max_val + log_B[t, j]

                states = torch.zeros(T - 1, dtype=torch.long, device=self.device)
                states[-1] = torch.argmax(log_delta[-1, :])
                for t in range(T - 3, -1, -1):
                    states[t] = psi[t + 1, states[t + 1]]

                states_list.append(states.cpu().numpy())

        return states_list[0] if single_input else states_list
