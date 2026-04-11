from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ..common.nle import NLEEstimator


class NFlowAR1HMMGibbsSampler(nn.Module):
    """
    AR(1) Hidden Markov Model with Normalizing Flow emission model
    Supports multiple sequences of different lengths
    
    The emission probability is modeled by a normalizing flow:
    P(x_t | y_t, n_step, s_t) = flow.log_prob(
        x_t, context=[theta_{s_t}, y_t, n_step]
    )
    """
        
    def __init__(self, 
                 nle_estimator: NLEEstimator, 
                 n_states: int,
                 theta_init_dist: torch.distributions.Distribution, 
                 n_step: int = 1,
                 learning_rate: float=0.01, 
                 device: str = 'cpu'):
        super(NFlowAR1HMMGibbsSampler, self).__init__()
        
        self.n_states = n_states
        self.flow_model = nle_estimator.estimator
        self.conditions_on_n_steps = nle_estimator.conditions_on_n_steps
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        if n_step < 1:
            raise ValueError("n_step must be a positive integer.")
        self.n_step = n_step
        
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
        # Adam使わないとemis_paramの最適化は本当に無理？
        self.optimizer = optim.Adam([self.theta], lr=learning_rate)
    
    def _initialize_hmm_parameters(self):
        """Initialize HMM transition and initial state parameters"""
        # Initial state probabilities (log space)
        self.register_buffer('log_pi', torch.log(torch.ones(self.n_states) / self.n_states))
        
        # Transition probabilities (log space)
        A = torch.distributions.Dirichlet(torch.ones(self.n_states)).sample((self.n_states,))
        self.register_buffer('log_A', torch.log(A))
    
    def _compute_log_emission_matrix(self, Y, n_steps):
        """
        Compute log emission probability matrix using normalizing flow
        
        Parameters:
        -----------
        Y : torch.Tensor, shape (T, n_features)
            Single observation sequence
        n_steps : torch.Tensor | list | np.ndarray, shape (T-1,)
            Number of latent simulation steps for each interval.
        
        Returns:
        --------
        log_B : torch.Tensor, shape (T-1, n_states)
            Log emission probabilities
            B_{t, s} = p(y_t | y_{t-1}, theta, z_t=s, Delta_{t-1})
        """
        """Compute emission log probabilities for all states and time steps."""
        T = Y.shape[0]
        n_intervals = T - 1
        log_B = torch.zeros(n_intervals, self.n_states, device=self.device)

        n_steps = torch.as_tensor(
            n_steps,
            device=self.device,
            dtype=Y.dtype,
        ).reshape(-1)
        if n_steps.shape[0] != n_intervals:
            raise ValueError(
                "n_steps must have length len(Y) - 1."
            )
        
        # Prepare batch inputs.
        # NLEEstimator uses [theta, y, n_step] as context, where y is the
        # initial state from which the simulator evolves for Delta_{t-1} steps.
        y_prev_batch = Y[:n_intervals].repeat_interleave(self.n_states, dim=0).contiguous()
        y_batch = Y[1:].repeat_interleave(self.n_states, dim=0).contiguous()
        
        # Create context batch for all states
        ctx_list = []
        for s in range(self.n_states):
            ctx_s_batch = self.theta[s].unsqueeze(0).repeat(n_intervals, 1).contiguous()
            ctx_list.append(ctx_s_batch)
        
        # (n_states, t) -> (n_states * t)
        ctx_batch = torch.stack(ctx_list, dim=1).reshape(-1, self.theta_dim)
        ctx_batch = torch.cat([ctx_batch, y_prev_batch], dim=-1)
        if self.conditions_on_n_steps:
            n_step_batch = n_steps.repeat_interleave(self.n_states).unsqueeze(-1)
            n_step_batch = n_step_batch.to(
                device=self.device,
                dtype=ctx_batch.dtype,
            )
            ctx_batch = torch.cat([ctx_batch, n_step_batch], dim=-1)
        
        # Compute log probabilities using NLE
        log_probs = self.flow_model.log_prob(y_batch.unsqueeze(0), condition=ctx_batch)
        # (n_states * t) -> (n_states, t)
        log_B = log_probs.reshape(n_intervals, self.n_states)
        
        return log_B
    
    def _forward_scaled(self, X):
        T = X.shape[0]
        nS = self.n_states
    
        # 1回だけ計算
        log_B = self._compute_log_emission_matrix(X)  # (T-1, nS)
    
        rows = []
        log_cs = []
    
        # t=0
        row0 = self.log_pi + log_B[0, :]                  # (nS,)
        c0   = torch.logsumexp(row0, dim=0)
        row0_hat = row0 - c0
        rows.append(row0_hat)
        log_cs.append(c0)
    
        # t>=1
        for t in range(1, T - 1):
            prev_hat = rows[-1]                            # (nS,)
            # for each j: logsumexp_i prev_hat[i] + log_A[i,j]
            trans = prev_hat.unsqueeze(1) + self.log_A     # (nS, nS)
            row_t = torch.logsumexp(trans, dim=0) + log_B[t, :]  # (nS,)
            ct    = torch.logsumexp(row_t, dim=0)
            rows.append(row_t - ct)
            log_cs.append(ct)
    
        log_alpha_hat = torch.stack(rows, dim=0)           # (T-1, nS)
        log_c = torch.stack(log_cs, dim=0)                 # (T-1,)
        return log_alpha_hat, log_c
        
    def _backward_scaled(self, X, log_c):
        T = X.shape[0]
        nS = self.n_states
        log_B = self._compute_log_emission_matrix(X)       # (T-1, nS)
    
        rows_rev = []
        # 最後はゼロ行
        next_beta = torch.zeros(nS, device=self.device)
        rows_rev.append(next_beta)
    
        # 逆向きに畳む
        for t in range(T - 3, -1, -1):
            # β_t[i] = logsumexp_j ( log_A[i,j] + log_B[t+1, j] + β_{t+1}[j] ) - log_c[t+1]
            tmp = self.log_A + (log_B[t+1, :] + next_beta).unsqueeze(0)  # (nS, nS)
            beta_t = torch.logsumexp(tmp, dim=1) - log_c[t+1]            # (nS,)
            rows_rev.append(beta_t)
            next_beta = beta_t
    
        # rows_rev: [β_{T-2}, β_{T-3}, ..., β_0]（最後のゼロ行を末尾に持つ）
        log_beta_hat = torch.stack(list(reversed(rows_rev)), dim=0)      # (T-1, nS)
        return log_beta_hat


    
    def _compute_posteriors_single(self, X):
        """
        Compute posterior probabilities for a single sequence
        
        Parameters:
        -----------
        X : torch.Tensor, shape (T, n_features)
            Single observation sequence
        
        Returns:
        --------
        log_gamma : torch.Tensor, shape (T-1, n_states)
            Log posterior probabilities of states
        log_xi : torch.Tensor, shape (T-2, n_states, n_states)
            Log posterior probabilities of state transitions
        log_likelihood : torch.Tensor
            Log likelihood of the sequence
        """
        T = X.shape[0]
        log_B = self._compute_log_emission_matrix(X)
        
        # Forward pass
        log_alpha_hat, log_c = self._forward_scaled(X)
        
        # Backward pass
        log_beta_hat = self._backward_scaled(X, log_c)
        
        # Compute log likelihood
        log_likelihood = torch.sum(log_c)
        
        # Compute log gamma
        log_gamma = log_alpha_hat + log_beta_hat
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
        
        # Compute log xi
        log_xi = torch.zeros(T - 2, self.n_states, self.n_states, device=self.device)
        for t in range(T - 2):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[t, i, j] = (log_alpha_hat[t, i] + self.log_A[i, j] + 
                                      log_B[t+1, j] + log_beta_hat[t+1, j] - log_c[t+1])
            
            log_xi[t, :, :] = log_xi[t, :, :] - torch.logsumexp(log_xi[t, :, :].flatten(), dim=0)
        
        return log_gamma, log_xi, log_likelihood
    
    def _compute_posteriors_multiple(self, X_list):
        """
        Compute posterior probabilities for multiple sequences
        
        Parameters:
        -----------
        X_list : List[torch.Tensor]
            List of observation sequences
        
        Returns:
        --------
        log_gamma_list : List[torch.Tensor]
            List of log posterior probabilities of states
        log_xi_list : List[torch.Tensor]
            List of log posterior probabilities of state transitions
        total_log_likelihood : torch.Tensor
            Total log likelihood across all sequences
        """
        log_gamma_list = []
        log_xi_list = []
        total_log_likelihood = 0
        
        for X in X_list:
            log_gamma, log_xi, log_likelihood = self._compute_posteriors_single(X)
            log_gamma_list.append(log_gamma)
            log_xi_list.append(log_xi)
            total_log_likelihood += log_likelihood
    
        self.log_gamma_list = log_gamma_list
        
        return log_gamma_list, log_xi_list, total_log_likelihood
    
    def _m_step_transitions(self, log_xi_list, log_gamma_list):
        """
        M-step for transition and initial state probabilities with multiple sequences
        
        Parameters:
        -----------
        log_xi_list : List[torch.Tensor]
            List of log posterior probabilities of state transitions
        log_gamma_list : List[torch.Tensor]
            List of log posterior probabilities of states
        """
        with torch.no_grad():
            # Update initial state probabilities (average across sequences)
            log_pi_acc = []
            for log_gamma in log_gamma_list:
                log_pi_acc.append(log_gamma[0, :])
            
            # Stack and compute log-sum-exp across sequences
            log_pi_stacked = torch.stack(log_pi_acc)
            self.log_pi.copy_(torch.logsumexp(log_pi_stacked, dim=0) - np.log(len(log_gamma_list)))
            
            # Update transition probabilities
            for i in range(self.n_states):
                # Accumulate across all sequences
                log_denominator_acc = []
                for log_gamma in log_gamma_list:
                    log_denominator_acc.append(torch.logsumexp(log_gamma[:-1, i], dim=0))
                log_denominator = torch.logsumexp(torch.stack(log_denominator_acc), dim=0)
                
                for j in range(self.n_states):
                    log_numerator_acc = []
                    for log_xi in log_xi_list:
                        if log_xi.shape[0] > 0:  # Check for sequences with at least 2 transitions
                            log_numerator_acc.append(torch.logsumexp(log_xi[:, i, j], dim=0))
                    
                    if log_numerator_acc:
                        log_numerator = torch.logsumexp(torch.stack(log_numerator_acc), dim=0)
                        self.log_A[i, j] = log_numerator - log_denominator
    
    def _m_step_flow_parameters(self, X_list, log_gamma_list, n_grad_steps=10):
        """
        M-step for normalizing flow parameters with multiple sequences
        Directly maximizes the log-likelihood using forward algorithm
        
        Parameters:
        -----------
        X_list : List[torch.Tensor]
            List of observation sequences
        log_gamma_list : List[torch.Tensor]
            List of log posterior probabilities of states (unused but kept for consistency)
        n_grad_steps : int
            Number of gradient steps
        """
        for step in range(n_grad_steps):
            self.optimizer.zero_grad()
    
            seq_ll_terms = []
            for X in X_list:
                _, log_c = self._forward_scaled(X)
                seq_ll_terms.append(log_c.sum())
    
            total_log_likelihood = torch.stack(seq_ll_terms).sum()
            loss = -total_log_likelihood
            loss.backward()
            self.optimizer.step()
    
    def fit(self, X_list: List[torch.Tensor], 
            n_iter=100, n_grad_steps=10, tol=1e-4, verbose=False):
        """
        Fit the model using EM algorithm with multiple sequences
        
        Parameters:
        -----------
        X_obs : List[torch.Tensor], List[np.ndarray], torch.Tensor, or np.ndarray
            Observation sequence(s). Can be:
            - List of tensors/arrays for multiple sequences
            - Single tensor/array for one sequence
        n_iter : int
            Maximum number of EM iterations
        n_grad_steps : int
            Number of gradient steps for flow parameters per M-step
        tol : float
            Convergence tolerance
        verbose : bool
            Print progress information
        
        Returns:
        --------
        log_likelihoods : list
            Log-likelihood values at each iteration
        """

        X_list = [X.to(self.device) for X in X_list]
        
        if verbose:
            print(f"Training on {len(X_list)} sequence(s)")
            print(f"Sequence lengths: {[X.shape[0] for X in X_list]}")
        
        log_likelihoods = []
        prev_log_likelihood = -float('inf')
        
        for iteration in range(n_iter):
            # E-step
            with torch.no_grad():
                log_gamma_list, log_xi_list, total_log_likelihood = self._compute_posteriors_multiple(X_list)
            
            log_likelihood_val = total_log_likelihood.item()
            log_likelihoods.append(log_likelihood_val)
            
            if verbose:
                avg_ll = log_likelihood_val / sum(X.shape[0] - 1 for X in X_list)
                print(f"Iteration {iteration + 1}: Log-likelihood = {log_likelihood_val:.6f} (avg: {avg_ll:.6f})")
            
            # Check convergence
            if log_likelihood_val - prev_log_likelihood < tol:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            
            # M-step
            self._m_step_transitions(log_xi_list, log_gamma_list)
            self._m_step_flow_parameters(X_list, log_gamma_list, n_grad_steps)
            
            prev_log_likelihood = log_likelihood_val
        
        return log_likelihoods
    
    def predict(self, X_list: List[torch.Tensor]):
        """
        Find the most likely state sequences using Viterbi algorithm
        
        Parameters:
        -----------
        X_obs : List[torch.Tensor], List[np.ndarray], torch.Tensor, or np.ndarray
            Observation sequence(s)
        
        Returns:
        --------
        states : List[np.ndarray] or np.ndarray
            Most likely state sequence(s)
            Returns single array if input was single sequence
        """
        single_input = not isinstance(X_list, list)
        if single_input:
            X_list = [X_list]

        X_list = [X.to(self.device) for X in X_list]
        
        states_list = []
        
        with torch.no_grad():
            for X in X_list:
                # Viterbi algorithm
                T = X.shape[0]
                log_B = self._compute_log_emission_matrix(X)
                
                # Initialize
                log_delta = torch.zeros(T - 1, self.n_states, device=self.device)
                psi = torch.zeros(T - 1, self.n_states, dtype=torch.long, device=self.device)
                log_delta[0, :] = self.log_pi + log_B[0, :]
                
                # Recursion
                for t in range(1, T - 1):
                    for j in range(self.n_states):
                        values = log_delta[t-1, :] + self.log_A[:, j]
                        max_val, max_idx = torch.max(values, dim=0)
                        psi[t, j] = max_idx
                        log_delta[t, j] = max_val + log_B[t, j]
                
                # Backtrack
                states = torch.zeros(T - 1, dtype=torch.long, device=self.device)
                states[-1] = torch.argmax(log_delta[-1, :])
                for t in range(T - 3, -1, -1):
                    states[t] = psi[t + 1, states[t + 1]]
                
                states_list.append(states.cpu().numpy())
        
        return states_list[0] if single_input else states_list
