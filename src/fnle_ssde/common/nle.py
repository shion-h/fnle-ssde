
"""
Refactored ARHMM-NLE: Autoregressive Hidden Markov Model with Neural Likelihood Estimation
for Lotka-Volterra dynamics
"""

import time
from tqdm import tqdm
from typing import Optional, Tuple, List
import torch
import pyro.distributions as dist
from torch.distributions import Independent, Normal
from sbi.inference import SNLE
from .dynamics import Dynamics


class NLEEstimator:
    """Neural Likelihood Estimator for FNLE of SDE."""
    def __init__(self,
                 dynamics: Dynamics,
                 sampling_dist = None,
                 device: str = 'cpu'):
        self.dynamics = dynamics
        self.device = device
        self.estimator = None
        
        if sampling_dist is None:
            self.sampling_dist = dist.MultivariateNormal(
                torch.zeros(dynamics.theta_dim, device=device),
                torch.eye(dynamics.theta_dim, device=device)
            )
        else:
            self.sampling_dist = sampling_dist
    
    def generate_training_data_one_theta(
            self, 
            theta: torch.Tensor, 
            x_ref: torch.Tensor, 
            ref_noize: float,
            n_steps: int,
            elapsed_times: Optional[torch.Tensor] = None,
            ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate training data for one parameter sample.
        
        Args:
            theta: Parameter vector
            x_ref: Reference trajectory
            n_steps: Number of steps to simulate forward
            elapsed_times: Optional elapsed time for each transition in x_ref.
                Shape should be (len(x_ref) - 1,). If omitted, elapsed time
                is fixed to n_steps * dynamics.dt for all samples.
        """
        xt_chunk = []
        ctx_chunk = []

        if elapsed_times is None:
            elapsed_times = torch.full(
                (len(x_ref) - 1,),
                fill_value=float(n_steps * self.dynamics.dt),
                dtype=x_ref.dtype,
                device=x_ref.device,
            )
        else:
            elapsed_times = elapsed_times.to(device=x_ref.device, dtype=x_ref.dtype)

        if elapsed_times.shape[0] != len(x_ref) - 1:
            raise ValueError('elapsed_times must have shape (len(x_ref) - 1,)')

        theta = theta.to(device=x_ref.device, dtype=x_ref.dtype)
        
        for t in range(1, len(x_ref)):
            # Use first state parameters (assuming single regime for training)
            
            # Initial condition with small noise
            x_init = x_ref[t-1] + torch.randn_like(x_ref[t-1]) * ref_noize
            x_init = self.dynamics.to_device(x_init)
            
            # Simulate forward
            x_sim = x_init
            for _ in range(n_steps):
                x_sim = self.dynamics.simulate_one_step(x_sim, theta)
                x_sim = torch.clamp(x_sim, min=0.0, max=1e4)
            
            # Create context
            elapsed_time = elapsed_times[t-1].unsqueeze(0)
            context = torch.cat([theta, x_init, elapsed_time], dim=-1)
            
            xt_chunk.append(x_sim)
            ctx_chunk.append(context)
        
        return xt_chunk, ctx_chunk
    
    def generate_training_data(
            self, 
            n_params: int, 
            x_ref: torch.Tensor, 
            ref_noize: float,
            n_steps: int,
            elapsed_times: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate complete training dataset.
        
        Args:
            n_params: Number of parameter samples
            x_ref: Reference trajectories for training
        """
        all_xt = []
        all_ctx = []
        
        print(f"Generating training data from {n_params} parameter samples...")
        for _ in tqdm(range(n_params)):
            theta = self.sampling_dist.sample().cpu()
            xt_chunk, ctx_chunk = self.generate_training_data_one_theta(
                theta, x_ref, ref_noize, n_steps, elapsed_times)
            all_xt.extend(xt_chunk)
            all_ctx.extend(ctx_chunk)
        
        return (torch.stack(all_xt).to(self.device),
                torch.stack(all_ctx).to(self.device))
    
    def train(self, x_ref: torch.Tensor, n_params: int = 500, 
              ref_noize: float = 0.02,
              n_steps: int = 50,
              elapsed_times: Optional[torch.Tensor] = None,
              batch_size: int = 256, lr: float = 5e-4, 
              epochs: int = 50) -> 'NLEEstimator':
        """
        Train the NLE estimator.
        
        Args:
            x_ref: Reference trajectories
            n_params: Number of parameter samples for training
            batch_size: Training batch size
            lr: Learning rate
            epochs: Number of epochs
        """
        # Generate training data
        start_time = time.time()
        xt_data, ctx_data = self.generate_training_data(
            n_params, x_ref, ref_noize, n_steps, elapsed_times)
        print(f"Data generation took {time.time() - start_time:.2f}s,"
              f"samples: {len(ctx_data)}")
        
        # Store for debugging
        self.xt_data = xt_data
        self.ctx_data = ctx_data
        
        # Create context prior (dummy)
        context_prior = Independent(
            Normal(torch.zeros(self.ctx_data.shape[1], device=self.device),
                   torch.ones(self.ctx_data.shape[1], device=self.device)), 1)
        
        # Train NLE
        print("Training NLE estimator...")
        start_time = time.time()
        
        inf = SNLE(
            prior=context_prior,
            density_estimator='nsf',
            device=self.device,
            show_progress_bars=True
        )
        
        self.estimator = inf.append_simulations(
            theta=ctx_data, x=xt_data
        ).train(
            training_batch_size=batch_size,
            learning_rate=lr,
            max_num_epochs=epochs
        )
        
        self.estimator.eval()
        print(f"\nNLE training took {time.time() - start_time:.2f}s")
        
        return self
