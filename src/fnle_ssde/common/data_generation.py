import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from .dynamics import Dynamics

class SwitchingSDEDataGenerator:
    """Autoregressive Hidden Markov Model with SDE dynamics."""
    
    def __init__(self, dynamics: Dynamics, device: str = 'cpu'):
        self.dynamics = dynamics
        self.device = device
    
    def generate(
        self, 
        params: Dict[str, torch.Tensor], 
        T: int, 
        x0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate a trajectory from the ARHMM.
        
        Args:
            params: Model parameters
            T: Number of time steps
            x0: Initial state (default: [1.0, 0.5])
        
        Returns:
            x: Observations (T, obs_dim)
            z: Hidden states (T,)
        """
        thetas = params['emissions']
        trans_probs = F.softmax(params["trans_logits"], dim=-1)
        init_probs = F.softmax(params["init_logits"], dim=-1)
        
        # Initialize arrays
        x = torch.zeros(T, self.dynamics.x_dim, device=self.device)
        z = torch.zeros(T, dtype=torch.long, device=self.device)
        
        # Initial conditions
        x[0] = x0
        z[0] = torch.multinomial(init_probs, 1).item()
        
        # Generate trajectory
        for t in range(1, T):
            # Sample next hidden state
            z[t] = torch.multinomial(trans_probs[z[t-1]], 1).item()
            # Simulate dynamics
            x[t] = self.dynamics.simulate_one_step(x[t-1], thetas[z[t].item()])
            # Clip values to prevent numerical instability
            x[t] = torch.clamp(x[t], min=0.0, max=1e4)

        return x, z


def generate_true_and_obs_data(generator,
                               num_series,
                               hmm_param_true,
                               n_of_all_steps,
                               X0,
                               obs_interval):
    X_obs = []
    Z_obs = []
    X_true = []
    Z_true = []
    obs_idx_list = []

    for i in range(num_series):
        x, z = generator.generate(hmm_param_true, n_of_all_steps, x0=X0)
        # Subsample
        obs_idx = np.arange(len(x))[::obs_interval]
        X_obs.append(x[::obs_interval])
        Z_obs.append(z[::obs_interval])
        X_true.append(x)
        Z_true.append(z)
        obs_idx_list.append(obs_idx)
        print(f"  Series {i+1}: {len(X_obs[-1])} observations")
    return X_obs, Z_obs, X_true, Z_true, obs_idx_list


def generate_data_of_state(nle, state, hmm_param_true, X_obs_concat,
                           n_steps_for_each_interval):
    # generate the next state by simulating forward from each observed state
    targets = []
    for x_i in X_obs_concat:
        x_sim = x_i.clone()
        for _ in range(n_steps_for_each_interval):
            x_sim = nle.dynamics.simulate_one_step(x_sim, hmm_param_true['emissions'][state])
            x_sim = torch.clamp(x_sim, min=0.0, max=1e4)
        targets.append(x_sim)
    return torch.stack(targets).clone()
