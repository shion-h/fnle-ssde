from abc import ABC, abstractmethod
import torch
from typing import Tuple

class Dynamics(ABC):
    def __init__(self, dt, device):
        self.dt = dt
        self.device = device
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device."""
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def simulate_one_step(
            self, x: torch.Tensor, 
            theta: torch.Tensor) -> torch.Tensor:
        # split_theta使う前提なのにthetaがtuple
        """
        Simulate one step of the SDE.
        
        Args:
            x: Current state
            emission_params: (A matrix, log std) for current HMM state
        """
        theta_drift, theta_diffusion = self.split_theta(theta)
        theta_drift = self.to_device(theta_drift)
        theta_diffusion = self.to_device(theta_diffusion)
        x = self.to_device(x)
        
        # Compute drift and diffusion terms
        drift = self.drift(x, theta_drift)
        diffusion = self.diffusion(x, theta_diffusion)
        
        # Brownian motion increment
        dW = (
            torch.randn(x.shape[0], device=self.device)
            * torch.sqrt(torch.tensor(self.dt, device=self.device))
        )        
        # Euler-Maruyama update
        return x + drift * self.dt + diffusion @ dW

    def simulate(self, x: torch.Tensor, 
                 theta: torch.Tensor,
                 n_step: int):
        this_x = x
        x_list = []
        for _ in range(n_step):
            this_x = self.simulate_one_step(this_x, theta)
            x_list.append(this_x)
        return x_list

    @abstractmethod
    def split_theta(self, theta: torch.Tensor) -> Tuple[torch.Tensor]:
        pass

    @abstractmethod
    def drift(self, x: torch.Tensor,
              theta_drift: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def diffusion(self, x: torch.Tensor,
                  theta_diffusion: torch.Tensor) -> torch.Tensor:
        pass


class LotkaVolterraDynamics(Dynamics):
    """Lotka-Volterra predator-prey dynamics."""
    def __init__(self, dt: float = 0.01, 
                 device: str = 'cpu'):
        super().__init__(dt, device)
        self.x_dim = 2  # predator and prey
        self.theta_dim = 6  # 4 drift param and 2 diffusion param
    
    def split_theta(self, theta: torch.Tensor) -> Tuple[torch.Tensor]:
        return (theta[:4], theta[4:6])

    def drift(self, x: torch.Tensor,
              theta_drift: torch.Tensor) -> torch.Tensor:
        # thetaは実数で与えられる想定→ここでexp使う
        """
        Compute drift term for Lotka-Volterra SDE.
        
        Args:
            x: State vector [predator, prey]
            params: Parameters [alpha, beta, gamma, delta]
            device: Device for computation
        """
        assert theta_drift.shape[0] == 4, "Expected 4 parameters for drift (alpha, beta, gamma, delta)"
        alpha, beta, gamma, delta = torch.exp(theta_drift)
        predator, prey = x[0], x[1]
        
        f_predator = gamma * prey * predator - delta * predator
        f_prey = alpha * prey - beta * prey * predator
        
        return torch.tensor([f_predator, f_prey],
                            device=self.device,
                            dtype=torch.float32)
    
    def diffusion(self, x: torch.Tensor,
                  theta_diffusion: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion term for Lotka-Volterra SDE.
        
        Args:
            x: State vector [predator, prey]
            sigma: Noise standard deviations [sigma1, sigma2]
            device: Device for computation
        """
        assert theta_diffusion.shape[0] == 2, "Expected 2 parameters for diffusion (sigma1, sigma2)"

        # Handle sigma dimensions
        sigma = torch.exp(theta_diffusion)
        
        # State-dependent diffusion matrix
        return torch.tensor([
            [sigma[0] * x[0] * x[1], 0.0],
            [0.0, sigma[1] * x[1] * x[0]]
        ], device=self.device, dtype=torch.float32)
