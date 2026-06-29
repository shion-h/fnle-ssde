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


class FluoreChemicalLangevinDynamics(Dynamics):
    r"""One-dimensional fluorescence chemical Langevin dynamics.

    The SDE is

        dY_t = alpha (beta - Y_t) dt
               + sqrt(gamma (beta + Y_t)) dW_t.

    The unconstrained parameter vector is
    ``theta = (log alpha, log beta, log gamma)``. All three physical
    parameters are obtained with an exponential transformation.
    """

    def __init__(self, dt: float = 0.01, device: str = "cpu"):
        super().__init__(dt, device)
        self.x_dim = 1
        self.theta_dim = 3

    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split parameters used by the drift and diffusion functions."""
        if theta.shape[0] != self.theta_dim:
            raise ValueError(
                f"Expected {self.theta_dim} parameters, got {theta.shape[0]}."
            )
        # beta appears in both the drift and diffusion terms.
        return theta[:2], theta[1:3]

    def drift(
        self, x: torch.Tensor, theta_drift: torch.Tensor
    ) -> torch.Tensor:
        """Return alpha * (beta - Y_t) as a one-dimensional vector."""
        alpha, beta = torch.exp(theta_drift)
        return alpha * (beta - x)

    def diffusion(
        self, x: torch.Tensor, theta_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """Return the 1 x 1 state-dependent diffusion matrix."""
        beta, gamma = torch.exp(theta_diffusion)
        # Full truncation keeps Euler-Maruyama finite if a step crosses -beta.
        variance_rate = gamma * torch.clamp_min(beta + x[0], 0.0)
        return torch.sqrt(variance_rate).reshape(1, 1)


class GeneExpressionCLEDynamics(Dynamics):
    r"""Two-dimensional chemical Langevin gene-expression dynamics.

    The state is ``x = (M, Y)`` and the SDE is implemented as

        dM_t = (alpha - beta M_t) dt
               + sqrt(alpha + beta M_t) dW_t^(M),

        dY_t = (gamma M_t - delta Y_t) dt
               + c sqrt(gamma M_t + delta Y_t) dW_t^(Y).

    The Brownian motions are independent. The unconstrained parameter vector is
    ``theta = (log alpha, log beta, log gamma, log delta, log c)``; all physical
    parameters are obtained by exponentiation.
    """

    def __init__(self, dt: float = 0.01, device: str = "cpu"):
        super().__init__(dt, device)
        self.x_dim = 2
        self.theta_dim = 5

    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split parameters needed by the drift and diffusion functions."""
        if theta.shape[0] != self.theta_dim:
            raise ValueError(
                f"Expected {self.theta_dim} parameters, got {theta.shape[0]}."
            )
        # Diffusion uses all four reaction rates and the additional scale c.
        return theta[:4], theta

    def drift(
        self, x: torch.Tensor, theta_drift: torch.Tensor
    ) -> torch.Tensor:
        """Return the drift vector for (M, Y)."""
        alpha, beta, gamma, delta = torch.exp(theta_drift)
        M, Y = x[0], x[1]
        return torch.stack(
            [
                alpha - beta * M,
                gamma * M - delta * Y,
            ]
        )

    def diffusion(
        self, x: torch.Tensor, theta_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """Return the diagonal 2 x 2 diffusion matrix."""
        alpha, beta, gamma, delta, c = torch.exp(theta_diffusion)
        M, Y = x[0], x[1]
        messenger_rate = alpha + beta * M
        expression_rate = gamma * M + delta * Y
        zeros = torch.zeros((), dtype=x.dtype, device=x.device)
        return torch.stack(
            [
                torch.stack([torch.sqrt(messenger_rate), zeros]),
                torch.stack([zeros, c * torch.sqrt(expression_rate)]),
            ]
        )
