from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.distributions import Distribution, TransformedDistribution
from torch.distributions.transforms import ExpTransform, Transform, identity_transform


class Dynamics(ABC):
    """Base dynamics with an explicit NLE-to-physical theta transform.

    The NLE and Gibbs sampler always use an unconstrained, real-valued theta.
    ``theta_transform`` maps that coordinate to the parameters consumed by
    ``split_theta()``, ``drift()``, and ``diffusion()``. Subclasses that do not
    override it retain the identity mapping.
    """

    theta_transform: Transform = identity_transform

    def __init__(self, dt, device):
        self.dt = dt
        self.device = device
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device."""
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def to_physical_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Map the real-valued NLE theta coordinate to physical parameters."""
        return self.theta_transform(theta)

    def to_nle_theta(self, physical_theta: torch.Tensor) -> torch.Tensor:
        """Map physical parameters to the real-valued NLE theta coordinate."""
        return self.theta_transform.inv(physical_theta)

    def pullback_theta_prior(
        self,
        physical_prior: Distribution,
    ) -> TransformedDistribution:
        """Express a physical-scale prior in the NLE theta coordinate.

        ``TransformedDistribution`` supplies the change-of-variables Jacobian,
        so the result can be passed directly as the sampler's ``theta_prior``.
        """
        return TransformedDistribution(
            physical_prior,
            [self.theta_transform.inv],
        )

    def simulate_one_step(
            self, x: torch.Tensor, 
            theta: torch.Tensor) -> torch.Tensor:
        """
        Simulate one step of the SDE.
        
        Args:
            x: Current state
            theta: Real-valued NLE parameter coordinate. It is transformed to
                the physical scale before the dynamics are evaluated.
        """
        physical_theta = self.to_physical_theta(theta)
        theta_drift, theta_diffusion = self.split_theta(physical_theta)
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
    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    theta_transform = ExpTransform()

    def __init__(self, dt: float = 0.01, 
                 device: str = 'cpu'):
        super().__init__(dt, device)
        self.x_dim = 2  # predator and prey
        self.theta_dim = 6  # 4 drift param and 2 diffusion param
    
    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (theta[:4], theta[4:6])

    def drift(self, x: torch.Tensor,
              theta_drift: torch.Tensor) -> torch.Tensor:
        """
        Compute drift term for Lotka-Volterra SDE.
        
        Args:
            x: State vector [predator, prey]
            params: Parameters [alpha, beta, gamma, delta]
            device: Device for computation
        """
        assert theta_drift.shape[0] == 4, "Expected 4 parameters for drift (alpha, beta, gamma, delta)"
        alpha, beta, gamma, delta = theta_drift
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
        sigma = theta_diffusion
        
        # State-dependent diffusion matrix
        return torch.tensor([
            [sigma[0] * x[0] * x[1], 0.0],
            [0.0, sigma[1] * x[1] * x[0]]
        ], device=self.device, dtype=torch.float32)


class SIRDynamics(Dynamics):
    r"""Chemical-Langevin susceptible-infected-recovered dynamics.

    For a fixed population size ``N``, the state is ``x = (S, I, R)`` and

        dS_t = -a_t dt - sqrt(a_t) dW_t^(infection),
        dI_t = (a_t - r_t) dt
               + sqrt(a_t) dW_t^(infection)
               - sqrt(r_t) dW_t^(recovery),
        dR_t = r_t dt + sqrt(r_t) dW_t^(recovery),

    where ``a_t = beta S_t I_t / N`` and ``r_t = gamma I_t``. The
    unconstrained parameter vector is ``theta = (log beta, log gamma)``.
    Euler-Maruyama proposals are projected onto the nonnegative simplex so
    that ``S + I + R = N`` remains true numerically.
    """

    theta_transform = ExpTransform()

    def __init__(
        self,
        dt: float = 0.01,
        population_size: float = 1_000.0,
        device: str = "cpu",
    ):
        super().__init__(dt, device)
        self.x_dim = 3
        self.theta_dim = 2
        self.population_size = population_size

    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both reaction rates enter the drift and diffusion terms."""
        return theta, theta

    def drift(
        self, x: torch.Tensor, theta_drift: torch.Tensor
    ) -> torch.Tensor:
        """Return the SIR drift vector."""
        beta, gamma = theta_drift
        susceptible, infected, _ = x
        infection = beta * susceptible * infected / self.population_size
        recovery = gamma * infected
        return torch.stack((-infection, infection - recovery, recovery))

    def diffusion(
        self, x: torch.Tensor, theta_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """Return the 3 x 3 reaction-noise loading matrix."""
        beta, gamma = theta_diffusion
        susceptible, infected, _ = x
        infection = torch.sqrt(
            (beta * susceptible * infected / self.population_size).clamp_min(0.0)
        )
        recovery = torch.sqrt((gamma * infected).clamp_min(0.0))
        zero = torch.zeros_like(infection)
        return torch.stack(
            (
                torch.stack((-infection, zero, zero)),
                torch.stack((infection, -recovery, zero)),
                torch.stack((zero, recovery, zero)),
            )
        )

    def simulate_one_step(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Take one step while preserving nonnegativity and population size."""
        proposal = super().simulate_one_step(x, theta).clamp_min(0.0)
        return proposal * (
            self.population_size / proposal.sum().clamp_min(1e-8)
        )


class FluoreChemicalLangevinDynamics(Dynamics):
    r"""One-dimensional fluorescence chemical Langevin dynamics.

    The SDE is

        dY_t = alpha (beta - Y_t) dt
               + sqrt(gamma (beta + Y_t)) dW_t.

    The unconstrained parameter vector is
    ``theta = (log alpha, log beta, log gamma)``. All three physical
    parameters are obtained with an exponential transformation.
    """

    theta_transform = ExpTransform()

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
        alpha, beta = theta_drift
        return alpha * (beta - x)

    def diffusion(
        self, x: torch.Tensor, theta_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """Return the 1 x 1 state-dependent diffusion matrix."""
        beta, gamma = theta_diffusion
        # Full truncation keeps Euler-Maruyama finite if a step crosses -beta.
        variance_rate = gamma * torch.clamp_min(beta + x[0], 0.0)
        return torch.sqrt(variance_rate).reshape(1, 1)


class GeneExpressionCLEDynamics(Dynamics):
    r"""Two-dimensional chemical Langevin gene-expression dynamics.

    ``M`` is the mRNA copy number and ``Y`` is the protein copy number. The
    state is ``x = (M, Y)`` and the SDE is implemented as

        dM_t = (alpha - beta M_t) dt
               + sqrt(alpha + beta M_t) dW_t^(M),

        dY_t = (gamma M_t - delta Y_t) dt
               + c sqrt(gamma M_t + delta Y_t) dW_t^(Y).

    The Brownian motions are independent. The unconstrained parameter vector is
    ``theta = (log alpha, log beta, log gamma, log delta, log c)``; all physical
    parameters are obtained by exponentiation.
    """

    theta_transform = ExpTransform()

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
        alpha, beta, gamma, delta = theta_drift
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
        alpha, beta, gamma, delta, c = theta_diffusion
        M, Y = x[0], x[1]
        mrna_rate = alpha + beta * M
        protein_rate = gamma * M + delta * Y
        zeros = torch.zeros((), dtype=x.dtype, device=x.device)
        return torch.stack(
            [
                torch.stack([torch.sqrt(mrna_rate), zeros]),
                torch.stack([zeros, c * torch.sqrt(protein_rate)]),
            ]
        )


class LatentMGeneExpressionCLEDynamics(Dynamics):
    r"""Gene-expression CLE with latent mRNA and fixed mRNA degradation rate.

    ``M`` is the latent mRNA copy number and ``Y`` is the protein copy number.
    The state is ``x = (M, Y)`` and beta is fixed to one:

        dM_t = (alpha - M_t) dt
               + sqrt(alpha + M_t) dW_t^(M),

        dY_t = (gamma M_t - delta Y_t) dt
               + c sqrt(gamma M_t + delta Y_t) dW_t^(Y).

    The unconstrained parameter vector is
    ``theta = (log alpha, log gamma, log delta, log c)``. All four inferred
    physical parameters are obtained by exponentiation.
    """

    theta_transform = ExpTransform()

    def __init__(self, dt: float = 0.01, device: str = "cpu"):
        super().__init__(dt, device)
        self.x_dim = 2
        self.theta_dim = 4

    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split parameters needed by the drift and diffusion functions."""
        if theta.shape[0] != self.theta_dim:
            raise ValueError(
                f"Expected {self.theta_dim} parameters, got {theta.shape[0]}."
            )
        # Drift uses alpha, gamma, and delta; diffusion additionally uses c.
        return theta[:3], theta

    def drift(
        self, x: torch.Tensor, theta_drift: torch.Tensor
    ) -> torch.Tensor:
        """Return the drift vector for (M, Y) with beta fixed to one."""
        alpha, gamma, delta = theta_drift
        M, Y = x[0], x[1]
        return torch.stack(
            [
                alpha - M,
                gamma * M - delta * Y,
            ]
        )

    def diffusion(
        self, x: torch.Tensor, theta_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """Return the diagonal 2 x 2 diffusion matrix."""
        alpha, gamma, delta, c = theta_diffusion
        M, Y = x[0], x[1]
        mrna_rate = alpha + M
        protein_rate = gamma * M + delta * Y
        zeros = torch.zeros((), dtype=x.dtype, device=x.device)
        return torch.stack(
            [
                torch.stack([torch.sqrt(mrna_rate), zeros]),
                torch.stack([zeros, c * torch.sqrt(protein_rate)]),
            ]
        )
