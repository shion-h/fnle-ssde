from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.distributions import Distribution, TransformedDistribution
from torch.distributions.transforms import CatTransform, ExpTransform, Transform, identity_transform


class Dynamics(ABC):
    """Base dynamics with an explicit NLE-to-physical theta transform.

    The NLE and MCMC sampler always use an unconstrained, real-valued theta.
    ``theta_transform`` maps that coordinate to the parameters consumed by
    ``split_theta()``, ``drift()``, and ``diffusion()``. Subclasses that do not
    override it retain the identity mapping.
    """

    theta_transform: Transform = identity_transform
    state_lower_bound: float | torch.Tensor | None = None

    def __init__(
        self,
        dt: float,
        device: str,
        state_upper_bound: float | torch.Tensor | None = None,
    ):
        self.dt = dt
        self.device = device
        self.state_upper_bound = state_upper_bound
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device."""
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def to_physical_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Map the real-valued NLE theta coordinate to physical parameters."""
        return self.theta_transform(theta)

    def to_nle_theta(self, physical_theta: torch.Tensor) -> torch.Tensor:
        """Map physical parameters to the real-valued NLE theta coordinate."""
        return self.theta_transform.inv(physical_theta)

    def physical_bounds_to_nle_bounds(
        self, lower: torch.Tensor, upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enclose a physical parameter box in NLE coordinates."""
        a, b = self.to_nle_theta(lower), self.to_nle_theta(upper)
        return torch.minimum(a, b), torch.maximum(a, b)

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

    def constrain_state(self, x: torch.Tensor) -> torch.Tensor:
        """Project a proposed state onto this dynamics' valid state space."""
        if self.state_lower_bound is not None:
            lower = torch.as_tensor(
                self.state_lower_bound,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.maximum(x, lower)
        if self.state_upper_bound is not None:
            upper = torch.as_tensor(
                self.state_upper_bound,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.minimum(x, upper)
        return x

    def simulate_one_step(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Simulate one step of the SDE.
        
        Args:
            x: Current state
            theta: Real-valued NLE parameter coordinate. It is transformed to
                the physical scale before the dynamics are evaluated.
        """
        x = self.to_device(x)
        theta = self.to_device(theta)
        physical_theta = self.to_physical_theta(theta)
        theta_drift, theta_diffusion = self.split_theta(physical_theta)
        
        # Compute drift and diffusion terms
        drift = self.drift(x, theta_drift)
        diffusion = self.diffusion(x, theta_diffusion)
        
        # Brownian motion increment
        noise_dim = diffusion.shape[-1]
        batch_shape = torch.broadcast_shapes(x.shape[:-1], theta.shape[:-1])
        dW = torch.randn(
            (*batch_shape, noise_dim),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        ) * x.new_tensor(self.dt).sqrt()
        # Euler-Maruyama update
        noise = torch.matmul(diffusion, dW.unsqueeze(-1)).squeeze(-1)
        return self.constrain_state(x + drift * self.dt + noise)

    def simulate_n_steps(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        n_steps: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return the state after ``n_steps`` Euler-Maruyama transitions."""
        if not isinstance(n_steps, int) or isinstance(n_steps, bool):
            raise TypeError("n_steps must be an integer.")
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative.")
        current = x
        for _ in range(n_steps):
            if generator is None:
                current = self.simulate_one_step(current, theta)
            else:
                current = self.simulate_one_step(
                    current,
                    theta,
                    generator=generator,
                )
        return current

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


class OUDynamics(Dynamics):
    r"""Scalar OU with Euler-Maruyama simulation.

    Internal theta is (log kappa, mu, log sigma); physical theta is
    (kappa, mu, sigma). There is no state clamp. Like other Dynamics subclasses,
    simulate_one_step and simulate_n_steps use the base Euler-Maruyama methods.
    """

    theta_transform = CatTransform(
        [ExpTransform(), identity_transform, ExpTransform()], dim=-1
    )

    def __init__(self, dt: float = 0.01, device: str = "cpu"):
        super().__init__(dt, device)
        self.x_dim = 1
        self.theta_dim = 3

    def split_theta(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return theta[..., :2], theta[..., 2:]

    def drift(self, x: torch.Tensor, theta_drift: torch.Tensor) -> torch.Tensor:
        return theta_drift[..., :1] * (theta_drift[..., 1:2] - x)

    def diffusion(self, x: torch.Tensor, theta_diffusion: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(theta_diffusion.expand(
            torch.broadcast_shapes(x.shape, theta_diffusion.shape)
        ))


class ExactTransitionOUDynamics(OUDynamics):
    """OU dynamics with exact transitions over arbitrary elapsed times.

    ``sample_transition``, ``transition_log_prob``, and ``sample_bridge`` use
    analytic Gaussian laws. ``simulate_n_steps`` draws one exact transition of
    duration ``n_steps * dt`` for the NLE training interface. The inherited
    Inherited ``simulate_one_step`` and ``simulate`` remain Euler-Maruyama;
    neither is used by exact OU training, data generation, or inference.
    """

    def transition_coefficients(
        self, theta: torch.Tensor, delta: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a, b, v in Y_next = a Y_prev + b + Normal(0, v)."""
        delta = torch.as_tensor(delta, dtype=theta.dtype, device=theta.device)
        if torch.any(~torch.isfinite(delta)) or torch.any(delta <= 0):
            raise ValueError("OU transition delta must be finite and positive.")
        kappa, mu, sigma = self.to_physical_theta(theta).unbind(-1)
        kd = kappa * delta
        a = torch.exp(-kd)
        b = -torch.expm1(-kd) * mu
        v = sigma.square() * (-torch.expm1(-2 * kd)) / (2 * kappa)
        return a.unsqueeze(-1), b.unsqueeze(-1), v.unsqueeze(-1)

    def transition_moments(
        self, y_prev: torch.Tensor, theta: torch.Tensor, delta: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, b, v = self.transition_coefficients(theta, delta)
        return a * y_prev + b, v

    def transition_log_prob(
        self, y_prev: torch.Tensor, y_next: torch.Tensor,
        theta: torch.Tensor, delta: float | torch.Tensor,
    ) -> torch.Tensor:
        mean, variance = self.transition_moments(y_prev, theta, delta)
        return torch.distributions.Normal(mean, variance.sqrt()).log_prob(y_next).sum(-1)

    def sample_transition(
        self, theta: torch.Tensor, y_prev: torch.Tensor, delta: float | torch.Tensor,
        *, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mean, variance = self.transition_moments(y_prev, theta, delta)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return mean + variance.sqrt() * noise

    def sample_bridge(
        self, left_y: torch.Tensor, right_y: torch.Tensor,
        times: torch.Tensor, interval_theta: torch.Tensor,
        *, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample interior states conditional on both endpoints and interval regimes.

        times includes both endpoints; interval_theta has len(times)-1 rows.
        Leading dimensions of the endpoints may be used to draw multiple bridges.
        """
        if times.ndim != 1 or times.numel() < 2:
            raise ValueError("Bridge times must include two endpoints.")
        if interval_theta.shape != (times.numel() - 1, self.theta_dim):
            raise ValueError("One theta row is required per bridge interval.")
        a, b, v = self.transition_coefficients(interval_theta, times.diff())
        # Conditional right-endpoint distribution given each intermediate state.
        A, B, V = [a[-1]], [b[-1]], [v[-1]]
        for i in range(len(a) - 2, -1, -1):
            next_a = A[-1]
            A.append(next_a * a[i])
            B.append(next_a * b[i] + B[-1])
            V.append(next_a.square() * v[i] + V[-1])
        A, B, V = A[::-1], B[::-1], V[::-1]
        current = left_y
        samples = []
        for i in range(len(a) - 1):
            mean = a[i] * current + b[i]
            denominator = A[i + 1].square() * v[i] + V[i + 1]
            gain = v[i] * A[i + 1] / denominator
            mean = mean + gain * (right_y - A[i + 1] * mean - B[i + 1])
            variance = v[i] * V[i + 1] / denominator
            current = mean + variance.sqrt() * torch.randn(
                mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
            )
            samples.append(current)
        if not samples:
            return left_y.new_empty((*left_y.shape[:-1], 0, self.x_dim))
        return torch.stack(samples, dim=-2)

    def simulate_n_steps(
        self, x: torch.Tensor, theta: torch.Tensor, n_steps: int,
        *, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw the exact final state after ``n_steps * dt`` elapsed time."""
        if not isinstance(n_steps, int) or isinstance(n_steps, bool):
            raise TypeError("n_steps must be an integer.")
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative.")
        if n_steps == 0:
            return x
        return self.sample_transition(theta, x, n_steps * self.dt, generator=generator)


class LotkaVolterraDynamics(Dynamics):
    r"""Lotka-Volterra dynamics with multiplicative environmental noise.

    For predator ``D`` and prey ``P``, the diffusion matrix is

        G(D, P) = diag(sigma_D D, sigma_P P).

    The two independent Brownian motions therefore represent environmental
    fluctuations acting proportionally on each population.
    """

    theta_transform = ExpTransform()
    state_lower_bound = 0.0

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        state_upper_bound: float | torch.Tensor | None = None,
    ):
        super().__init__(dt, device, state_upper_bound)
        self.x_dim = 2  # predator and prey
        self.theta_dim = 6  # 4 drift param and 2 diffusion param
    
    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return theta[..., :4], theta[..., 4:6]

    def drift(self, x: torch.Tensor,
              theta_drift: torch.Tensor) -> torch.Tensor:
        """
        Compute drift term for Lotka-Volterra SDE.
        
        Args:
            x: State vector [predator, prey]
            params: Parameters [alpha, beta, gamma, delta]
            device: Device for computation
        """
        if theta_drift.shape[-1] != 4:
            raise ValueError("Expected 4 parameters for drift.")
        alpha, beta, gamma, delta = theta_drift.unbind(dim=-1)
        predator, prey = x.unbind(dim=-1)
        
        f_predator = gamma * prey * predator - delta * predator
        f_prey = alpha * prey - beta * prey * predator
        
        return torch.stack((f_predator, f_prey), dim=-1)
    
    def diffusion(self, x: torch.Tensor,
                  theta_diffusion: torch.Tensor) -> torch.Tensor:
        """
        Compute multiplicative environmental noise for Lotka-Volterra SDE.
        
        Args:
            x: State vector [predator, prey]
            sigma: Noise standard deviations [sigma1, sigma2]
            device: Device for computation
        """
        if theta_diffusion.shape[-1] != 2:
            raise ValueError("Expected 2 parameters for diffusion.")
        return torch.diag_embed(theta_diffusion * x)


class SIRDynamics(Dynamics):
    r"""Chemical-Langevin SIR dynamics in reduced ``(S, R)`` coordinates.

    The infected population is reconstructed as ``I = N - S - R``:

        dS_t = -a_t dt - sqrt(a_t) dW_t^(infection),
        dR_t = r_t dt + sqrt(r_t) dW_t^(recovery),

    This representation removes the deterministic conservation constraint from
    the NLE target and separates the two Brownian motions. Here
    ``a_t = beta S_t I_t / N``, ``r_t = gamma I_t``, and
    ``theta = (log beta, log gamma)``.
    """

    theta_transform = ExpTransform()
    state_lower_bound = 0.0

    def __init__(
        self,
        dt: float = 0.01,
        population_size: float = 1_000.0,
        device: str = "cpu",
        state_upper_bound: float | torch.Tensor | None = None,
    ):
        super().__init__(dt, device, state_upper_bound)
        self.x_dim = 2
        self.theta_dim = 2
        self.population_size = population_size

    def _infected(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the nonnegative infected population."""
        return (self.population_size - x[..., 0] - x[..., 1]).clamp_min(0.0)

    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both reaction rates enter the drift and diffusion terms."""
        return theta, theta

    def drift(
        self, x: torch.Tensor, theta_drift: torch.Tensor
    ) -> torch.Tensor:
        """Return the SIR drift in the configured state representation."""
        beta, gamma = theta_drift.unbind(dim=-1)
        susceptible = x[..., 0]
        infected = self._infected(x)
        infection = beta * susceptible * infected / self.population_size
        recovery = gamma * infected
        return torch.stack((-infection, recovery), dim=-1)

    def diffusion(
        self, x: torch.Tensor, theta_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """Return the reaction-noise loading matrix."""
        beta, gamma = theta_diffusion.unbind(dim=-1)
        susceptible = x[..., 0]
        infected = self._infected(x)
        infection = torch.sqrt(
            (beta * susceptible * infected / self.population_size).clamp_min(0.0)
        )
        recovery = torch.sqrt((gamma * infected).clamp_min(0.0))
        diagonal = torch.stack((-infection, recovery), dim=-1)
        return torch.diag_embed(diagonal)

    def constrain_state(self, x: torch.Tensor) -> torch.Tensor:
        """Enforce nonnegativity and ``S + R <= population_size``."""
        x = super().constrain_state(x)
        total = x.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (self.population_size / total).clamp_max(1.0)
        return x * scale


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
    state_lower_bound = 0.0

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        state_upper_bound: float | torch.Tensor | None = None,
    ):
        super().__init__(dt, device, state_upper_bound)
        self.x_dim = 1
        self.theta_dim = 3

    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split parameters used by the drift and diffusion functions."""
        if theta.shape[-1] != self.theta_dim:
            raise ValueError(
                f"Expected {self.theta_dim} parameters, got {theta.shape[-1]}."
            )
        # beta appears in both the drift and diffusion terms.
        return theta[..., :2], theta[..., 1:3]

    def drift(
        self, x: torch.Tensor, theta_drift: torch.Tensor
    ) -> torch.Tensor:
        """Return alpha * (beta - Y_t) as a one-dimensional vector."""
        alpha, beta = theta_drift.unbind(dim=-1)
        return (alpha * (beta - x[..., 0])).unsqueeze(-1)

    def diffusion(
        self, x: torch.Tensor, theta_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """Return the 1 x 1 state-dependent diffusion matrix."""
        beta, gamma = theta_diffusion.unbind(dim=-1)
        # Full truncation keeps Euler-Maruyama finite if a step crosses -beta.
        variance_rate = gamma * torch.clamp_min(beta + x[..., 0], 0.0)
        return torch.sqrt(variance_rate)[..., None, None]


class GeneExpressionCLEDynamics(Dynamics):
    r"""Two-dimensional chemical Langevin gene-expression dynamics.

    ``M`` is the mRNA copy number and ``Y`` is protein fluorescence intensity.
    With fluorescence per protein ``c**2``, the protein copy number is
    ``Y / c**2``. The state is ``x = (M, Y)`` and the SDE is implemented as

        dM_t = (alpha - beta M_t) dt
               + sqrt(alpha + beta M_t) dW_t^(M),

        dY_t = (gamma M_t - delta Y_t) dt
               + c sqrt(gamma M_t + delta Y_t) dW_t^(Y).

    The Brownian motions are independent. The unconstrained parameter vector is
    ``theta = (log alpha, log beta, log gamma, log delta, log c)``; all physical
    parameters are obtained by exponentiation.
    """

    theta_transform = ExpTransform()
    state_lower_bound = 0.0

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        state_upper_bound: float | torch.Tensor | None = None,
    ):
        super().__init__(dt, device, state_upper_bound)
        self.x_dim = 2
        self.theta_dim = 5

    def split_theta(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split parameters needed by the drift and diffusion functions."""
        if theta.shape[-1] != self.theta_dim:
            raise ValueError(
                f"Expected {self.theta_dim} parameters, got {theta.shape[-1]}."
            )
        # Diffusion uses all four reaction rates and the additional scale c.
        return theta[..., :4], theta

    def drift(
        self, x: torch.Tensor, theta_drift: torch.Tensor
    ) -> torch.Tensor:
        """Return the drift vector for (M, Y)."""
        alpha, beta, gamma, delta = theta_drift.unbind(dim=-1)
        M, Y = x.unbind(dim=-1)
        return torch.stack(
            [
                alpha - beta * M,
                gamma * M - delta * Y,
            ],
            dim=-1,
        )

    def diffusion(
        self, x: torch.Tensor, theta_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """Return the diagonal 2 x 2 diffusion matrix."""
        alpha, beta, gamma, delta, c = theta_diffusion.unbind(dim=-1)
        M, Y = x.unbind(dim=-1)
        diagonal = torch.stack(
            (
                torch.sqrt((alpha + beta * M).clamp_min(0.0)),
                c * torch.sqrt((gamma * M + delta * Y).clamp_min(0.0)),
            ),
            dim=-1,
        )
        return torch.diag_embed(diagonal)


class ReparametrizedGeneExpressionCLEDynamics(GeneExpressionCLEDynamics):
    r"""CLE whose first physical parameter is rho = alpha/beta.

    Physical theta is (rho, beta, gamma, delta, c) and the real-valued NLE
    coordinate is its component-wise logarithm. The reaction rate alpha is
    recovered as rho * beta before evaluating the original CLE dynamics.
    A physical-scale prior on rho is therefore specified directly through
    the inherited ``pullback_theta_prior`` method.
    """

    def split_theta(
        self, theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if theta.shape[-1] != self.theta_dim:
            raise ValueError(
                f"Expected {self.theta_dim} parameters, got {theta.shape[-1]}."
            )
        original_theta = torch.cat(
            ((theta[..., 0] * theta[..., 1]).unsqueeze(-1), theta[..., 1:]),
            dim=-1,
        )
        return super().split_theta(original_theta)
