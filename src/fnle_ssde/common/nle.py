import time
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Union
import torch
import pyro.distributions as dist
from torch.distributions import Independent, Normal
from sbi.inference import SNLE
from .dynamics import Dynamics

NStepsType = Union[int, Tuple[int], List[int]]
PathLike = Union[str, Path]


class NLEEstimator:
    """Neural Likelihood Estimator for FNLE of SDE."""
    def __init__(self,
                 dynamics: Dynamics,
                 sampling_dist = None,
                 device: str = 'cpu'):
        self.dynamics = dynamics
        self.device = device
        self.estimator = None
        self.conditions_on_n_steps = False
        
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
            n_steps: NStepsType
            ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate training data for one parameter sample.
        
        Args:
            theta: Parameter vector
            x_ref: Reference trajectory
            n_steps: Fixed number of steps, or a length-1 tuple/list for random 1..max_n_steps
        """
        xt_chunk = []
        ctx_chunk = []
        
        for t in range(1, len(x_ref)):
            # Use first state parameters (assuming single regime for training)
            
            # Initial condition with small noise
            x_init = x_ref[t-1] + torch.randn_like(x_ref[t-1]) * ref_noize
            x_init = self.dynamics.to_device(x_init)
            
            # Simulate forward
            x_sim = x_init
            sampled_n_steps = self._sample_n_steps(n_steps)
            for _ in range(sampled_n_steps):
                x_sim = self.dynamics.simulate_one_step(x_sim, theta)
                x_sim = torch.clamp(x_sim, min=0.0, max=1e4)
            
            # Create context
            if self._uses_n_steps_conditioning(n_steps):
                n_steps_tensor = torch.tensor(
                    [sampled_n_steps], device=x_init.device, dtype=x_init.dtype
                )
                context = torch.cat([theta, x_init, n_steps_tensor], dim=-1)
            else:
                context = torch.cat([theta, x_init], dim=-1)
            
            xt_chunk.append(x_sim)
            ctx_chunk.append(context)
        
        return xt_chunk, ctx_chunk
    
    def generate_training_data(
            self, 
            n_params: int, 
            x_ref: torch.Tensor, 
            ref_noize: float,
            n_steps: NStepsType,
            cache_path: PathLike | None = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate complete training dataset.
        
        Args:
            n_params: Number of parameter samples
            x_ref: Reference trajectories for training
            ref_noize: Noise level for initial conditions
            n_steps: Fixed number of steps, or a length-1 tuple/list for random 1..max_n_steps
            cache_path: Optional path to save or reuse generated training data
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print(f"Loading cached training data from {cache_path}...")
                xt_data, ctx_data = self._load_training_data_cache(cache_path)
                return xt_data.to(self.device), ctx_data.to(self.device)

        all_xt = []
        all_ctx = []
        
        print(f"Generating training data from {n_params} parameter samples...")
        for _ in tqdm(range(n_params)):
            theta = self.sampling_dist.sample().cpu()
            xt_chunk, ctx_chunk = self.generate_training_data_one_theta(
                theta, x_ref, ref_noize, n_steps)
            all_xt.extend(xt_chunk)
            all_ctx.extend(ctx_chunk)

        xt_data = torch.stack(all_xt).to(self.device)
        ctx_data = torch.stack(all_ctx).to(self.device)

        if cache_path is not None:
            self._save_training_data_cache(
                cache_path=cache_path,
                xt_data=xt_data,
                ctx_data=ctx_data,
                n_params=n_params,
                ref_noize=ref_noize,
                n_steps=n_steps,
                x_ref_shape=tuple(x_ref.shape),
            )

        return xt_data, ctx_data
    
    def train(self, x_ref: torch.Tensor, n_params: int = 500, 
              ref_noize: float = 0.02,
              n_steps: NStepsType = 50,
              cache_path: PathLike | None = None,
              batch_size: int = 256, lr: float = 5e-4, 
              epochs: int = 50) -> 'NLEEstimator':
        """
        Train the NLE estimator.
        
        Args:
            x_ref: Reference trajectories
            n_params: Number of parameter samples for training
            ref_noize: Noise level for initial conditions
            n_steps: Fixed number of steps, or a length-1 tuple/list for random 1..max_n_steps
            cache_path: Optional path to save or reuse generated training data
            batch_size: Training batch size
            lr: Learning rate
            epochs: Number of epochs
        """
        self.conditions_on_n_steps = self._uses_n_steps_conditioning(n_steps)

        # Generate training data
        start_time = time.time()
        xt_data, ctx_data = self.generate_training_data(
            n_params, x_ref, ref_noize, n_steps, cache_path=cache_path)
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

    def _sample_n_steps(self, n_steps: NStepsType) -> int:
        if isinstance(n_steps, int):
            if n_steps < 1:
                raise ValueError("n_steps must be a positive integer.")
            return n_steps

        if isinstance(n_steps, (tuple, list)) and len(n_steps) == 1:
            max_steps = n_steps[0]
            if not isinstance(max_steps, int):
                raise TypeError("Random n_steps max must be an integer.")
            if max_steps < 1:
                raise ValueError("Random n_steps max must be positive.")
            return torch.randint(1, max_steps + 1, (1,)).item()

        raise TypeError(
            "n_steps must be an int or a length-1 tuple/list containing max_n_steps."
        )

    def _uses_n_steps_conditioning(self, n_steps: NStepsType) -> bool:
        return isinstance(n_steps, (tuple, list))

    def _save_training_data_cache(
            self,
            cache_path: Path,
            xt_data: torch.Tensor,
            ctx_data: torch.Tensor,
            n_params: int,
            ref_noize: float,
            n_steps: NStepsType,
            x_ref_shape: Tuple[int, ...]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "xt_data": xt_data.detach().cpu(),
                "ctx_data": ctx_data.detach().cpu(),
                "metadata": {
                    "n_params": n_params,
                    "ref_noize": ref_noize,
                    "n_steps": n_steps,
                    "x_ref_shape": x_ref_shape,
                    "conditions_on_n_steps": self.conditions_on_n_steps,
                },
                "member_variables": {
                    "dynamics": self.dynamics,
                    "sampling_dist": self.sampling_dist,
                    "device": self.device,
                    "conditions_on_n_steps": self.conditions_on_n_steps,
                },
            },
            cache_path,
        )
        print(f"Saved training data cache to {cache_path}")

    def _load_training_data_cache(
            self,
            cache_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = torch.load(cache_path, map_location="cpu")
        metadata = cache.get("metadata", {})
        if "conditions_on_n_steps" in metadata:
            self.conditions_on_n_steps = metadata["conditions_on_n_steps"]
        self.cached_member_variables = cache.get("member_variables", {})
        return cache["xt_data"], cache["ctx_data"]
