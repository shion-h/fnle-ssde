import time
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Tuple, List, Union
import torch
import pyro.distributions as dist
from torch.distributions import Distribution, Independent, Normal
from scipy.interpolate import CubicSpline
from sbi.inference import SNLE
from sbi.neural_nets.factory import likelihood_nn
from .dynamics import Dynamics

PathLike = Union[str, Path]

# nflows 0.14 still calls torch.triangular_solve inside LU transforms.
# This is a dependency-level deprecation warning; suppress only that warning.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.triangular_solve is deprecated.*",
    category=UserWarning,
    module=r"nflows\.transforms\.lu",
)


class NLEEstimator:
    """Neural Likelihood Estimator for FNLE of SDE."""
    def __init__(self,
                 dynamics: Dynamics,
                 sampling_dist = None,
                 device: str = 'cpu',
                 model_cache_path: PathLike | None = None,
                 target_type: Literal["x_next", "scaled_dx"] = "x_next",
                 density_model: str = "nsf",
                 hidden_features: int = 50,
                 num_transforms: int = 5,
                 num_bins: int = 10):
        self.dynamics = dynamics
        self.device = device
        self.estimator = None
        self.conditions_on_n_steps = False
        self.model_cache_path = Path(model_cache_path) if model_cache_path is not None else None
        self.target_type = target_type
        if self.target_type not in {"x_next", "scaled_dx"}:
            raise ValueError("target_type must be either 'x_next' or 'scaled_dx'.")
        self.density_model = density_model
        self.hidden_features = int(hidden_features)
        self.num_transforms = int(num_transforms)
        self.num_bins = int(num_bins)
        
        if sampling_dist is None:
            self.sampling_dist = dist.MultivariateNormal(
                torch.zeros(dynamics.theta_dim, device=device),
                torch.eye(dynamics.theta_dim, device=device)
            )
        else:
            self.sampling_dist = sampling_dist

        if self.model_cache_path is not None and self.model_cache_path.exists():
            self.load_model(self.model_cache_path)
    
    def generate_training_data_one_theta(
            self, 
            theta: torch.Tensor, 
            x_ref: torch.Tensor, 
            ref_noize: float,
            max_n_steps: int | None = None,
            n_transitions: int | None = None,
            observed_dims: torch.Tensor | None = None,
            unobserved_dims: torch.Tensor | None = None,
            unobserved_init_dist: Distribution | None = None,
            state_clamp_bounds: tuple[float | None, float | None] | None = (0.0, 1e4),
            noisy_init_strategy: Literal["clamp", "resample"] = "clamp",
            max_noisy_init_attempts: int = 1000,
            ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate training data for one parameter sample.
        
        Args:
            theta: Parameter vector
            x_ref: Reference trajectory
            max_n_steps: If None, simulate one step and do not condition on n_steps.
                If an int, sample n_steps uniformly from 1..max_n_steps and append
                the sampled value to the context.
            n_transitions: Number of transitions to generate for this theta. If None,
                use every adjacent reference state, preserving the original behavior.
            observed_dims: Latent-state indices represented by x_ref columns. If
                None, x_ref must contain the complete latent state.
            unobserved_dims: Complement of observed_dims, prepared once by
                generate_training_data.
            unobserved_init_dist: Distribution whose sample fills latent dimensions
                absent from x_ref.
            state_clamp_bounds: Optional bounds applied when constructing the
                noisy initial state. Simulated states are constrained by the
                supplied Dynamics instance.
            noisy_init_strategy: How to handle noisy initial states outside
                state_clamp_bounds. "clamp" preserves the historical behavior;
                "resample" redraws the complete initial state until it is valid.
            max_noisy_init_attempts: Maximum draws for "resample".
        """
        xt_chunk = []
        ctx_chunk = []

        if n_transitions is None:
            ref_indices = range(1, len(x_ref))
        else:
            ref_indices = torch.randint(1, len(x_ref), size=(n_transitions,)).tolist()

        for t in ref_indices:
            x_ref_value = x_ref[t - 1]
            attempts = (
                max_noisy_init_attempts if noisy_init_strategy == "resample" else 1
            )
            for _ in range(attempts):
                x_init = self._draw_noisy_initial_state(
                    x_ref_value=x_ref_value,
                    ref_noize=ref_noize,
                    observed_dims=observed_dims,
                    unobserved_dims=unobserved_dims,
                    unobserved_init_dist=unobserved_init_dist,
                )
                if (
                    noisy_init_strategy == "clamp"
                    or self._is_within_bounds(x_init, state_clamp_bounds)
                ):
                    break
            else:
                raise RuntimeError(
                    "Could not draw a noisy initial state inside "
                    f"state_clamp_bounds={state_clamp_bounds} after "
                    f"{max_noisy_init_attempts} attempts."
                )
            if noisy_init_strategy == "clamp":
                x_init = self._clamp_to_bounds(
                    x_init,
                    state_clamp_bounds,
                    argument_name="state_clamp_bounds",
                )
            x_init = self.dynamics.to_device(x_init)
            
            # Simulate forward
            sampled_n_steps = self._sample_n_steps(max_n_steps)
            x_sim = self.dynamics.simulate_n_steps(
                x_init,
                theta,
                sampled_n_steps,
            )
            
            # Create context
            if self.conditions_on_n_steps:
                n_steps_tensor = torch.tensor(
                    [sampled_n_steps], device=x_init.device, dtype=x_init.dtype
                )
                context = torch.cat([theta, x_init, n_steps_tensor], dim=-1)
            else:
                context = torch.cat([theta, x_init], dim=-1)
            
            xt_chunk.append(
                self._to_training_target(
                    x_prev=x_init,
                    x_next=x_sim,
                    n_steps=sampled_n_steps,
                )
            )
            ctx_chunk.append(context)
        
        return xt_chunk, ctx_chunk
    
    def generate_training_data(
            self, 
            n_params: int, 
            x_ref: torch.Tensor, 
            ref_noize: float,
            max_n_steps: int | None = None,
            samples_per_theta: int | None = None,
            observed_dims: torch.Tensor | None = None,
            unobserved_init_dist: Distribution | None = None,
            state_clamp_bounds: tuple[float | None, float | None] | None = (0.0, 1e4),
            noisy_init_strategy: Literal["clamp", "resample"] = "clamp",
            max_noisy_init_attempts: int = 1000,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate complete training dataset.
        
        Args:
            n_params: Number of parameter samples
            x_ref: Reference trajectories for training
            ref_noize: Noise level for initial conditions
            max_n_steps: If None, simulate one step and do not condition on n_steps.
                If an int, sample n_steps uniformly from 1..max_n_steps and append
                the sampled value to the context.
            samples_per_theta: Number of transitions generated per parameter sample.
                If None, use all adjacent reference states for each theta. Set to 1
                to train on one transition per sampled theta.
            observed_dims: Latent-state indices represented by x_ref columns.
            unobserved_init_dist: Distribution used to initialize omitted latent
                dimensions independently for each generated transition.
            state_clamp_bounds: Optional bounds applied to noisy initial states.
                Subsequent states are constrained by the Dynamics instance.
        """
        all_xt = []
        all_ctx = []

        observed_dims, unobserved_dims = self._prepare_initial_state_dimensions(
            x_ref=x_ref,
            observed_dims=observed_dims,
            unobserved_init_dist=unobserved_init_dist,
        )
        if samples_per_theta is not None and samples_per_theta < 1:
            raise ValueError("samples_per_theta must be positive.")
        if noisy_init_strategy not in {"clamp", "resample"}:
            raise ValueError("noisy_init_strategy must be 'clamp' or 'resample'.")
        if max_noisy_init_attempts < 1:
            raise ValueError("max_noisy_init_attempts must be positive.")
        
        print(f"Generating training data from {n_params} parameter samples...")
        for _ in tqdm(range(n_params)):
            theta = self.sampling_dist.sample().cpu()
            xt_chunk, ctx_chunk = self.generate_training_data_one_theta(
                theta,
                x_ref,
                ref_noize,
                max_n_steps,
                n_transitions=samples_per_theta,
                observed_dims=observed_dims,
                unobserved_dims=unobserved_dims,
                unobserved_init_dist=unobserved_init_dist,
                state_clamp_bounds=state_clamp_bounds,
                noisy_init_strategy=noisy_init_strategy,
                max_noisy_init_attempts=max_noisy_init_attempts,
            )
            all_xt.extend(xt_chunk)
            all_ctx.extend(ctx_chunk)

        xt_data = torch.stack(all_xt).to(self.device)
        ctx_data = torch.stack(all_ctx).to(self.device)

        return xt_data, ctx_data
    
    @classmethod
    def build_spline_reference_path(
            cls,
            obs_times: torch.Tensor,
            x_obs: torch.Tensor,
            reference_times: torch.Tensor,
            spline_clamp_bounds: tuple[float | None, float | None] | None = (0.0, None),
            ) -> torch.Tensor:
        """Interpolate irregular observations onto a reference-time grid.

        Natural cubic splines are the standard reference-path construction for
        NLE training. Set ``spline_clamp_bounds=None`` when the state space
        permits negative values.
        """
        obs_times = torch.as_tensor(obs_times)
        x_obs = torch.as_tensor(x_obs)
        reference_times = torch.as_tensor(reference_times)
        if obs_times.ndim != 1 or reference_times.ndim != 1:
            raise ValueError("obs_times and reference_times must be one-dimensional.")
        if x_obs.ndim != 2 or x_obs.shape[0] != obs_times.numel():
            raise ValueError("x_obs must have shape (len(obs_times), observed_dim).")
        if obs_times.numel() < 2 or not torch.all(obs_times[1:] > obs_times[:-1]):
            raise ValueError("obs_times must contain at least two increasing values.")
        if reference_times.numel() < 2 or not torch.all(
            reference_times[1:] > reference_times[:-1]
        ):
            raise ValueError("reference_times must contain increasing values.")
        if reference_times[0] < obs_times[0] or reference_times[-1] > obs_times[-1]:
            raise ValueError("reference_times must lie inside the observation interval.")

        spline = CubicSpline(
            obs_times.detach().cpu().numpy(),
            x_obs.detach().cpu().numpy(),
            axis=0,
            bc_type="natural",
        )
        x_ref = torch.as_tensor(
            spline(reference_times.detach().cpu().numpy()),
            dtype=x_obs.dtype,
            device=x_obs.device,
        )
        return cls._clamp_to_bounds(
            x_ref,
            spline_clamp_bounds,
            argument_name="spline_clamp_bounds",
        )

    def train(self, x_ref: torch.Tensor | None = None, n_params: int = 500,
              ref_noize: float = 0.02,
              max_n_steps: int | None = None,
              batch_size: int = 256, lr: float = 5e-4, 
              epochs: int = 50,
              stop_after_epochs: int = 20,
              samples_per_theta: int | None = None,
              observed_dims: torch.Tensor | None = None,
              unobserved_init_dist: Distribution | None = None,
              state_clamp_bounds: tuple[float | None, float | None] | None = (0.0, 1e4),
              noisy_init_strategy: Literal["clamp", "resample"] = "clamp",
              max_noisy_init_attempts: int = 1000,
              obs_times: torch.Tensor | None = None,
              x_obs: torch.Tensor | None = None,
              reference_times: torch.Tensor | None = None,
              spline_clamp_bounds: tuple[float | None, float | None] | None = (0.0, None),
              training_data_cache_path: PathLike | None = None,
              ) -> 'NLEEstimator':
        """
        Train the NLE estimator.
        
        Args:
            x_ref: Reference trajectory. If omitted, it is constructed from
                obs_times, x_obs, and reference_times by natural cubic spline.
            n_params: Number of parameter samples for training
            ref_noize: Noise level for initial conditions
            max_n_steps: If None, simulate one step and do not condition on n_steps.
                If an int, sample n_steps uniformly from 1..max_n_steps and append
                the sampled value to the context.
            batch_size: Training batch size
            lr: Learning rate
            epochs: Number of epochs
            stop_after_epochs: Stop after this many epochs without validation
                improvement.
            samples_per_theta: Number of transitions generated per sampled theta.
                If None, preserve the original behavior and use len(x_ref)-1
                transitions per theta. Set to 1 for one transition per theta.
            observed_dims: Latent-state indices represented by x_ref columns.
            unobserved_init_dist: Distribution used to initialize latent dimensions
                that are not represented in x_ref.
            state_clamp_bounds: Optional state-space bounds applied to noisy
                initial states. Subsequent states are constrained by the
                Dynamics instance.
            noisy_init_strategy: Clamp or redraw noisy initial states that fall
                outside state_clamp_bounds.
            max_noisy_init_attempts: Maximum redraws under "resample".
            obs_times: Irregular observation times used for spline construction.
            x_obs: Observations with shape (N, observed_dim).
            reference_times: Grid on which the spline reference path is evaluated.
            spline_clamp_bounds: Optional bounds applied to the spline path.
            training_data_cache_path: Optional separate cache for generated
                training tensors. This allows interrupted training to restart
                without rerunning the simulator.
        """
        spline_inputs = (obs_times, x_obs, reference_times)
        if x_ref is None:
            if any(value is None for value in spline_inputs):
                raise ValueError(
                    "Provide x_ref, or provide obs_times, x_obs, and reference_times."
                )
            x_ref = self.build_spline_reference_path(
                obs_times=obs_times,
                x_obs=x_obs,
                reference_times=reference_times,
                spline_clamp_bounds=spline_clamp_bounds,
            )
        elif any(value is not None for value in spline_inputs):
            raise ValueError(
                "Do not provide obs_times, x_obs, or reference_times together with x_ref."
            )

        self.conditions_on_n_steps = max_n_steps is not None
        if max_n_steps is not None and max_n_steps < 1:
            raise ValueError("max_n_steps must be positive when specified.")

        start_time = time.time()
        xt_data = None
        ctx_data = None
        x_ref_shape = tuple(x_ref.shape)
        training_data_cache_path = (
            Path(training_data_cache_path)
            if training_data_cache_path is not None
            else None
        )

        if training_data_cache_path is not None and training_data_cache_path.exists():
            print(f"Loading cached training data from {training_data_cache_path}...")
            training_data_cache = torch.load(
                training_data_cache_path,
                map_location=self.device,
                weights_only=False,
            )
            xt_data = training_data_cache["xt_data"]
            ctx_data = training_data_cache["ctx_data"]

        if self.model_cache_path is not None and self.model_cache_path.exists():
            cache = self._load_model_cache(self.model_cache_path)
            xt_data = cache.get("xt_data")
            ctx_data = cache.get("ctx_data")
            metadata = cache.get("metadata", {})
            if xt_data is not None and ctx_data is not None:
                print(f"Loading cached training data from {self.model_cache_path}...")
                if (
                    state_clamp_bounds is not None
                    and not metadata.get("state_clamp_applies_to_x_init", False)
                ):
                    raise ValueError(
                        "Cached NLE training data predates applying "
                        "state_clamp_bounds to noisy x_init. Use a different "
                        "cache path to regenerate the training data."
                    )
                cached_strategy = metadata.get("noisy_init_strategy", "clamp")
                if cached_strategy != noisy_init_strategy:
                    raise ValueError(
                        "Cached NLE training data used noisy_init_strategy="
                        f"{cached_strategy!r}, requested {noisy_init_strategy!r}. "
                        "Use a different cache path to regenerate the data."
                    )
                if "conditions_on_n_steps" in metadata:
                    self.conditions_on_n_steps = metadata["conditions_on_n_steps"]
                xt_data = xt_data.to(self.device)
                ctx_data = ctx_data.to(self.device)

        if xt_data is None or ctx_data is None:
            xt_data, ctx_data = self.generate_training_data(
                n_params,
                x_ref,
                ref_noize,
                max_n_steps,
                samples_per_theta=samples_per_theta,
                observed_dims=observed_dims,
                unobserved_init_dist=unobserved_init_dist,
                state_clamp_bounds=state_clamp_bounds,
                noisy_init_strategy=noisy_init_strategy,
                max_noisy_init_attempts=max_noisy_init_attempts,
            )
            if training_data_cache_path is not None:
                training_data_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "xt_data": xt_data.detach().cpu(),
                        "ctx_data": ctx_data.detach().cpu(),
                    },
                    training_data_cache_path,
                )
                print(f"Saved training data to {training_data_cache_path}")
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
            density_estimator=likelihood_nn(
                model=self.density_model,
                hidden_features=self.hidden_features,
                num_transforms=self.num_transforms,
                num_bins=self.num_bins,
            ),
            device=self.device,
            show_progress_bars=True
        )
        
        self.estimator = inf.append_simulations(
            theta=ctx_data, x=xt_data
        ).train(
            training_batch_size=batch_size,
            learning_rate=lr,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=epochs
        )
        
        self.estimator.eval()
        print(f"\nNLE training took {time.time() - start_time:.2f}s")

        if self.model_cache_path is not None:
            self.save_model(
                self.model_cache_path,
                n_params=n_params,
                ref_noize=ref_noize,
                max_n_steps=max_n_steps,
                max_num_epochs=epochs,
                stop_after_epochs=stop_after_epochs,
                samples_per_theta=samples_per_theta,
                x_ref_shape=x_ref_shape,
                observed_dims=observed_dims,
                state_clamp_bounds=state_clamp_bounds,
                noisy_init_strategy=noisy_init_strategy,
                max_noisy_init_attempts=max_noisy_init_attempts,
            )
        
        return self

    def save_model(
            self,
            model_path: PathLike,
            n_params: int | None = None,
            ref_noize: float | None = None,
            max_n_steps: int | None = None,
            max_num_epochs: int | None = None,
            stop_after_epochs: int | None = None,
            samples_per_theta: int | None = None,
            x_ref_shape: Tuple[int, ...] | None = None,
            observed_dims: torch.Tensor | None = None,
            state_clamp_bounds: tuple[float | None, float | None] | None = (0.0, 1e4),
            noisy_init_strategy: Literal["clamp", "resample"] = "clamp",
            max_noisy_init_attempts: int = 1000) -> None:
        if self.estimator is None:
            raise ValueError("No trained estimator is available to save.")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "estimator": self.estimator,
                "metadata": {
                    "device": self.device,
                    "conditions_on_n_steps": self.conditions_on_n_steps,
                    "target_type": self.target_type,
                    "density_model": self.density_model,
                    "hidden_features": self.hidden_features,
                    "num_transforms": self.num_transforms,
                    "num_bins": self.num_bins,
                    "n_params": n_params,
                    "ref_noize": ref_noize,
                    "max_n_steps": max_n_steps,
                    "max_num_epochs": max_num_epochs,
                    "stop_after_epochs": stop_after_epochs,
                    "samples_per_theta": samples_per_theta,
                    "x_ref_shape": x_ref_shape,
                    "observed_dims": (
                        observed_dims.detach().cpu()
                        if observed_dims is not None
                        else None
                    ),
                    "state_clamp_bounds": state_clamp_bounds,
                    "state_clamp_applies_to_x_init": True,
                    "noisy_init_strategy": noisy_init_strategy,
                    "max_noisy_init_attempts": max_noisy_init_attempts,
                },
                "member_variables": {
                    "dynamics": self.dynamics,
                    "sampling_dist": self.sampling_dist,
                    "device": self.device,
                    "conditions_on_n_steps": self.conditions_on_n_steps,
                    "target_type": self.target_type,
                    "density_model": self.density_model,
                    "hidden_features": self.hidden_features,
                    "num_transforms": self.num_transforms,
                    "num_bins": self.num_bins,
                },
                "xt_data": getattr(self, "xt_data", None),
                "ctx_data": getattr(self, "ctx_data", None),
            },
            model_path,
        )
        print(f"Saved NLE model to {model_path}")

    def load_model(self, model_path: PathLike) -> 'NLEEstimator':
        model_path = Path(model_path)
        cache = self._load_model_cache(model_path)
        self.estimator = cache["estimator"]
        metadata = cache.get("metadata", {})
        if "conditions_on_n_steps" in metadata:
            self.conditions_on_n_steps = metadata["conditions_on_n_steps"]
        if "target_type" in metadata:
            self.target_type = metadata["target_type"]
        if "density_model" in metadata:
            self.density_model = metadata["density_model"]
        if "hidden_features" in metadata:
            self.hidden_features = metadata["hidden_features"]
        if "num_transforms" in metadata:
            self.num_transforms = metadata["num_transforms"]
        if "num_bins" in metadata:
            self.num_bins = metadata["num_bins"]
        self.cached_member_variables = cache.get("member_variables", {})
        self.xt_data = cache.get("xt_data")
        self.ctx_data = cache.get("ctx_data")

        if hasattr(self.estimator, "to"):
            self.estimator.to(self.device)
        if hasattr(self.estimator, "eval"):
            self.estimator.eval()

        print(f"Loaded NLE model from {model_path}")
        return self

    @staticmethod
    def _normalize_bounds(
            bounds: tuple[float | None, float | None] | None,
            *,
            argument_name: str) -> tuple[float | None, float | None] | None:
        """Validate optional lower/upper bounds and return a stable tuple."""
        if bounds is None:
            return None
        if len(bounds) != 2:
            raise ValueError(f"{argument_name} must contain (lower, upper).")
        lower, upper = bounds
        if lower is not None:
            lower = float(lower)
        if upper is not None:
            upper = float(upper)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"{argument_name} must satisfy lower <= upper."
            )
        return lower, upper

    @staticmethod
    def _clamp_to_bounds(
            value: torch.Tensor,
            bounds: tuple[float | None, float | None] | None,
            *,
            argument_name: str) -> torch.Tensor:
        """Apply optional bounds without imposing a state-space assumption."""
        normalized = NLEEstimator._normalize_bounds(
            bounds,
            argument_name=argument_name,
        )
        if normalized is None:
            return value
        lower, upper = normalized
        if lower is not None and upper is not None:
            return torch.clamp(value, min=lower, max=upper)
        if lower is not None:
            return torch.clamp_min(value, lower)
        if upper is not None:
            return torch.clamp_max(value, upper)
        return value

    @staticmethod
    def _is_within_bounds(
            value: torch.Tensor,
            bounds: tuple[float | None, float | None] | None) -> bool:
        """Return whether every component lies inside optional state bounds."""
        normalized = NLEEstimator._normalize_bounds(
            bounds,
            argument_name="state_clamp_bounds",
        )
        if normalized is None:
            return True
        lower, upper = normalized
        if lower is not None and torch.any(value < lower):
            return False
        if upper is not None and torch.any(value > upper):
            return False
        return True

    def _prepare_initial_state_dimensions(
            self,
            *,
            x_ref: torch.Tensor,
            observed_dims: torch.Tensor | None,
            unobserved_init_dist: Distribution | None,
            ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Validate the fixed state layout once before generating transitions."""
        if x_ref.ndim != 2:
            raise ValueError("x_ref must have shape (num_reference_times, state_dim).")
        if observed_dims is None:
            if x_ref.shape[1] != self.dynamics.x_dim:
                raise ValueError(
                    f"Full-state x_ref must have {self.dynamics.x_dim} columns."
                )
            return None, None

        observed_dims = torch.as_tensor(
            observed_dims, dtype=torch.long, device=x_ref.device
        )
        if observed_dims.ndim != 1:
            raise ValueError("observed_dims must be one-dimensional.")
        if x_ref.shape[1] != observed_dims.numel():
            raise ValueError("x_ref columns must match the number of observed_dims.")
        if torch.unique(observed_dims).numel() != observed_dims.numel():
            raise ValueError("observed_dims entries must be unique.")
        if torch.any((observed_dims < 0) | (observed_dims >= self.dynamics.x_dim)):
            raise ValueError("observed_dims contains an invalid state index.")

        all_dims = torch.arange(self.dynamics.x_dim, device=x_ref.device)
        unobserved_dims = all_dims[~torch.isin(all_dims, observed_dims)]
        if unobserved_dims.numel() == 0:
            raise ValueError("Use observed_dims=None when x_ref contains the full state.")
        if unobserved_init_dist is None:
            raise ValueError(
                "unobserved_init_dist is required when x_ref omits "
                "latent-state dimensions."
            )
        sample_shape = (
            unobserved_init_dist.batch_shape + unobserved_init_dist.event_shape
        )
        if sample_shape != torch.Size([unobserved_dims.numel()]):
            raise ValueError(
                "unobserved_init_dist.sample() must have shape "
                f"({unobserved_dims.numel()},)."
            )
        return observed_dims, unobserved_dims

    def _draw_noisy_initial_state(
            self,
            *,
            x_ref_value: torch.Tensor,
            ref_noize: float,
            observed_dims: torch.Tensor | None,
            unobserved_dims: torch.Tensor | None,
            unobserved_init_dist: Distribution | None) -> torch.Tensor:
        """Draw one complete initial state around a reference-state row."""
        noisy_reference = (
            x_ref_value + torch.randn_like(x_ref_value) * ref_noize
        )
        if observed_dims is None:
            return noisy_reference

        x_init = torch.empty(
            self.dynamics.x_dim,
            dtype=x_ref_value.dtype,
            device=x_ref_value.device,
        )
        x_init[observed_dims] = noisy_reference
        x_init[unobserved_dims] = torch.as_tensor(
            unobserved_init_dist.sample(),
            dtype=x_ref_value.dtype,
            device=x_ref_value.device,
        )
        return x_init

    def _sample_n_steps(self, max_n_steps: int | None) -> int:
        if max_n_steps is None:
            return 1
        return torch.randint(1, max_n_steps + 1, (1,)).item()

    def _load_model_cache(self, model_path: Path) -> dict:
        # The cache stores the trained estimator object, not only tensors.
        # This intentionally uses pickle-backed loading, so only load trusted cache files.
        return torch.load(model_path, map_location="cpu", weights_only=False)

    def _target_scale(
            self,
            n_steps: torch.Tensor,
            dtype: torch.dtype,
            device: torch.device) -> torch.Tensor:
        delta_t = torch.as_tensor(n_steps, device=device, dtype=dtype) * float(self.dynamics.dt)
        return torch.sqrt(delta_t).unsqueeze(-1)

    def _to_training_target(
            self,
            x_prev: torch.Tensor,
            x_next: torch.Tensor,
            n_steps: int | torch.Tensor) -> torch.Tensor:
        """Map x_next to the density-estimator target space."""
        if self.target_type == "x_next":
            return x_next
        scale = self._target_scale(
            torch.as_tensor(n_steps, device=x_next.device),
            dtype=x_next.dtype,
            device=x_next.device,
        ).squeeze(0)
        return (x_next - x_prev) / scale

    def _target_to_x_next(
            self,
            x_prev: torch.Tensor,
            target_sample: torch.Tensor,
            n_steps: torch.Tensor) -> torch.Tensor:
        """Map a density-estimator sample back to x_next space."""
        if self.target_type == "x_next":
            return target_sample
        scale = self._target_scale(n_steps, dtype=target_sample.dtype, device=target_sample.device)
        return x_prev + scale * target_sample

    def _x_next_to_target(
            self,
            x_prev: torch.Tensor,
            x_next: torch.Tensor,
            n_steps: torch.Tensor) -> torch.Tensor:
        """Map x_next observations to the density-estimator target space."""
        if self.target_type == "x_next":
            return x_next
        scale = self._target_scale(n_steps, dtype=x_next.dtype, device=x_next.device)
        return (x_next - x_prev) / scale

    def _build_transition_context(
            self,
            theta: torch.Tensor,
            x_prev: torch.Tensor,
            n_steps: int | torch.Tensor) -> torch.Tensor:
        """Build the estimator condition from transition-level inputs."""
        context = torch.cat([theta, x_prev], dim=-1)
        if not self.conditions_on_n_steps:
            return context
        n_steps_tensor = torch.as_tensor(
            n_steps,
            device=context.device,
            dtype=context.dtype,
        )
        batch_shape = context.shape[:-1]
        if n_steps_tensor.shape == batch_shape + (1,):
            n_steps_tensor = n_steps_tensor.squeeze(-1)
        n_steps_tensor = torch.broadcast_to(n_steps_tensor, batch_shape)
        return torch.cat([context, n_steps_tensor.unsqueeze(-1)], dim=-1)

    def transition_log_prob(
            self,
            x_next: torch.Tensor,
            *,
            theta: torch.Tensor,
            x_prev: torch.Tensor,
            n_steps: int | torch.Tensor,
            include_jacobian: bool = True) -> torch.Tensor:
        """
        Evaluate log p(x_next | x_prev, theta, n_steps).

        For target_type='scaled_dx', the flow is trained on
        r = (x_next - x_prev) / sqrt(n_steps * dt).  The Jacobian term is
        included by default so the returned density is in x_next space.
        """
        if self.estimator is None:
            raise ValueError("No trained estimator is available.")
        context = self._build_transition_context(theta, x_prev, n_steps)
        target = self._x_next_to_target(x_prev=x_prev, x_next=x_next, n_steps=n_steps)
        log_prob = self.estimator.log_prob(target.unsqueeze(0), condition=context)
        if self.target_type == "scaled_dx" and include_jacobian:
            D = x_next.shape[-1]
            scale = self._target_scale(n_steps, dtype=x_next.dtype, device=x_next.device).squeeze(-1)
            log_prob = log_prob - D * torch.log(scale)
        return log_prob

    def sample_transition(
            self,
            *,
            theta: torch.Tensor,
            x_prev: torch.Tensor,
            n_steps: int | torch.Tensor) -> torch.Tensor:
        """Draw one x_next sample for each transition input row."""
        if self.estimator is None:
            raise ValueError("No trained estimator is available.")
        context = self._build_transition_context(theta, x_prev, n_steps)
        target_sample = self.estimator.sample((1,), condition=context)[0]
        return self._target_to_x_next(x_prev=x_prev, target_sample=target_sample, n_steps=n_steps)
