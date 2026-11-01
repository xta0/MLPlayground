"""
Rectified-flow training and inference utilities.

* ``t = 0`` is clean data.
* ``t = 1`` is Gaussian noise.
* Sampling integrates the learned velocity from ``t = 1`` down to ``t = 0``.

For the straight probability path

    x_t = (1 - t) * x_0 + t * noise

the flow-matching target is the constant velocity ``noise - x_0``.
"""

import torch


def _as_batch_timesteps(timesteps, reference):
    """Validate timesteps and reshape them for image-like broadcasting."""
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.as_tensor(
            timesteps,
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        timesteps = timesteps.to(device=reference.device, dtype=reference.dtype)

    if timesteps.ndim == 0:
        timesteps = timesteps.expand(reference.shape[0])
    if timesteps.ndim != 1 or timesteps.shape[0] != reference.shape[0]:
        raise ValueError(
            "timesteps must be a scalar or have shape [batch_size]; "
            f"received {tuple(timesteps.shape)} for batch size "
            f"{reference.shape[0]}"
        )
    if not torch.all((timesteps >= 0) & (timesteps <= 1)):
        raise ValueError("timesteps must lie in the closed interval [0, 1]")

    return timesteps.view(reference.shape[0], *([1] * (reference.ndim - 1)))


class RectifiedFlowScheduler:
    """Euler scheduler matching the existing Metal implementation.

    ``time_shift`` is the Metal scheduler's ``alpha`` parameter. Values above
    one make adjacent steps smaller near the noisy endpoint (``t = 1``) and
    larger near the clean endpoint (``t = 0``). A value of one is uniform.
    """

    def __init__(self, num_inference_steps=50, time_shift=3.0):
        if time_shift <= 0:
            raise ValueError("time_shift must be positive")

        self.time_shift = float(time_shift)
        self.num_inference_steps = None
        self.timesteps = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps,
        device=None,
        dtype=torch.float32,
    ):
        """Create the descending, shifted inference schedule.

        As in the Metal implementation, zero is not stored in ``timesteps``;
        it is the implicit destination of the final Euler step.
        """
        if isinstance(num_inference_steps, bool) or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        if int(num_inference_steps) != num_inference_steps:
            raise ValueError("num_inference_steps must be a positive integer")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")

        self.num_inference_steps = int(num_inference_steps)
        step_indices = torch.arange(
            self.num_inference_steps,
            device=device,
            dtype=dtype,
        )
        uniform_timesteps = 1.0 - step_indices / self.num_inference_steps
        shift = torch.as_tensor(
            self.time_shift,
            device=device,
            dtype=dtype,
        )
        self.timesteps = (
            shift
            * uniform_timesteps
            / (1.0 + (shift - 1.0) * uniform_timesteps)
        )
        return self.timesteps

    def step(self, model_output, step_index, sample):
        """Take one explicit-Euler step toward clean data.

        ``model_output`` must be the velocity ``dx_t / dt``. Indexing the
        schedule directly avoids unreliable floating-point timestep lookup.
        """
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise TypeError("step_index must be an integer")
        if step_index < 0 or step_index >= self.num_inference_steps:
            raise IndexError(
                f"step_index must be in [0, {self.num_inference_steps - 1}]"
            )
        if model_output.shape != sample.shape:
            raise ValueError(
                "model_output and sample must have the same shape; received "
                f"{tuple(model_output.shape)} and {tuple(sample.shape)}"
            )

        timesteps = self.timesteps.to(device=sample.device, dtype=sample.dtype)
        current_timestep = timesteps[step_index]
        if step_index + 1 < self.num_inference_steps:
            next_timestep = timesteps[step_index + 1]
        else:
            next_timestep = torch.zeros((), device=sample.device, dtype=sample.dtype)

        dt = next_timestep - current_timestep
        return sample + dt * model_output

    @staticmethod
    def sample_training_timesteps(batch_size, device=None, dtype=torch.float32):
        """Sample uniform continuous times for flow-matching training."""
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if int(batch_size) != batch_size:
            raise ValueError("batch_size must be a positive integer")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        return torch.rand(int(batch_size), device=device, dtype=dtype)

    @staticmethod
    def add_noise(clean_samples, noise, timesteps):
        """Construct ``x_t`` and its supervised flow-matching velocity.

        Returns:
            A pair ``(noisy_samples, target_velocity)`` where
            ``target_velocity = noise - clean_samples``.
        """
        if clean_samples.shape != noise.shape:
            raise ValueError(
                "clean_samples and noise must have the same shape; received "
                f"{tuple(clean_samples.shape)} and {tuple(noise.shape)}"
            )
        if clean_samples.ndim < 2:
            raise ValueError("clean_samples must include batch and feature dimensions")

        t = _as_batch_timesteps(timesteps, clean_samples)
        noisy_samples = (1.0 - t) * clean_samples + t * noise
        target_velocity = noise - clean_samples
        return noisy_samples, target_velocity
