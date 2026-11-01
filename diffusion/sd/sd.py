import argparse
import torch
from rectified_flow import RectifiedFlowScheduler


@torch.no_grad()
def measure_latent_statistics(
    vae,
    dataloader,
    device=None,
    max_batches=None,
    sample_posterior=True,
):
    """Freeze a VAE and estimate scalar statistics of its encoded latents.

    Statistics are accumulated on the CPU in float64 without retaining whole
    batches.  The returned ``scaling_factor`` is ``1 / std``; multiplying VAE
    latents by it gives the diffusion model data with approximately unit
    standard deviation.
    """
    if max_batches is not None:
        if isinstance(max_batches, bool) or not isinstance(max_batches, int):
            raise ValueError("max_batches must be a positive integer or None")
        if max_batches <= 0:
            raise ValueError("max_batches must be a positive integer or None")

    if device is None:
        first_tensor = next(vae.parameters(), None)
        if first_tensor is None:
            first_tensor = next(vae.buffers(), None)
        device = (
            first_tensor.device
            if first_tensor is not None
            else torch.device("cpu")
        )
    else:
        device = torch.device(device)

    vae.to(device).eval()
    vae.requires_grad_(False)

    value_sum = 0.0
    squared_value_sum = 0.0
    value_count = 0
    image_count = 0
    batch_count = 0

    for batch in dataloader:
        if max_batches is not None and batch_count >= max_batches:
            break

        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not isinstance(images, torch.Tensor):
            raise TypeError("Each batch must be a tensor or start with a tensor")
        if images.ndim < 2:
            raise ValueError("Image batches must include batch and feature dimensions")

        posterior = vae.encode(images.to(device))
        latents = posterior.sample() if sample_posterior else posterior.mode()
        values = latents.detach().to(device="cpu", dtype=torch.float64)

        value_sum += values.sum().item()
        squared_value_sum += values.square().sum().item()
        value_count += values.numel()
        image_count += images.shape[0]
        batch_count += 1

    if value_count == 0:
        raise ValueError("The DataLoader produced no latent values")

    mean = value_sum / value_count
    variance = max(squared_value_sum / value_count - mean * mean, 0.0)
    std = variance**0.5
    if std == 0.0:
        raise ValueError("Encoded latents have zero standard deviation")

    return {
        "mean": mean,
        "std": std,
        "scaling_factor": 1.0 / std,
        "image_count": image_count,
        "batch_count": batch_count,
        "value_count": value_count,
    }


def sample(args):
    pass

def train(args):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="DDPM")
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Target total epoch count, including epochs loaded with --resume.",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--dataset-path", default="data")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20_000,
        help="Random training subset size; use 0 for the full CelebA split.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Seed used to select a reproducible random training subset.",
    )
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument(
        "--ema-beta",
        type=float,
        default=0.9999,
        help="Decay used to average model weights for sampling.",
    )
    parser.add_argument(
        "--ema-warmup-steps",
        type=int,
        default=2_000,
        help="Optimizer steps before exponential averaging begins.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume, for example models/DDPM/ckpt.pt.",
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
