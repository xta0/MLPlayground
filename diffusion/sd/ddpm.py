import argparse
import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ema import EMA
from unet import UNet
from utils import get_data, save_images, setup_logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        device="mps",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)  # [1000]
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def add_noise(self, x0, t):
        """
        x_t = sqrt(alpha_hat) * x_0 + sqrt(1 - alpha_hat) * epsilon
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        epsilon = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def denoise_and_add_noise(self, x, t, predicted_noise, z):
        """
        Take one reverse-diffusion step and inject noise with variance beta_t.
        """
        alpha = self.alpha[t].view(-1, 1, 1, 1).to(self.device)
        alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1).to(self.device)
        beta = self.beta[t].view(-1, 1, 1, 1).to(self.device)

        return (
            1
            / torch.sqrt(alpha)
            * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise)
            + torch.sqrt(beta) * z
        )

    def sample_timesteps(self, n):
        return torch.randint(
            low=0, high=self.noise_steps, size=(n,), device=self.device
        )

    def sample(
        self,
        model,
        num_samples,
        save_intermediate=False,
        save_rate=20,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if save_intermediate and save_rate <= 0:
            raise ValueError("save_rate must be positive")

        logging.info(f"Sampling {num_samples} new images....")

        was_training = model.training
        model.eval()

        with torch.no_grad():
            # x_T ~ N(0, 1), sample initial noise
            x = torch.randn(num_samples, 3, self.img_size, self.img_size).to(
                self.device
            )

            # Intermediate states are useful for visualizing denoising, but
            # copying them to CPU every few steps is unnecessary during normal
            # training-time sampling.
            intermediate = [x.cpu().numpy()] if save_intermediate else []

            for i in range(self.noise_steps - 1, 0, -1):
                if i % 100 == 0:
                    logging.info(f"Sampling timestep {i:3d}")

                t = torch.full(
                    (num_samples,),
                    i,
                    device=self.device,
                    dtype=torch.long,
                )
                predicted_noise = model(x, t)

                # sample a random noise to inject back into the image, except for the last step
                z = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = self.denoise_and_add_noise(x, t, predicted_noise, z)

                if save_intermediate and (i % save_rate == 0 or i < 8):
                    intermediate.append(x.detach().cpu().numpy())

        if was_training:
            model.train()

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        stacked_intermediate = np.stack(intermediate) if save_intermediate else None
        return x, stacked_intermediate


def load_training_checkpoint(
    path,
    model,
    optimizer,
    device,
    ema_model=None,
    ema=None,
):
    """Restore training state and return the next epoch and global step."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required_keys = {"epoch", "model", "optimizer"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Checkpoint is missing required keys: {missing}")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if ema_model is not None:
        if "ema_model" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model"])
            if ema is not None:
                ema.step = int(checkpoint.get("ema_step", 0))
                if ema.step < 0:
                    raise ValueError("Checkpoint EMA step must be non-negative")
        else:
            # Checkpoints created before EMA support have no averaged weights.
            # Starting from the restored training weights is the correct
            # initialization; EMA warmup begins with the next optimizer step.
            ema_model.load_state_dict(model.state_dict())

    # torch.load(..., map_location="cpu") keeps checkpoint loading portable.
    # Move AdamW's running statistics to the training device afterward.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    start_epoch = int(checkpoint["epoch"])
    if start_epoch < 0:
        raise ValueError("Checkpoint epoch must be non-negative")

    # Older checkpoints did not store global_step. The caller supplies the
    # fallback after the DataLoader is built because it knows its length.
    global_step = checkpoint.get("global_step")
    if global_step is not None:
        global_step = int(global_step)

    return start_epoch, global_step


def train(args):
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if args.sample_every <= 0:
        raise ValueError("sample_every must be positive")

    setup_logging(args.run_name)
    device = args.device
    logging.info(f"Using device: {device}")

    dataloader = get_data(args)
    if len(dataloader) == 0:
        raise ValueError(
            "The DataLoader has no batches. Check the dataset path or use a "
            "batch size no larger than the dataset when drop_last=True."
        )

    model = UNet().to(device)

    # The optimizer updates `model`; `ema_model` is a separate, frozen copy
    # used only for stable sample generation.
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    ema = EMA(
        beta=args.ema_beta,
        warmup_steps=args.ema_warmup_steps,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(log_dir=os.path.join("runs", args.run_name))
    batches_per_epoch = len(dataloader)

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, saved_global_step = load_training_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
            ema_model=ema_model,
            ema=ema,
        )
        global_step = (
            saved_global_step
            if saved_global_step is not None
            else start_epoch * batches_per_epoch
        )
        logging.info(
            f"Resumed {args.resume} after epoch {start_epoch} "
            f"at global step {global_step}."
        )

    if start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint has completed {start_epoch} epochs, but --epochs is "
            f"{args.epochs}. Set --epochs higher than {start_epoch}."
        )

    logging.info(
        f"Training on {len(dataloader.dataset)} images for "
        f"epochs {start_epoch + 1}-{args.epochs} with "
        f"{batches_per_epoch} batches per epoch."
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            logging.info(f"Starting epoch {epoch + 1}...")

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
            )

            for batch_index, (images, _) in enumerate(pbar):
                images = images.to(device)
                batch_size = images.shape[0]

                """
                Each image receives an integer timestep sampled uniformly from [1, noise_steps - 1].
                If the batch size is 4, t could be:

                tensor([734, 82, 451, 963])
                """

                t = diffusion.sample_timesteps(batch_size)
                x_t, noise = diffusion.add_noise(images, t)

                optimizer.zero_grad(set_to_none=True)
                predicted_noise = model(x_t, t)
                loss = mse(predicted_noise, noise)
                loss.backward()
                optimizer.step()

                # Smooth the newly updated weights. EMA is not involved in
                # backpropagation and does not have its own optimizer.
                ema.update(ema_model, model)

                loss_value = loss.item()
                pbar.set_postfix(loss=f"{loss_value:.4f}")
                logger.add_scalar("Loss/train", loss_value, global_step)
                global_step += 1

            # Save the completed epoch before sampling so an interruption in
            # sample generation cannot discard the epoch's training progress.
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema_model": ema_model.state_dict(),
                "ema_step": ema.step,
                "loss": loss_value,
                "global_step": global_step,
            }
            torch.save(
                checkpoint,
                os.path.join("models", args.run_name, "ckpt.pt"),
            )

            if (epoch + 1) % args.sample_every == 0:
                sampled_images, _ = diffusion.sample(
                    # Sampling from averaged weights is generally more stable
                    # than sampling from the latest, noisier training weights.
                    ema_model,
                    num_samples=args.sample_count,
                )
                save_images(
                    sampled_images,
                    os.path.join(
                        "results",
                        args.run_name,
                        f"{epoch + 1}.jpg",
                    ),
                )
    finally:
        logger.close()


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
