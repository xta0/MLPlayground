import argparse
import json
import os

import torch
from tqdm.auto import tqdm

from .data import SUPPORTED_DATASETS, prepare_image_data
from .loss import autoencoder_kl_loss, kl_weight_at_step
from .model import AutoencoderKL
from .utils import (
    get_device,
    load_torch_file,
    model_state_dict,
    move_optimizer_state,
    save_checkpoint,
)


@torch.no_grad()
def validate_autoencoder(
    model,
    validation_loader,
    kl_weight,
    mse_weight=0.25,
    image_count=8,
    progress_description=None,
):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "kl_loss": 0.0,
        "weighted_kl_loss": 0.0,
    }
    sample_count = 0
    fixed_images = None
    fixed_reconstructions = None

    try:
        progress = tqdm(
            validation_loader,
            desc=progress_description or "Validation",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=progress_description is None,
        )
        for batch in progress:
            images = batch[0].to(device)
            reconstructions, posterior = model(
                images,
                sample_posterior=False,
            )
            loss, reconstruction_loss, kl_loss = autoencoder_kl_loss(
                images,
                reconstructions,
                posterior,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
            )
            weighted_kl_loss = kl_weight * kl_loss
            batch_size = images.shape[0]

            totals["loss"] += loss.item() * batch_size
            totals["reconstruction_loss"] += reconstruction_loss.sum().item()
            totals["kl_loss"] += kl_loss.sum().item()
            totals["weighted_kl_loss"] += weighted_kl_loss.sum().item()
            sample_count += batch_size
            progress.set_postfix(
                loss=f"{totals['loss'] / sample_count:.4f}",
                recon=(
                    f"{totals['reconstruction_loss'] / sample_count:.4f}"
                ),
                kl=f"{totals['kl_loss'] / sample_count:.4f}",
                refresh=False,
            )

            if fixed_images is None:
                fixed_images = images[:image_count].detach().cpu()
                fixed_reconstructions = (
                    reconstructions[:image_count].detach().cpu()
                )
    finally:
        model.train(was_training)

    if sample_count == 0:
        raise ValueError("The validation DataLoader has no batches.")

    metrics = {name: total / sample_count for name, total in totals.items()}
    return metrics, fixed_images, fixed_reconstructions


def train_autoencoder(
    model,
    train_loader,
    epochs=50,
    lr=3e-4,
    kl_weight=0.1,
    kl_warmup_steps=10_000,
    mse_weight=0.25,
    checkpoint_every=10,
    checkpoint_dir=None,
    checkpoint_config=None,
    resume_checkpoint=None,
    writer=None,
    validation_loader=None,
    validation_image_count=8,
):
    device = get_device()
    print(f"Using device: {device}")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.5, 0.9),
        eps=1e-8,
    )
    start_epoch = 0
    global_step = 0
    best_validation_reconstruction_loss = float("inf")

    if resume_checkpoint is not None:
        required_keys = {"epoch", "global_step", "model", "optimizer"}
        missing_keys = required_keys.difference(resume_checkpoint)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"Resume checkpoint is missing required keys: {missing}")

        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
        start_epoch = int(resume_checkpoint["epoch"])
        global_step = int(resume_checkpoint["global_step"])
        best_validation_reconstruction_loss = float(
            resume_checkpoint.get(
                "best_validation_reconstruction_loss",
                float("inf"),
            )
        )
        rng_state = resume_checkpoint.get("torch_rng_state")
        if rng_state is not None:
            torch.random.set_rng_state(rng_state)
        if device.type == "mps" and "mps_rng_state" in resume_checkpoint:
            torch.mps.set_rng_state(resume_checkpoint["mps_rng_state"])
        if device.type == "cuda" and "cuda_rng_state_all" in resume_checkpoint:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state_all"])

        if start_epoch >= epochs:
            raise ValueError(
                f"Checkpoint has completed {start_epoch} epochs, but --epochs "
                f"is {epochs}. Set --epochs higher than {start_epoch}."
            )
        print(
            f"Resumed after epoch {start_epoch} at global step {global_step}."
        )

    losses = {
        "loss": [],
        "recon_loss": [],
        "kld_loss": [],
        "weighted_kld_loss": [],
        "kl_weight": [],
    }
    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_weighted_kl = 0.0
        batches = 0
        validation_metrics = None

        progress = tqdm(
            train_loader,
            desc=f"Train {epoch_number}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            x = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon_x, posterior = model(x)
            active_kl_weight = kl_weight_at_step(
                target_weight=kl_weight,
                global_step=global_step,
                warmup_steps=kl_warmup_steps,
            )
            loss, recon_loss, kld_loss = autoencoder_kl_loss(
                x,
                recon_x,
                posterior,
                kl_weight=active_kl_weight,
                mse_weight=mse_weight,
            )
            loss.backward()
            optimizer.step()

            weighted_kld_loss = active_kl_weight * kld_loss
            loss_value = loss.item()
            reconstruction_loss_value = recon_loss.mean().item()
            kl_loss_value = kld_loss.mean().item()
            weighted_kl_loss_value = weighted_kld_loss.mean().item()
            losses["loss"].append(loss_value)
            losses["recon_loss"].append(reconstruction_loss_value)
            losses["kld_loss"].append(kl_loss_value)
            losses["weighted_kld_loss"].append(weighted_kl_loss_value)
            losses["kl_weight"].append(active_kl_weight)

            if writer is not None:
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar(
                    "train/reconstruction_loss",
                    reconstruction_loss_value,
                    global_step,
                )
                writer.add_scalar(
                    "train/kl_loss",
                    kl_loss_value,
                    global_step,
                )
                writer.add_scalar(
                    "train/weighted_kl_loss",
                    weighted_kl_loss_value,
                    global_step,
                )
                writer.add_scalar(
                    "train/kl_weight",
                    active_kl_weight,
                    global_step,
                )

            epoch_loss += loss_value
            epoch_recon += reconstruction_loss_value
            epoch_kl += kl_loss_value
            epoch_weighted_kl += weighted_kl_loss_value
            batches += 1
            global_step += 1
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                recon=f"{reconstruction_loss_value:.4f}",
                kl=f"{kl_loss_value:.4f}",
                kl_weight=f"{active_kl_weight:.5f}",
                refresh=False,
            )

        print(
            f"Epoch [{epoch_number}/{epochs}], "
            f"Loss: {epoch_loss / batches:.4f}, "
            f"Recon Loss: {epoch_recon / batches:.4f}, "
            f"KLD Loss: {epoch_kl / batches:.4f}, "
            f"Weighted KLD: {epoch_weighted_kl / batches:.4f}, "
            f"KL Weight: {active_kl_weight:.6f}"
        )
        if writer is not None:
            writer.add_scalar("epoch/loss", epoch_loss / batches, epoch_number)
            writer.add_scalar(
                "epoch/reconstruction_loss",
                epoch_recon / batches,
                epoch_number,
            )
            writer.add_scalar(
                "epoch/kl_loss",
                epoch_kl / batches,
                epoch_number,
            )
            writer.add_scalar(
                "epoch/weighted_kl_loss",
                epoch_weighted_kl / batches,
                epoch_number,
            )
            writer.add_scalar(
                "epoch/kl_weight",
                active_kl_weight,
                epoch_number,
            )

        if validation_loader is not None:
            validation_metrics, validation_images, validation_reconstructions = (
                validate_autoencoder(
                    model,
                    validation_loader,
                    kl_weight=active_kl_weight,
                    mse_weight=mse_weight,
                    image_count=validation_image_count,
                    progress_description=(
                        f"Valid {epoch_number}/{epochs}"
                    ),
                )
            )
            print(
                f"Validation [{epoch_number}/{epochs}], "
                f"Loss: {validation_metrics['loss']:.4f}, "
                "Recon Loss: "
                f"{validation_metrics['reconstruction_loss']:.4f}, "
                f"KLD Loss: {validation_metrics['kl_loss']:.4f}, "
                "Weighted KLD: "
                f"{validation_metrics['weighted_kl_loss']:.4f}"
            )

            if writer is not None:
                for name, value in validation_metrics.items():
                    writer.add_scalar(
                        f"validation/{name}",
                        value,
                        epoch_number,
                    )

                validation_images = ((validation_images + 1.0) / 2.0).clamp(0, 1)
                validation_reconstructions = (
                    (validation_reconstructions + 1.0) / 2.0
                ).clamp(0, 1)
                writer.add_images(
                    "validation/originals",
                    validation_images,
                    epoch_number,
                )
                writer.add_images(
                    "validation/reconstructions",
                    validation_reconstructions,
                    epoch_number,
                )
        if checkpoint_dir is not None:
            is_best = (
                validation_metrics is not None
                and validation_metrics["reconstruction_loss"]
                < best_validation_reconstruction_loss
            )
            if is_best:
                best_validation_reconstruction_loss = validation_metrics[
                    "reconstruction_loss"
                ]

            checkpoint = {
                "version": 1,
                "epoch": epoch_number,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_validation_reconstruction_loss": (
                    best_validation_reconstruction_loss
                ),
                "validation_metrics": validation_metrics,
                "config": checkpoint_config or {},
                "torch_rng_state": torch.random.get_rng_state(),
            }
            if device.type == "mps":
                checkpoint["mps_rng_state"] = torch.mps.get_rng_state()
            elif device.type == "cuda":
                checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            latest_path = os.path.join(checkpoint_dir, "latest.pt")
            save_checkpoint(checkpoint, latest_path)
            print(f"Saved checkpoint: {latest_path}")

            if checkpoint_every > 0 and epoch_number % checkpoint_every == 0:
                periodic_path = os.path.join(
                    checkpoint_dir,
                    f"epoch_{epoch_number:03d}.pt",
                )
                save_checkpoint(checkpoint, periodic_path)
                print(f"Saved checkpoint: {periodic_path}")

            if is_best:
                best_path = os.path.join(checkpoint_dir, "best.pt")
                save_checkpoint(checkpoint, best_path)
                print(f"Saved best checkpoint: {best_path}")

    return model, losses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="celeba64")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Target total epoch count, including epochs restored by --resume.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--kl-warmup-steps", type=int, default=10_000)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Periodic checkpoint interval; 0 disables periodic snapshots.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/vae",
        help="Base directory containing one checkpoint directory per run.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume model, optimizer, epoch, and global step from a full checkpoint.",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Initialize model weights only and start a fresh training run.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="runs/vae")
    parser.add_argument("--validation-image-count", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size % 4 != 0:
        raise ValueError(
            "image_size must be divisible by 4 for the AutoencoderKL spatial VAE."
        )
    if args.kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    if args.kl_warmup_steps < 0:
        raise ValueError("kl_warmup_steps must be non-negative")
    if args.validation_image_count <= 0:
        raise ValueError("validation_image_count must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume and --init-from cannot be used together")

    train_loader = prepare_image_data(
        dataset_name=args.dataset,
        split="train",
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        normalize=True,
        shuffle=True,
        num_workers=args.num_workers,
        download=args.download,
        subset_fraction=args.train_fraction,
        subset_seed=args.subset_seed,
    )
    validation_loader = prepare_image_data(
        dataset_name=args.dataset,
        split="valid",
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        normalize=True,
        shuffle=False,
        num_workers=args.num_workers,
        download=False,
        subset_fraction=1.0,
        subset_seed=args.subset_seed,
    )

    batch = next(iter(train_loader))
    sample = batch[0]
    print("training sample:", sample.shape)
    print("value range:", sample.min().item(), sample.max().item())
    print("training images:", len(train_loader.dataset))
    print("validation images:", len(validation_loader.dataset))

    model = AutoencoderKL(
        in_channels=3,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        dropout=args.dropout,
    )

    latent_size = args.image_size // model.downsample_factor
    print(
        "spatial latent shape:",
        (args.batch_size, args.latent_channels, latent_size, latent_size),
    )

    run_name = (
        args.run_name
        or f"vae_{args.dataset}_{args.image_size}_f{model.downsample_factor}"
    )
    tensorboard_log_dir = os.path.join(args.log_dir, run_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    training_config = {
        **vars(args),
        "run_name": run_name,
        "latent_size": latent_size,
        "downsample_factor": model.downsample_factor,
        "tensorboard_log_dir": tensorboard_log_dir,
        "input_range": "[-1, 1]",
        "output_activation": "tanh",
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w") as config_file:
        json.dump(training_config, config_file, indent=2)

    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = load_torch_file(args.resume)
        if not isinstance(resume_checkpoint, dict):
            raise TypeError("--resume must point to a full training checkpoint")
    elif args.init_from is not None:
        initial_payload = load_torch_file(args.init_from)
        model.load_state_dict(model_state_dict(initial_payload))
        print(f"Initialized model weights from {args.init_from}.")

    from torch.utils.tensorboard import SummaryWriter

    purge_step = (
        int(resume_checkpoint.get("global_step", 0))
        if resume_checkpoint is not None
        else None
    )
    writer = SummaryWriter(
        log_dir=tensorboard_log_dir,
        purge_step=purge_step,
    )
    print(f"TensorBoard log directory: {tensorboard_log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    try:
        train_autoencoder(
            model,
            train_loader,
            epochs=args.epochs,
            lr=args.lr,
            kl_weight=args.kl_weight,
            kl_warmup_steps=args.kl_warmup_steps,
            mse_weight=args.mse_weight,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=checkpoint_dir,
            checkpoint_config=training_config,
            resume_checkpoint=resume_checkpoint,
            writer=writer,
            validation_loader=validation_loader,
            validation_image_count=args.validation_image_count,
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
