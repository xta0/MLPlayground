from torch.nn import functional as F


def kl_weight_at_step(target_weight, global_step, warmup_steps):
    if target_weight < 0:
        raise ValueError("target_weight must be non-negative")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if warmup_steps == 0:
        return target_weight

    progress = min(global_step / warmup_steps, 1.0)
    return target_weight * progress


def autoencoder_kl_loss(
    x,
    recon_x,
    posterior,
    kl_weight=0.1,
    mse_weight=0.25,
):
    l1_per_sample = F.l1_loss(
        recon_x,
        x,
        reduction="none",
    ).mean(dim=(1, 2, 3))
    mse_per_sample = F.mse_loss(
        recon_x,
        x,
        reduction="none",
    ).mean(dim=(1, 2, 3))
    recon_per_sample = l1_per_sample + mse_weight * mse_per_sample
    kl_per_sample = posterior.kl()
    loss = (recon_per_sample + kl_weight * kl_per_sample).mean()
    return loss, recon_per_sample, kl_per_sample
