import torch


class EMA:
    def __init__(self, beta=0.9999, warmup_steps=2000):
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        self.beta = beta
        self.warmup_steps = warmup_steps
        self.step = 0

    @torch.no_grad()
    def update(self, ema_model, model):
        """Update ``ema_model`` after one optimizer step on ``model``."""
        if self.step < self.warmup_steps:
            # During warmup, keep the EMA model identical to the training model.
            ema_model.load_state_dict(model.state_dict())
        else:
            for ema_param, model_param in zip(
                ema_model.parameters(),
                model.parameters(),
            ):
                ema_param.mul_(self.beta).add_(
                    model_param,
                    alpha=1.0 - self.beta,
                )

            # Buffers are not learned by the optimizer, so copy rather than
            # average them. This also supports modules such as BatchNorm if
            # they are added to the model later.
            for ema_buffer, model_buffer in zip(
                ema_model.buffers(),
                model.buffers(),
            ):
                ema_buffer.copy_(model_buffer)

        self.step += 1
