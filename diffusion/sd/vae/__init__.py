"""Variational autoencoder components for latent diffusion."""

from .model import AutoencoderKL, DiagonalGaussianDistribution

__all__ = ["AutoencoderKL", "DiagonalGaussianDistribution"]
