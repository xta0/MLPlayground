import unittest

import torch
from torch import nn

from sd import LatentRectifiedFlow, measure_latent_statistics


class FixedPosterior:
    def __init__(self, latents):
        self.latents = latents

    def sample(self):
        return self.latents

    def mode(self):
        return self.latents


class ScalingVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def encode(self, images):
        return FixedPosterior(images * 2.0)


class ConstantVelocityModel(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.register_buffer("velocity", velocity)
        self.received_timesteps = []

    def forward(self, sample, timesteps):
        self.received_timesteps.append(timesteps.detach().clone())
        return self.velocity.expand_as(sample)


class LatentRectifiedFlowTests(unittest.TestCase):
    def test_sample_recovers_clean_latents_with_perfect_velocity(self):
        clean = torch.tensor([[[[1.0]]], [[[3.0]]]])
        noise = torch.tensor([[[[5.0]]], [[[7.0]]]])
        model = ConstantVelocityModel(noise - clean)
        flow = LatentRectifiedFlow(num_inference_steps=4, time_shift=3.0)

        result = flow.sample(
            model,
            num_samples=2,
            latent_shape=(1, 1, 1),
            initial_noise=noise,
        )

        torch.testing.assert_close(result, clean)

    def test_sample_scales_normalized_time_for_the_current_unet(self):
        noise = torch.zeros(1, 1, 1, 1)
        model = ConstantVelocityModel(torch.zeros_like(noise))
        flow = LatentRectifiedFlow(
            num_inference_steps=2,
            time_shift=1.0,
            model_time_scale=1000.0,
        )

        flow.sample(
            model,
            num_samples=1,
            latent_shape=(1, 1, 1),
            initial_noise=noise,
        )

        torch.testing.assert_close(
            torch.cat(model.received_timesteps),
            torch.tensor([1000.0, 500.0]),
        )

    def test_sample_restores_training_mode(self):
        noise = torch.zeros(1, 1, 1, 1)
        model = ConstantVelocityModel(torch.zeros_like(noise)).train()
        flow = LatentRectifiedFlow(num_inference_steps=1)

        flow.sample(
            model,
            num_samples=1,
            latent_shape=(1, 1, 1),
            initial_noise=noise,
        )

        self.assertTrue(model.training)

    def test_sample_validates_initial_noise_shape(self):
        model = ConstantVelocityModel(torch.zeros(1, 1, 1, 1))
        flow = LatentRectifiedFlow(num_inference_steps=1)

        with self.assertRaises(ValueError):
            flow.sample(
                model,
                num_samples=2,
                latent_shape=(1, 1, 1),
                initial_noise=torch.zeros(1, 1, 1, 1),
            )


class MeasureLatentStatisticsTests(unittest.TestCase):
    def test_measures_streaming_statistics_and_freezes_vae(self):
        vae = ScalingVAE().train()
        batches = [
            (torch.tensor([[[[1.0]]], [[[3.0]]]]),),
            (torch.tensor([[[[5.0]]]]),),
        ]

        statistics = measure_latent_statistics(vae, batches)

        self.assertAlmostEqual(statistics["mean"], 6.0)
        self.assertAlmostEqual(statistics["std"], (32.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(
            statistics["scaling_factor"],
            (3.0 / 32.0) ** 0.5,
        )
        self.assertEqual(statistics["image_count"], 3)
        self.assertEqual(statistics["batch_count"], 2)
        self.assertFalse(vae.training)
        self.assertFalse(vae.anchor.requires_grad)

    def test_honors_max_batches(self):
        vae = ScalingVAE()
        batches = [
            torch.tensor([[[[1.0]]], [[[2.0]]]]),
            torch.tensor([[[[100.0]]]]),
        ]

        statistics = measure_latent_statistics(
            vae,
            batches,
            max_batches=1,
        )

        self.assertAlmostEqual(statistics["mean"], 3.0)
        self.assertAlmostEqual(statistics["std"], 1.0)
        self.assertEqual(statistics["batch_count"], 1)


if __name__ == "__main__":
    unittest.main()
