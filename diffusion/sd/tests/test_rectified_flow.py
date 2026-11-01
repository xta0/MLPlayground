import unittest

import torch

from rectified_flow import RectifiedFlowScheduler


class RectifiedFlowSchedulerTests(unittest.TestCase):
    def test_shifted_timesteps_match_metal_scheduler(self):
        scheduler = RectifiedFlowScheduler(
            num_inference_steps=4,
            time_shift=3.0,
        )

        expected = torch.tensor([1.0, 0.9, 0.75, 0.5])
        torch.testing.assert_close(scheduler.timesteps, expected)

    def test_uniform_schedule_when_time_shift_is_one(self):
        scheduler = RectifiedFlowScheduler(
            num_inference_steps=4,
            time_shift=1.0,
        )

        expected = torch.tensor([1.0, 0.75, 0.5, 0.25])
        torch.testing.assert_close(scheduler.timesteps, expected)

    def test_add_noise_uses_noise_at_t_one_and_data_at_t_zero(self):
        clean = torch.tensor([[[[1.0]]], [[[3.0]]]])
        noise = torch.tensor([[[[5.0]]], [[[7.0]]]])
        timesteps = torch.tensor([0.0, 1.0])

        noisy, target_velocity = RectifiedFlowScheduler.add_noise(
            clean,
            noise,
            timesteps,
        )

        torch.testing.assert_close(noisy[0], clean[0])
        torch.testing.assert_close(noisy[1], noise[1])
        torch.testing.assert_close(target_velocity, noise - clean)

    def test_perfect_velocity_recovers_clean_sample(self):
        generator = torch.Generator().manual_seed(42)
        clean = torch.randn(2, 4, 8, 8, generator=generator)
        noise = torch.randn(2, 4, 8, 8, generator=generator)
        velocity = noise - clean
        scheduler = RectifiedFlowScheduler(
            num_inference_steps=7,
            time_shift=3.0,
        )

        sample = noise
        for step_index in range(scheduler.num_inference_steps):
            sample = scheduler.step(velocity, step_index, sample)

        torch.testing.assert_close(sample, clean, atol=1e-6, rtol=1e-6)

    def test_step_rejects_an_out_of_range_index(self):
        scheduler = RectifiedFlowScheduler(num_inference_steps=2)
        sample = torch.zeros(1, 4, 8, 8)

        with self.assertRaises(IndexError):
            scheduler.step(sample, 2, sample)


if __name__ == "__main__":
    unittest.main()
