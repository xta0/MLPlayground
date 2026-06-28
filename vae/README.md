# VAE Training and Inference

This project trains a convolutional variational autoencoder on 64x64 RGB images.
The main entrypoints are:

- `vae_training.py` for training and checkpoint creation.
- `vae_inference.py` for loading a checkpoint and saving reconstruction/generated image grids.

## Dataset Layout

For CelebA aligned images, use a flat image folder:

```text
img_align_celeba/
  000001.jpg
  000002.jpg
  ...
```

Run it with `--dataset celeba64`:

```bash
python vae_training.py --dataset celeba64 --data-root ./img_align_celeba
```

`imagefolder64` is also available for Torchvision `ImageFolder` style datasets:

```text
my_dataset/
  class_a/
    image1.jpg
  class_b/
    image2.jpg
```

## Training

Basic CelebA training:

```bash
python vae_training.py --dataset celeba64 --data-root ./img_align_celeba
```

Common options:

```bash
python vae_training.py \
  --dataset celeba64 \
  --data-root ./img_align_celeba \
  --image-size 64 \
  --batch-size 64 \
  --epochs 50 \
  --lr 3e-04 \
  --latent-dim 256 \
  --feature-channels 512
```

Training uses a fixed size-normalized reconstruction MSE plus KL term:

```text
loss = mean_pixel_mse + 1e-3 * kld
```

By default, training writes:

```text
vae_celeba64_64.pth
vae_celeba64_64.json
```

Use `--model-file` to choose a different checkpoint path:

```bash
python vae_training.py --dataset celeba64 --data-root ./img_align_celeba --model-file vae_faces.pth
```

If `--model-file` already exists, the training script loads it instead of training a new model.

## Inference

Generate samples and save reconstructions:

```bash
python vae_inference.py \
  --model-file vae_celeba64_64.pth \
  --data-root ./img_align_celeba \
  --output-path .
```

`--output-path` is a directory. By default, inference writes to the current directory:

```text
reconstructions.png
sample_000.png
sample_001.png
...
generated_grid.png
```

Write all inference outputs to a separate folder:

```bash
python vae_inference.py \
  --model-file vae_celeba64_64.pth \
  --data-root ./img_align_celeba \
  --output-path samples
```

The inference script reads the matching JSON config next to the model file when it exists,
for example `vae_celeba64_64.json`. You can override those values with CLI flags such as
`--latent-dim`, `--feature-channels`, and `--image-size`.

Useful inference options:

```bash
python vae_inference.py \
  --model-file vae_celeba64_64.pth \
  --data-root ./img_align_celeba \
  --sample-count 32 \
  --reconstruction-count 8 \
  --temperature 0.8 \
  --nrow 8 \
  --output-path samples
```

## Script Reference

Training:

```bash
python vae_training.py --help
```

Inference:

```bash
python vae_inference.py --help
```
