# DDPM from Scratch

An educational implementation of an unconditional Denoising Diffusion
Probabilistic Model (DDPM) in PyTorch. The project trains a time-conditioned
UNet with self-attention to generate 64×64 CelebA faces. It includes Apple
Metal (MPS) support, exponential moving average (EMA) weights, resumable
checkpoints, TensorBoard logging, standalone sampling, and a real-versus-
generated comparison utility.

The code intentionally keeps the diffusion process explicit. It is designed
for learning and experimentation rather than as a production diffusion
framework.

## What is implemented

- Linear 1,000-step noise schedule
- Closed-form forward noising at arbitrary timesteps
- Noise-prediction (`epsilon`) training objective
- UNet with residual convolution blocks, timestep conditioning, skip
  connections, and spatial self-attention
- Group normalization, which does not depend on batch statistics
- EMA weights for more stable generation
- Checkpoints containing the model, optimizer, EMA model, EMA step, epoch,
  loss, and global training step
- Deterministic subset selection for smaller experiments
- Periodic sample generation and TensorBoard loss logging
- Standalone EMA sampling and CelebA comparison scripts

## Project layout

| File | Purpose |
| --- | --- |
| `ddpm.py` | Forward/reverse diffusion, training loop, checkpointing, and CLI |
| `unet.py` | Time-conditioned UNet and self-attention blocks |
| `ema.py` | Exponential moving average implementation |
| `utils.py` | CelebA transforms, DataLoader, image saving, and output directories |
| `sample.py` | Generate a grid from a trained checkpoint, preferring EMA weights |
| `compare_samples.py` | Place random real CelebA images above generated samples |

Generated and downloaded artifacts are intentionally ignored by Git:

```text
data/       CelebA images and metadata
models/     training checkpoints
results/    generated image grids
runs/       TensorBoard event files
```

## Diffusion formulation

The linear schedule contains `T = 1000` beta values from `1e-4` to `0.02`:

```text
beta_t       = linear_beta_schedule(t)
alpha_t      = 1 - beta_t
alpha_bar_t  = product(alpha_0 ... alpha_t)
```

For every image in a batch, training independently samples a timestep and
noise tensor. The clean image can then be noised directly without applying all
preceding steps:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

The UNet receives `x_t` and `t` and predicts `epsilon`. Training minimizes:

```text
MSE(epsilon_predicted, epsilon)
```

Generation begins with Gaussian noise and applies the learned reverse update
from timestep 999 down to 1. The implementation uses fixed `beta_t` variance,
with no extra random noise injected on the final reverse step.

## Model architecture

The default UNet has **22,267,907 trainable parameters**. Spatial dimensions
below assume the default 64×64 RGB input.

| Stage | Operation | Output shape |
| --- | --- | --- |
| Input | Noisy RGB image | `[B, 3, 64, 64]` |
| `inc` | DoubleConv | `[B, 64, 64, 64]` |
| `down1` | Pool, residual DoubleConv, channel DoubleConv | `[B, 128, 32, 32]` |
| `down2` | Down + self-attention | `[B, 256, 16, 16]` |
| `down3` | Down + self-attention | `[B, 256, 8, 8]` |
| Bottleneck | `256 → 512 → 512 → 256` | `[B, 256, 8, 8]` |
| `up1` | Upsample, concatenate `16×16` skip, conv, attention | `[B, 128, 16, 16]` |
| `up2` | Upsample, concatenate `32×32` skip, conv | `[B, 64, 32, 32]` |
| `up3` | Upsample, concatenate `64×64` skip, conv | `[B, 64, 64, 64]` |
| Output | 1×1 convolution | `[B, 3, 64, 64]` |

### DoubleConv and residual blocks

A regular `DoubleConv` is:

```text
Conv3×3 → GroupNorm(8) → GELU → Conv3×3 → GroupNorm(8)
```

When the block is residual, its input and convolution result are added and a
final GELU is applied. Residual blocks therefore require matching input and
output channels.

### Timestep conditioning

Integer timesteps are converted into a 256-dimensional sinusoidal encoding.
Every downsample and upsample block passes this encoding through:

```text
SiLU → Linear(256, output_channels)
```

The result is reshaped from `[B, C]` to `[B, C, 1, 1]` and broadcast across
the feature map. This tells every resolution how much noise is present in its
input.

### Self-attention

Attention is applied at the 16×16 and 8×8 resolutions. A feature map is
converted from `[B, C, H, W]` into image tokens shaped `[B, H*W, C]`, followed
by pre-normalized four-head self-attention and a feed-forward residual block.
Global attention is intentionally omitted at 32×32 and 64×64 because its
memory and compute grow quadratically with the number of spatial tokens.

## Environment setup

Python 3.12 was used during development. Create and activate an isolated
environment, then install the runtime dependencies:

```bash
conda create -n ddpm python=3.12
conda activate ddpm
pip install torch torchvision tensorboard tqdm numpy pillow matplotlib
```

`ruff` is optional but useful for formatting and linting:

```bash
pip install ruff
```

The CLI selects MPS automatically when it is available and otherwise uses the
CPU. NVIDIA users can pass `--device cuda` explicitly.

## CelebA dataset

Training uses torchvision's official CelebA `train` split, which contains
162,770 aligned images. On first use, ask torchvision to download the dataset:

```bash
python ddpm.py \
    --run-name DDPM-download-test \
    --epochs 1 \
    --batch-size 16 \
    --max-train-samples 1000 \
    --download
```

The default dataset root is `data`, and torchvision expects the CelebA files
under `data/celeba/`.

CelebA is a third-party dataset and is not distributed with this repository.
Review its official terms before using or redistributing the data or trained
artifacts.

Each training image receives the following transforms:

1. Center-crop the aligned 178×218 image to 178×178.
2. Resize to the configured image size (64×64 by default).
3. Apply a random horizontal flip.
4. Convert to a tensor in `[0, 1]`.
5. Normalize every RGB channel with mean `0.5` and standard deviation `0.5`,
   producing the model range `[-1, 1]`.

By default, training uses a deterministic random subset of 20,000 images.
Pass `--max-train-samples 0` to use the full official training split. The
subset selection is reproducible through `--subset-seed`.

## Training recipe

The full-data 64×64 recipe used for the main experiment is:

```bash
python ddpm.py \
    --run-name DDPM-full-ema \
    --epochs 20 \
    --batch-size 64 \
    --image-size 64 \
    --dataset-path data \
    --lr 1e-4 \
    --num-workers 4 \
    --max-train-samples 0 \
    --sample-every 1 \
    --sample-count 8
```

On the full split with `batch_size=64` and `drop_last=True`, each epoch has
2,543 optimizer steps. Twenty epochs therefore produce 50,860 updates.

The batch size and worker count above were selected for an M3 Max with 128 GB
of unified memory. They are not universal defaults. Reduce the batch size if
memory pressure occurs, and benchmark `num_workers` for the local storage and
CPU.

### EMA

The training model is updated by AdamW. A separate frozen model is maintained
for sampling using:

```text
ema_weight = beta * ema_weight + (1 - beta) * model_weight
```

Defaults:

```text
EMA beta:          0.9999
EMA warmup:        2,000 optimizer steps
```

During warmup, EMA weights exactly copy the training weights. After warmup,
they are exponentially averaged. Periodic training samples and standalone
sampling use the EMA model because it is generally more stable than the latest
training weights.

Override these settings with `--ema-beta` and `--ema-warmup-steps`.

### Resume training

`--epochs` always means the target total epoch count, not the number of epochs
to add. To continue an epoch-20 checkpoint through epoch 40:

```bash
python ddpm.py \
    --run-name DDPM-full-ema \
    --resume models/DDPM-full-ema/ckpt.pt \
    --epochs 40 \
    --batch-size 64 \
    --max-train-samples 0 \
    --num-workers 4
```

Resume restores the regular model, EMA model and step, AdamW state, completed
epoch, and TensorBoard global step. AdamW's saved parameter groups also restore
the checkpoint's learning rate; the current CLI does not override that restored
optimizer learning rate.

Keep dataset size, batch size, image size, and subset settings consistent when
resuming unless the change is intentional. Older checkpoints without EMA
weights initialize EMA from the restored training model.

## Monitoring

Start TensorBoard from the repository root:

```bash
tensorboard --logdir runs
```

Then open the local URL printed by TensorBoard, normally
`http://localhost:6006`.

The logged metric is per-batch noise-prediction MSE. It is expected to be
noisy because every image receives a newly sampled timestep and noise tensor.
A falling loss is useful, but visual samples remain essential: low aggregate
MSE does not guarantee that every reverse-diffusion step produces good images.

At the end of each epoch, the latest checkpoint is written to:

```text
models/<run-name>/ckpt.pt
```

Periodic sample grids are written to:

```text
results/<run-name>/<epoch>.jpg
```

## Generate images

`sample.py` loads EMA weights when present and falls back to regular model
weights for older checkpoints:

```bash
python sample.py \
    --checkpoint models/DDPM-full-ema/ckpt.pt \
    --output results/DDPM-full-ema/final_samples.png \
    --num-samples 16 \
    --seed 42
```

Use the same seed to compare checkpoints or sampling changes. The reverse
process is stochastic, so different seeds can vary noticeably in quality.

## Compare real and generated images

Create one PNG with a 4×4 grid of randomly selected real training images above
a 4×4 grid generated with EMA weights:

```bash
python compare_samples.py \
    --checkpoint models/DDPM-full-ema/ckpt.pt \
    --dataset-path data \
    --output results/DDPM-full-ema/real-vs-generated.png \
    --num-samples 16 \
    --columns 4 \
    --seed 2026
```

The real images use the training center crop and resize but omit random
flipping and normalization so they display in their natural RGB range.

## Checkpoint format

Current checkpoints are dictionaries with these keys:

| Key | Contents |
| --- | --- |
| `epoch` | Number of fully completed epochs |
| `model` | Latest training-model state dictionary |
| `optimizer` | AdamW state dictionary |
| `ema_model` | EMA model state dictionary |
| `ema_step` | Number of completed EMA updates |
| `loss` | Loss from the final batch of the saved epoch |
| `global_step` | Number of completed optimizer steps |

Checkpoints are loaded onto the CPU first for portability. Optimizer tensors are
moved to the selected training device after loading.

Architecture and diffusion hyperparameters are not embedded in the checkpoint.
The loading code must therefore use a compatible `UNet`, image size, timestep
count, and noise schedule.

## Known limitations and useful next experiments

- The model is unconditional: it cannot request identities, attributes, or
  prompts.
- Output resolution is 64×64, which limits fine facial and hair detail.
- Sampling performs 999 sequential UNet evaluations and is therefore slow.
- The schedule is linear and the reverse variance is fixed to `beta_t`.
- Evaluation is visual; FID, precision, recall, and validation loss are not yet
  implemented.
- Checkpoints restore training state but not Python, DataLoader, CPU, or device
  RNG state, so resume is continuous but not bit-for-bit deterministic.
- Only the latest checkpoint is retained for each run name.

Good next experiments include training longer, comparing fixed seeds across
checkpoints, adding a cosine noise schedule, implementing DDIM sampling,
measuring FID, increasing UNet capacity, and moving to 128×128 after the 64×64
baseline is well understood.
