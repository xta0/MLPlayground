# Image to text

Train a Transformer to generate captions for Flickr8k images.

## Dataset

Run both commands from the project directory with [uv](https://docs.astral.sh/uv/) installed. Place the caption annotations at `data/flicker8k/dataset_flickr8k.json` and the JPEG files in `data/flicker8k/Images/`. The directory name `flicker8k` matches the path in the code.

## Training

```sh
uv run -m image2text.train
```

Training starts from fresh weights and runs for 50 epochs by default. After each epoch it evaluates Flickr8k's validation images and logs `loss/train` and `loss/val` to TensorBoard under `runs/`. Both losses are averaged over non-padding target tokens. The epoch with the lowest validation loss so far in the run supplies `checkpoints/best.pth`, so you can stop after validation loss has stopped improving.

### TensorBoard

Start the dashboard from the project directory in a separate terminal.

```sh
uv run tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

Open [TensorBoard](http://127.0.0.1:6006/), select your run, and look for `loss/train` and `loss/val` under **Scalars**. Each point represents one completed epoch.

## Inference

```sh
uv run -m image2text.main --split train
```

Training images are the default. Use `--split test` to sample held-out images. Inference loads `checkpoints/best.pth` when available and falls back to the older `checkpoints/model.pth` with its original vocabulary. It writes eight predicted and ground-truth caption comparisons to `results/`. Each run replaces the previous batch. The loader samples image–caption pairs, so the same image can appear with different reference captions.

## Architecture

- **Vision encoder** splits a 128 × 128 RGB image into 256 non-overlapping 8 × 8 patches. It projects each patch to a 192-dimensional embedding, adds a learned positional embedding, and processes the sequence with six Transformer blocks using eight attention heads each.
- **Caption decoder** embeds caption tokens and adds sinusoidal positional embeddings. Its six Transformer blocks use eight-head causal self-attention over the caption and cross-attention over the encoded image patches. A linear layer produces a score for each vocabulary token at each caption position.
- **Training and generation** both rely on next-token prediction. During training, the decoder receives reference tokens and predicts the following token at each position. Cross-entropy ignores padding targets. During inference, the decoder starts with `<start>` and samples one token at a time until it emits `<end>` or reaches 50 steps.
