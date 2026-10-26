# ViT experiments

Vision transformer experiments with CIFAR-10. Uses Python 3.12 and uv.

Install the locked dependencies:

```bash
uv sync --locked
```

Run the existing CIFAR-10 download script:

```bash
uv run main.py
```

The first run downloads the dataset into `data/`.

Add dependencies with `uv add <package>`. Commit `uv.lock` to keep installs reproducible.
