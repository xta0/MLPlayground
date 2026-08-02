# Serving vLLM on Apple Silicon Mac

This project uses the vLLM Metal plugin instead of the regular PyPI `vllm`
package. The regular package pulled in `xgrammar`, whose native
`apache-tvm-ffi` dependency segfaulted on Apple Silicon before vLLM could start.

The working setup is:

- Python 3.12 in the `vlm` conda environment
- `vllm==0.26.0+cpu`
- `vllm-metal`
- `xgrammar` and `apache-tvm-ffi` removed
- one local patch to keep vLLM from importing `xgrammar` for type annotations

## 1. Activate the environment

```bash
conda activate vlm
python --version
```

Use Python 3.12. Avoid Python 3.14 for this stack.

## 2. Remove the broken packages

```bash
pip uninstall -y vllm xgrammar apache-tvm-ffi
```

`xgrammar` was the direct source of the segfault:

```text
TVMFFIEnvRegisterCAPI
xgrammar::__TVMFFIStaticInitFunc0()
```

## 3. Install the macOS ARM CPU vLLM wheel

```bash
pip install --force-reinstall \
  "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl"
```

## 4. Install the latest vLLM Metal wheel

Run this inside the active `vlm` environment:

```bash
python - <<'PY'
import json
import subprocess
import sys
import urllib.request

release_url = "https://api.github.com/repos/vllm-project/vllm-metal/releases/latest"
release = json.load(urllib.request.urlopen(release_url))
wheel_url = next(
    asset["browser_download_url"]
    for asset in release["assets"]
    if asset["name"].endswith(".whl")
)

print(wheel_url)
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--force-reinstall",
    wheel_url,
])
PY
```

Check that the Metal plugin is installed:

```bash
python -m pip show vllm vllm-metal
```

Expected package shape:

```text
vllm:       0.26.0+cpu
vllm-metal: installed
```

## 5. Remove xgrammar again if pip brought it back

The `vllm` wheel declares `xgrammar` as a dependency, but basic serving does not
need it. If it is present, remove it:

```bash
pip uninstall -y xgrammar apache-tvm-ffi
```

Check:

```bash
python -m pip show xgrammar apache-tvm-ffi
```

Both should report `Package(s) not found`.

## 6. Patch vLLM so xgrammar remains lazy

Without this patch, vLLM imports `xgrammar` only to evaluate type annotations in
`backend_xgrammar.py`. With `xgrammar` uninstalled, that gives:

```text
ModuleNotFoundError: No module named 'xgrammar'
```

Patch the installed file:

```bash
python - <<'PY'
from pathlib import Path

p = Path(
    "INSTALL_PREFIX/lib/python3.12/site-packages/"
    "vllm/v1/structured_output/backend_xgrammar.py"
)

# Resolve the file inside the active environment.
import vllm
p = Path(vllm.__file__).parent / "v1/structured_output/backend_xgrammar.py"

s = p.read_text()
line = "from __future__ import annotations\n"
marker = "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"

if line not in s:
    s = s.replace(marker, marker + line, 1)
    p.write_text(s)
    print(f"patched {p}")
else:
    print(f"already patched {p}")
PY
```

Verify:

```bash
python - <<'PY'
from pathlib import Path
import vllm

p = Path(vllm.__file__).parent / "v1/structured_output/backend_xgrammar.py"
print("\n".join(p.read_text().splitlines()[:4]))
PY
```

The top of the file should include:

```python
from __future__ import annotations
```

This patch is local to the conda environment. Reinstalling or upgrading vLLM can
overwrite it.

## 7. Serve the model

Use `float16`, not `bfloat16`, on Mac:

```bash
vllm serve Qwen/Qwen3-0.6B --dtype=float16 --max-model-len 4096
```

On startup, vLLM should report the Metal plugin:

```text
Available plugins for group vllm.platform_plugins:
- metal -> vllm_metal:register
Platform plugin metal is activated
```

## Troubleshooting

If you see the original segfault again:

```bash
python -c "import xgrammar"
```

If that segfaults, remove the packages again:

```bash
pip uninstall -y xgrammar apache-tvm-ffi
```

If serving fails with `ModuleNotFoundError: No module named 'xgrammar'`, rerun
the patch in step 6.

If you see `No Metal device available`, run vLLM from a normal macOS terminal
session with access to the GPU. Headless, sandboxed, or virtualized sessions may
not expose Metal.

Structured outputs / grammar-guided decoding will not work while `xgrammar` is
uninstalled. This setup is intended for normal chat/completions serving.
