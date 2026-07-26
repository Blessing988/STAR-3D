#!/bin/bash
set -euo pipefail

source /mmfs1/apps/pyenvs/anaconda3-2022.05/bin/activate base || true

for env in \
  /path/to/conda_envs/aicity \
  /path/to/conda_envs/physicalai_track1_detector \
  /path/to/conda_envs/rfdetr \
  /path/to/conda_envs/dfine
do
  echo "ENV=$env"
  source activate "$env" || true
  python - <<'PY'
import importlib.util
mods = ["torch", "MinkowskiEngine", "mmcv", "plyfile", "pointnet2", "yaml", "scipy", "sklearn"]
for mod in mods:
    print(f"{mod}={bool(importlib.util.find_spec(mod))}")
try:
    import torch
    print("torch_version", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("torch_cuda_available", torch.cuda.is_available())
except Exception as exc:
    print("torch_error", repr(exc))
PY
done
