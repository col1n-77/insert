#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${INSERTANYTHING_PYTHON:-python3}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
cd "$project_root"
exec "$python_path" -m pytest "$@"
