#!/usr/bin/env bash
# Reuse the machine's Isaac Sim stack without changing its Conda environment.
set -euo pipefail
export PYTHONNOUSERSITE=1

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_python="${INSERTANYTHING_BASE_PYTHON:-$(command -v python3 || true)}"
venv_path="$project_root/.venv-isaac51"

if [[ ! -x "$base_python" ]]; then
    printf 'Missing Isaac Lab Python: %s; set INSERTANYTHING_BASE_PYTHON\n' "$base_python" >&2
    exit 1
fi
"$base_python" - <<'PY'
import sys
from importlib.metadata import version

if sys.version_info[:2] != (3, 11):
    raise SystemExit("InsertAnything requires Python 3.11.")
for name in ("isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_rl", "rl_games", "torch", "toml", "wheel"):
    print(f"{name}: {version(name)}")
PY

if [[ ! -f "$venv_path/pyvenv.cfg" ]]; then
    "$base_python" -m venv --system-site-packages "$venv_path"
fi
"$venv_path/bin/python" - "$base_python" <<'PY'
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 11):
    raise SystemExit("Existing .venv does not use Python 3.11; preserve it and choose another environment.")
config = Path(sys.prefix, "pyvenv.cfg").read_text()
if "include-system-site-packages = true" not in config:
    raise SystemExit("Existing .venv cannot reuse Isaac Sim; preserve it and choose another environment.")
if Path(sys._base_executable).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit("Existing project environment uses a different base Python; preserving it.")
PY
"$venv_path/bin/python" -m pip install --no-deps --no-build-isolation --no-cache-dir \
    -e "$project_root/source/InsertAnything"
"$venv_path/bin/python" -m pip --no-cache-dir check
printf '\nReady: %s/scripts/python.sh scripts/list_envs.py\n' "$project_root"
