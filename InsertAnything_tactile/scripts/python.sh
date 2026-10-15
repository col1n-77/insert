#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_root="${INSERTANYTHING_ROOT:-$(dirname "$project_root")}"
python_path="${INSERTANYTHING_PYTHON:-$base_root/.venv-isaac51/bin/python}"
if [[ ! -x "$python_path" ]]; then
    printf 'Missing Isaac Lab Python: %s. Run the root setup script or set INSERTANYTHING_PYTHON.\n' "$python_path" >&2
    exit 1
fi
export INSERTANYTHING_ROOT="$base_root"
export PYTHONPATH="$project_root/source:$base_root/source/InsertAnything${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$project_root"
exec "$python_path" "$@"
