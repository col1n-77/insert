#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${INSERTANYTHING_PYTHON:-$project_root/.venv-isaac51/bin/python}"
if [[ ! -x "$python_path" ]]; then
    printf 'Run scripts/setup_local_env.sh or set INSERTANYTHING_PYTHON to an Isaac Lab Python executable.\n' >&2
    exit 1
fi
export PYTHONPATH="$project_root/source/InsertAnything${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$project_root"
exec "$python_path" "$@"
