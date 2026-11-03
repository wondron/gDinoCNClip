#!/usr/bin/env bash
# Linux entry point. ORT 1.19 does not search the pip-installed NVIDIA libraries, so put them on
# LD_LIBRARY_PATH (same command as the project README), then run benchmark.py with these arguments.
# Choose the interpreter with PYTHON=/path/to/python bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
NVIDIA_LIB_DIRS=$("$PYTHON" -c 'import site; from pathlib import Path; roots = site.getsitepackages() + [site.getusersitepackages()]; print(":".join(sorted({str(p) for root in roots for p in (Path(root) / "nvidia").glob("*/lib") if p.is_dir()})))')
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PYTHON" benchmark.py "$@"
