#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON_BIN:-python}" "${ROOT_DIR}/scripts/infer_examples.py" "$@"
