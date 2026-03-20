#!/usr/bin/env bash
# Direct metrics from existing GT + recon NIfTIs only (no inference). CPU OK.
#
#   bash direct_compute_metrics.sh
#   bash direct_compute_metrics.sh -- --skip-lpips

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python direct_compute_metrics.py --benchmark_root ../outputs/vae_benchmark "$@"
