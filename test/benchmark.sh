#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Single CT:  bash benchmark.sh /path/to/scan.nii.gz
#   All in dir: bash benchmark.sh /path/to/ct_dir

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

INPUT="${1:-/path/to/ct_dir}"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/vae_benchmark}"

PATCH_SIZE=(64 64 64)
OVERLAP=(16 16 16)
DEVICE="cuda"
AMP_FLAG="--amp"

cd "$SCRIPT_DIR"

if [[ -f "$INPUT" ]] && { [[ "$INPUT" == *.nii.gz ]] || [[ "$INPUT" == *.nii ]]; }; then
  INPUT_FLAG=(--input_ct "$INPUT")
  echo "[benchmark.sh] Single CT mode: $INPUT"
else
  INPUT_FLAG=(--input_ct_dir "$INPUT")
  echo "[benchmark.sh] Directory mode (all NIfTIs): $INPUT"
fi

for model_dir in "$CKPT_ROOT"/*; do
  if [[ ! -d "$model_dir" ]]; then
    continue
  fi

  model_name="$(basename "$model_dir")"

  if [[ "$model_name" == "segmenter" ]]; then
    continue
  fi

  if [[ -f "$model_dir/vae/config.json" || -f "$model_dir/autoencoder_best.pt" ]]; then
    echo "[benchmark.sh] Running model: $model_name"
    python benchmark_vae.py \
      "${INPUT_FLAG[@]}" \
      --model_dir "$model_dir" \
      --output_root "$OUTPUT_ROOT" \
      --patch_size "${PATCH_SIZE[@]}" \
      --overlap "${OVERLAP[@]}" \
      --num_splits 1 \
      --device "$DEVICE" \
      $AMP_FLAG
  else
    echo "[benchmark.sh] Skipping non-VAE folder: $model_name"
  fi
done

echo "[benchmark.sh] Done."
