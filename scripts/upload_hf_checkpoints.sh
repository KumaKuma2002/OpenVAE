#!/usr/bin/env bash
# Upload local OpenVAE-related checkpoints to SMILE-project/OpenVAE.
# Requires: hf auth login (valid token with repo write + gated model access accepted)
set -euo pipefail
REPO="SMILE-project/OpenVAE"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

hf upload "$REPO" "$ROOT/ckpt/MAISI/maisi_autoencoder.pt" "MAISI/maisi_autoencoder.pt" \
  --commit-message "Add MAISI pretrained autoencoder (MONAI MAISI)"

hf upload "$REPO" "$ROOT/ckpt/OpenVAE-3D-4x-patch64-10K/autoencoder_best.pt" \
  "OpenVAE-3D-4x-patch64-10K/autoencoder_best.pt" \
  --commit-message "Add OpenVAE-3D-4x-patch64-10K autoencoder_best"

hf upload "$REPO" "$ROOT/ckpt/OpenVAE-3D-4x-patch64-10K/autoencoder_latest.pt" \
  "OpenVAE-3D-4x-patch64-10K/autoencoder_latest.pt" \
  --commit-message "Add OpenVAE-3D-4x-patch64-10K autoencoder_latest"

echo "Done. Optionally: hf upload $REPO $ROOT/README.md --path-in-repo README.md"
