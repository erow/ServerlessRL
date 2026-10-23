#!/usr/bin/env bash
# Run serverless training inside the container with the client-only ART venv (see serverless/setup_env.sh).
# Called through scripts/submit.sh, which enters the container and mounts COCO at /data/coco:
#
#   sbatch serverless/submit.sh --name ssl-r1-gemma4-run1
set -euo pipefail
VENV="${SERVERLESS_VENV:-/scratch/project_465003400/SSLVLM/venvs/serverless-art}"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "missing $VENV; run: bash serverless/setup_env.sh" >&2
  exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
  # Fall back to the W&B key stored by `wandb login`; ART only reads the environment variable.
  WANDB_API_KEY="$(awk '{for (i = 1; i < NF; i++) { if ($i == "machine") m = $(i + 1); if ($i == "password" && m == "api.wandb.ai") { print $(i + 1); exit } }}' "$HOME/.netrc")"
  export WANDB_API_KEY
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set (CoreWeave Forge / W&B API key)" >&2
  exit 2
fi
# The shared venv and container site-packages carry an incompatible openai; use only the ART venv.
unset PYTHONPATH
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Remote endpoints need network access; the offline HF flags from submit.sh are irrelevant here.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" TOKENIZERS_PARALLELISM=false
# SERVERLESS_ENTRY=serverless.caption_game runs the masked-view caption game instead of the SSL-R1 puzzles.
exec python -m "${SERVERLESS_ENTRY:-serverless.train}" "$@"
