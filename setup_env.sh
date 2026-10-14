#!/usr/bin/env bash
# Client-only venv for serverless RL (OpenPipe ART). No torch/vLLM: training and inference run remotely.
#
#   bash serverless/setup_env.sh
#
# Kept separate from the shared FaithfulEval venv on purpose: openpipe-art pins openai<3 and scipy<1.18, which
# would downgrade packages that venv relies on (openai 3.x, scipy 1.18).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
VENV="${SERVERLESS_VENV:-/scratch/project_465003400/SSLVLM/venvs/serverless-art}"
PIP_CACHE="${PIP_CACHE_DIR:-/scratch/project_465003400/cache/pip}"
# PyPI 0.5.20 cannot read the token ids the serverless endpoint now returns (Dynamo "nvext" engine data), and
# training fails with "Trainable Choice is missing vLLM prompt_token_ids/token_ids". Pin the GitHub commit that
# adds that support until a release includes it.
ART_SPEC="${ART_SPEC:-openpipe-art @ git+https://github.com/OpenPipe/ART.git@c7dda222015f728cf625fbb9719b2152dfd41d24}"

mkdir -p "$(dirname "$VENV")" "$PIP_CACHE"
module purge
module load Local-LAIF lumi-aif-singularity-bindings

singularity exec "$SIF" bash -s <<EOF
set -euo pipefail
unset PYTHONPATH
python3 -m venv "$VENV"
source "$VENV/bin/activate"
export PIP_CACHE_DIR="$PIP_CACHE"
pip install --upgrade pip
pip install "$ART_SPEC" "wandb==0.28.0" numpy pillow pyarrow pytest
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python - <<'PY'
import art
from art.serverless.backend import ServerlessBackend  # noqa: F401
import serverless.train  # noqa: F401
print("art", getattr(art, "__version__", "?"), "ok")
PY
EOF
