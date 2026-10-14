#!/bin/bash -l
# CPU-only Slurm job for serverless RL: rollouts and training run on CoreWeave, so no GPUs are requested.
#   sbatch serverless/submit.sh --name ssl-r1-gemma4-run1 [train.py options]
# Resubmitting with the same --name resumes from the latest serverless checkpoint.
#SBATCH --account=project_465003400
#SBATCH --partition=small
#SBATCH --job-name=sslvlm-serverless
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/project_465003400/SSLVLM/outputs/slurm/%x-%j.out
#SBATCH --error=/scratch/project_465003400/SSLVLM/outputs/slurm/%x-%j.err
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec bash "$PROJECT_DIR/scripts/submit.sh" -- bash "$PROJECT_DIR/serverless/run.sh" "$@"
