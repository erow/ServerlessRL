"""Defaults for serverless RL. Paper values follow SSL-R1 (arXiv:2604.20705) Sec. 4.1 and Fig. 5."""

from __future__ import annotations

import os
from pathlib import Path

from sslvlm.config import COCO_DIR, OUTPUT_ROOT

BASE_MODEL = os.environ.get("SERVERLESS_BASE_MODEL", "google/gemma-4-26B-A4B-it")
WANDB_PROJECT = os.environ.get("SERVERLESS_WANDB_PROJECT", "sslvlm-serverless")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
SERVERLESS_OUTPUT = Path(os.environ.get("SERVERLESS_OUTPUT", OUTPUT_ROOT / "serverless"))

# COCO train2017 is the training source (flat JPEG directory mounted from train2017.sqsh).
TRAIN_IMAGES = Path(os.environ.get("SERVERLESS_TRAIN_IMAGES", COCO_DIR))
# Held-out puzzle monitoring uses val2017 images read directly from the ZIP; never trained on.
VAL_IMAGES = Path(
    os.environ.get("SERVERLESS_VAL_IMAGES", "/scratch/project_465003400/datasets/coco/val2017.zip")
)

# SSL-R1 training details. "Global batch size 256" is not specified as prompts vs. responses;
# the default counts responses: 16 prompts x 16 responses = 256 trajectories per step.
GROUP_SIZE = 16
GROUPS_PER_STEP = 16
LEARNING_RATE = 1e-6
STEPS_MULTI_TASK = 2000
STEPS_SINGLE_TASK = 1000
TEMPERATURE = 1.0
# Gemma 4's recommended sampling; not in the paper. Set to None to sample from the full distribution.
TOP_P = 0.95
TOP_K = 64
KL_PENALTY = 0.0  # KL and entropy terms removed (SSL-R1 follows Visual Jigsaw)
MAX_COMPLETION_TOKENS = 2048  # not reported in the paper
SEED = 0

# Reward (Sec. 3.2.2): accuracy in [0, 1] plus 0.2 for strict <think>...</think> then \boxed{} format.
FORMAT_REWARD = 0.2

# Task setup (Sec. 4.1). "easy" values are the paper's ablation settings.
ROTATION_STEPS = 4  # S; easy/hard ablation uses 8 (45-degree steps)
INPAINT_RATIO = 0.5  # alpha; easy = 0.25
INPAINT_MIN_SIDE = 28
ORDER_GRID = 3  # M = N; easy = 2
CORR_POINTS = 6  # K; easy = 4
CORR_MIN_DIST = 40
SIM_CROP = 224
NUM_CHOICES = 4

# Images are sent inline as JPEG data URIs because the serverless backend cannot read local files.
JPEG_QUALITY = 95
MAX_IMAGE_SIDE = 640  # COCO long side is <= 640, so this is normally a no-op
MIN_IMAGE_SIDE = 112

# Gemma 4 native thinking (chat_template_kwargs.enable_thinking). The thinking arrives in message.reasoning and the
# final answer in message.content; training covers both because token ids include the thinking tokens.
ENABLE_THINKING = True

FORMAT_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    # "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)
