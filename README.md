# Serverless RL baseline: SSL-R1 on COCO with Gemma 4

This directory trains `google/gemma-4-26B-A4B-it` with
[CoreWeave Serverless RL](https://docs.coreweave.com/products/post-training/serverless-training/rl) through OpenPipe
ART's `ServerlessBackend`. The training objective is a composition of self-supervised visual tasks built from COCO
train2017. The baseline follows SSL-R1 (Xie et al., arXiv:2604.20705).

Rollouts, the LoRA update, and checkpoint hosting all run remotely. The local process is CPU-only: it generates
puzzles, sends chat requests, scores responses with verifiable rewards, and submits trajectory groups.

## Layout

| File | Role |
| --- | --- |
| `tasks/` | The five SSL-R1 puzzles. Each one turns a COCO image into interleaved prompt text and images plus a ground-truth answer. |
| `rewards.py` | Strict `\boxed{}` parsing, accuracy reward, and the 0.2 format reward. |
| `objective.py` | Composes tasks into one objective: mixture spec, deterministic stratified schedule, OpenAI messages. |
| `images.py` | Streamed COCO access (sqsh-mounted directory or ZIP) and JPEG data-URI encoding. |
| `train.py` | ART training loop: GRPO groups, held-out puzzle evaluation, local JSONL provenance. |
| `setup_env.sh`, `run.sh`, `submit.sh` | Client-only venv, container entry, CPU Slurm job. |

## Baseline fidelity

Taken directly from the paper:

- **Tasks and defaults:** rotation (S=4), visual similarity (SimCLR augmentations, 224 px crops, 3 negatives from
  other images), region inpainting (α=0.5, minimum side 28, black mask, 3 same-size distractor regions from the same
  image), patch ordering (3×3), and geometric correspondence (K=6 points at least 40 px apart, random affine, red
  numbered markers).
- **Prompts:** the task prompts are reproduced word for word from Fig. 5, with a format prompt appended to each.
- **Reward:** accuracy plus a 0.2 format bonus. Accuracy is binary for single answers and the fraction of correct
  positions for sequences.
- **Optimisation:** GRPO with group mean/std advantages (ART `loss_fn="ppo"`, `scale_rewards=True`), no KL or
  entropy term, 16 samples per prompt at temperature 1.0, learning rate 1e-6, 2,000 steps for multi-task runs, and
  tasks mixed in equal proportion.
- **Easier settings** (`--difficulty easy`): S=4, weak augmentation, α=0.25, a 2×2 grid, and K=4. The harder
  rotation setting is `--task-params '{"rotation": {"steps": 8}}'`.

Choices this implementation had to make. Report them alongside any results:

- **Format prompt (deliberate deviation):** Gemma 4 reasons without `<think>` tags, so the paper's sentence requiring
  them is commented out in `config.FORMAT_PROMPT`. The format check follows whatever prompt is configured. Without
  the `<think>` sentence, format credit needs reasoning text followed by exactly one `\boxed{}`; with it, the
  paper's strict `<think>…</think>` then `\boxed{}` rule applies. Each `run_config_*.json` records the prompt and
  rule used. On the step-0 held-out set, 98% of responses already met the no-think rule, against 13% for the
  `<think>` rule, so the format bonus is close to constant and accuracy drives learning.

- **Batch size:** the paper's "global batch size of 256" does not say whether it counts prompts or responses. The
  default is 16 prompts × 16 responses = 256 trajectories per step. For the prompt interpretation, use
  `--groups-per-step 256`, which costs 16 times more per step.
- **Model and adapter:** the paper fine-tunes Qwen2.5-VL. Here the model is Gemma 4 26B-A4B with a **rank-1 LoRA**,
  which the service fixes and does not let you change. The paper's 1e-6 learning rate was set for a different
  parameterisation and may be too small for a rank-1 adapter. Check reward curves in a pilot before running the full
  budget.
- **Unreported settings:** the maximum completion length (2,048 tokens by default), image resolution (COCO's native
  size, long side ≤ 640), and the affine ranges for correspondence (rotation ±45°, shear ±15°, scale 0.7–1.0,
  shift ±10%) are not in the paper.
- **Answer parsing:** only the last `\boxed{}` is read. A choice may be written `C` or `(C)`. An angle may carry a
  degree sign. A sequence must contain exactly K or M·N integers. Nothing else is fuzzy-matched.
- **Data generation:** puzzles are generated on the fly instead of from a fixed 591K dump. Step *s*, slot *i*
  always yields the same puzzle for a given `--seed`, so a resumed run sees the same data.

## The task-composition objective

`--tasks` sets the objective. It accepts a preset or a weighted list:

```text
ssl-r1                     all five tasks, equal weights (default)
cor, cor+ord, cor+ord+rot, cor+ord+rot+sim   Table 2 incremental mixtures
rotation:2,ordering:1      custom weights; abbreviations cor/ord/rot/sim/inp are accepted
```

Every task shares one reward definition. Prompts are assigned to tasks by a deterministic interleaved schedule, so
each step stays close to the target proportions. To add a task, implement `generate(rng, source) -> Puzzle` with
`kind` set to `choice`, `integer`, or `sequence`, then register it in `tasks/__init__.py`.

## Masked-view caption game (experimental)

`caption_game.py` is a label-free self-play objective, separate from the SSL-R1 puzzles. For each image the same
policy plays three roles:

| Role | Sees | Reward |
| --- | --- | --- |
| Captioner | The full image | Share of judge samples that picked its caption as the real one; 0 if it hedges or falls outside the word window |
| Forger | The image with a salient region greyed out; told to describe the whole scene | Share of judge samples it fooled; 0 if it mentions the mask, hedges, or its length is outside 0.6–1.6× the paired real caption or the captioner's word window |
| Judge | The full image and one (real, forged) pair in random order | 1 for naming the full-image caption, plus the 0.2 format bonus |

The label comes from which view wrote the caption, so no annotation is used. The masked region is the
highest-edge-energy box among 8 random candidates. Its size starts at `--mask-ratio` and is steered so judge
accuracy stays near `--judge-target` (turn off with `--fixed-mask`). Near-identical pairs (word Jaccard above
0.9) are not judged. Each image gives one captioner group and one forger group of `--caption-samples`, and one judge
group of `--judge-samples` per judged pair.

```bash
SERVERLESS_ENTRY=serverless.caption_game sbatch serverless/submit.sh --name capgame-smoke --dry-run
# Pilot 1: judge only, captions from the frozen base checkpoint
SERVERLESS_ENTRY=serverless.caption_game sbatch serverless/submit.sh --name capgame-judge \
    --train-roles judge --freeze-captions --steps 200 --val-every 20
# Pilot 2: full three-role game
SERVERLESS_ENTRY=serverless.caption_game sbatch serverless/submit.sh --name capgame-full --steps 500 --val-every 20
```

Outputs mirror the puzzle runs: `train_metrics.jsonl` (judge accuracy, `judge/picked_a` position bias, forger gate
pass and leak rates, caption lengths, mask ratio, zero-variance group share), `val_metrics.jsonl` and
`val_games_stepNNNNN.jsonl` (greedy captions, forgeries and judgements on fixed val2017 images at the initial mask
ratio), and `mask_ratio.jsonl` for resume. Downstream checks (POPE, AMBER, MMVP, CHAIR) are not wired in here.

## Setup

1. Create a CoreWeave Forge API key. Export it as `WANDB_API_KEY` and keep it out of the repository.
2. Build the client venv once:

   ```bash
   bash serverless/setup_env.sh   # -> /scratch/project_465003400/SSLVLM/venvs/serverless-art
   ```

   This venv is separate from the shared FaithfulEval venv on purpose. `openpipe-art` pins `openai<3` and
   `scipy<1.18`, and installing it there would downgrade both. It contains no torch or vLLM.

## Run

CPU check, with no API calls. It writes the generated puzzle images and prompts under `<output>/dry_run/`:

```bash
sbatch serverless/submit.sh --name smoke --dry-run
```

Pilot first, then the full baseline. Resubmit with the same `--name` to resume from the latest checkpoint; the CPU
partition caps a job at 72 hours.

```bash
sbatch serverless/submit.sh --name ssl-r1-gemma4-pilot --steps 50 --val-every 10
sbatch serverless/submit.sh --name ssl-r1-gemma4-run1
```

`submit.sh` asks for a CPU node only. The job needs outbound HTTPS to `forge.coreweave.com` and W&B.

## Outputs

`$SERVERLESS_OUTPUT/<name>/` defaults to `/scratch/project_465003400/SSLVLM/outputs/serverless/<name>/`:

- `run_config_*.json`: CLI arguments, resolved task parameters, and seed.
- `model.json`: serverless model id, W&B entity and project, and base model.
- `train_metrics.jsonl`: reward, accuracy, format, and parse rate for each step, overall and per task. Also the
  share of zero-variance groups, timings, and the checkpoint artifact.
- `val_metrics.jsonl` and `val_responses_stepNNNNN.jsonl`: greedy accuracy on a fixed held-out puzzle set built
  from COCO **val2017**, with raw responses.

W&B logging goes to the run ART creates, named after `--name`, in project `--project`:

- `train/*`, `val/*`, `loss/*`: ART's metrics. `puzzle_train/*` and `puzzle_val/*`: per-task reward, accuracy,
  format, and parse rates.
Rollouts are traced in **Weave** (`https://wandb.ai/<entity>/<project>/weave`), one call per rollout:

- **Ops:** `rollout_train` covers training rollouts. `--weave-train-sample-rate` sets the fraction traced (default
  1.0). `rollout_val` covers held-out rollouts, which are always traced. Calls are named `<task> <puzzle_id>`; the
  id is `train-<step>-<slot>` or `val-0-<slot>`.
- **Inputs:** split, task, puzzle_id, answer, prompt text with `<image>` placeholders, the puzzle images in prompt
  order, temperature, and puzzle metadata (source image, boxes, permutation, affine).
- **Output:** raw response, reward, accuracy, format, parsed, truncated, completion tokens, and the checkpoint step
  that generated it.
- **Images:** ART's managed OpenAI client isn't auto-patched by Weave, so the op logs the images itself.
- **Turning it off:** `--no-weave`.

Rollouts call the explicit checkpoint `…/<name>:step<N>`, because serverless LoRA inference rejects unversioned
names. Checkpoints are W&B artifacts. Serverless Inference serves them as
`wandb-artifact:///<entity>/<project>/<name>:step<N>`.

## Interpretation and limitations

- Puzzle accuracy measures only the self-supervised tasks. Per `AGENTS.md`, choose checkpoints with MMVP pair
  accuracy and keep the other twelve benchmarks until the checkpoint is fixed. `sslvlm.evaluate_benchmarks` runs
  models locally, so a serverless Gemma 4 checkpoint must first be downloaded from its W&B artifact or evaluated
  through the hosted endpoint. Neither path is implemented here.
- Each trajectory carries its images as base64 data URIs, about 70 MB of request payload per default step.
- **Inference concurrency:** on 2026-10-14 the `erow` account was limited to 20 simultaneous Gemma 4 requests per
  user; further requests got HTTP 429. The documented default is 2,000. `--max-concurrency` defaults to 16, and
  429, 5xx, and timeout errors are retried with backoff (`--max-retries`). Each step's retry count is logged as
  `request_retries`. To run faster, ask forge-support@coreweave.com for a higher limit, then raise
  `--max-concurrency`.
- The training API's handling of multi-image trajectories for Gemma 4 has not been checked against the live service
  from this repository. Run the pilot and inspect the W&B traces before a long run.
