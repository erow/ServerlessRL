# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CPU-only client that trains `google/gemma-4-26B-A4B-it` with CoreWeave Serverless RL via OpenPipe ART's
`ServerlessBackend`, reproducing the SSL-R1 baseline (arXiv:2604.20705) on COCO. Rollouts, LoRA updates and
checkpoint hosting are remote; locally we generate puzzles, call the chat endpoint, score with verifiable rewards,
and submit trajectory groups. See `README.md` for paper-fidelity choices that must be reported with results.

**Packaging:** this repo is the `serverless/` subpackage of a larger SSLVLM project. Code imports `serverless.*`
and `sslvlm.config` (`COCO_DIR`, `OUTPUT_ROOT`), and scripts assume `PYTHONPATH` is the parent project root. It
won't import standalone without that parent checkout.

## Commands

All are run from the parent project root on LUMI (Slurm + Singularity):

```bash
bash serverless/setup_env.sh                              # build client venv (ART pinned to a git commit; no torch/vLLM)
sbatch serverless/submit.sh --name smoke --dry-run        # one step of puzzles/messages to <output>/dry_run/, no API calls
sbatch serverless/submit.sh --name X --steps 50 --val-every 10   # pilot
sbatch serverless/submit.sh --name X                      # full run; same --name resumes from latest checkpoint
```

`run.sh` is the in-container entry (`python -m serverless.train`); it needs `WANDB_API_KEY` (falls back to
`~/.netrc`) and deliberately unsets `PYTHONPATH` before activating the venv because the shared venv's `openai` is
incompatible. Keep this venv separate: `openpipe-art` pins `openai<3`, `scipy<1.18`. CLI flags live in
`train.py`'s argparse; defaults in `config.py` (env-overridable via `SERVERLESS_*`). No test suite is checked in
(pytest is installed in the venv).

## Architecture

- `tasks/`: each SSL task exposes `generate(rng, source) -> Puzzle` (`tasks/base.py`) with `kind` in
  `choice | integer | sequence`; register new tasks in `tasks/__init__.py`.
- `rewards.py`: one shared reward for all tasks — strict parsing of the *last* `\boxed{}`, accuracy (binary, or
  fraction of correct positions for sequences) + 0.2 format bonus. The format rule depends on
  `config.FORMAT_PROMPT` (the `<think>` sentence is intentionally commented out for Gemma 4).
- `objective.py`: parses `--tasks` (presets like `ssl-r1`, `cor+ord`, or `rotation:2,ordering:1`) and builds a
  deterministic interleaved schedule; step *s*, slot *i* always yields the same puzzle for a given seed, which
  is what makes resume reproducible. Don't introduce unseeded randomness.
- `images.py`: COCO access (sqsh-mounted dir for train2017, ZIP for val2017) and JPEG data-URI encoding.
- `train.py`: ART loop — GRPO groups (`loss_fn="ppo"`, `scale_rewards=True`), held-out val eval on val2017,
  JSONL provenance (`run_config_*.json`, `train_metrics.jsonl`, `val_*.jsonl`) under `$SERVERLESS_OUTPUT/<name>/`.
- `tracing.py`: Weave ops `rollout_train`/`rollout_val`; logs images manually since ART's client isn't auto-patched.

Gotchas: rollouts must target the versioned checkpoint `<name>:step<N>` (unversioned names are rejected);
the account is rate-limited (~20 concurrent requests), so `--max-concurrency` defaults to 16 with retry/backoff
on 429/5xx.
