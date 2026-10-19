"""Serverless GRPO on SSL-R1 puzzles from COCO, via OpenPipe ART's ServerlessBackend.

    WANDB_API_KEY=... python -m serverless.train --name ssl-r1-gemma4-run1
    python -m serverless.train --name smoke --dry-run      # CPU only, no API calls

Rollouts are generated through the serverless OpenAI-compatible endpoint, scored locally with the verifiable
reward, and sent as trajectory groups to a LoRA training step. Each step's puzzles derive from (seed, step), so a
resumed run sees the same data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from serverless import config as C
from serverless import tracing
from serverless.dynamic import DynamicDifficulty
from serverless.images import ImageSource
from serverless.objective import PRESETS, Objective, puzzle_messages
from serverless.rewards import Score
from serverless.tasks import Puzzle
MATCH_ANSWER = re.compile(r"\\boxed{(.*)}")

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="model/run name; reusing it resumes from the latest checkpoint")
    p.add_argument("--project", default=C.WANDB_PROJECT)
    p.add_argument("--entity", default=C.WANDB_ENTITY)
    p.add_argument("--base-model", default=C.BASE_MODEL)
    p.add_argument("--tasks", default="ssl-r1", help=f"preset {sorted(PRESETS)} or 'task[:weight],...'")
    p.add_argument("--difficulty", choices=["paper", "easy"], default="paper")
    p.add_argument("--task-params", default=None, help='JSON overrides, e.g. \'{"rotation": {"steps": 8}}\'')
    p.add_argument("--dynamic-difficulty", action="store_true",
                   help="sample task t with p_t ∝ weight_t / acc_t (EMA of training accuracy); see serverless.dynamic")
    p.add_argument("--dynamic-momentum", type=float, default=0.8, help="EMA weight on the previous accuracy estimate")
    p.add_argument("--dynamic-min-accuracy", type=float, default=0.05, help="accuracy floor in the 1/acc weighting")
    p.add_argument("--dynamic-init", type=Path, default=None,
                   help="initial accuracies from a serverless.difficulty summary.json (default: 0.5 for every task)")
    p.add_argument("--steps", type=int, default=C.STEPS_MULTI_TASK)
    p.add_argument("--group-size", type=int, default=C.GROUP_SIZE, help="responses per prompt")
    p.add_argument("--groups-per-step", type=int, default=C.GROUPS_PER_STEP, help="prompts per step")
    p.add_argument("--learning-rate", type=float, default=C.LEARNING_RATE)
    p.add_argument("--kl-penalty", type=float, default=C.KL_PENALTY)
    p.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    p.add_argument("--max-completion-tokens", type=int, default=C.MAX_COMPLETION_TOKENS,
                   help="includes thinking tokens when --enable-thinking is on")
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=C.ENABLE_THINKING,
                   help="Gemma 4 native thinking via chat_template_kwargs.enable_thinking")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                   help="stream completions (non-streaming requests are cut off by the proxy after 120 s)")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--max-concurrency", type=int, default=16,
                   help="simultaneous inference requests (this account's per-user limit measured at 20)")
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--max-retries", type=int, default=8, help="per-request retries on 429/5xx/timeouts, with backoff")
    p.add_argument("--max-exceptions", type=float, default=0.05, help="tolerated failed rollouts (fraction or count)")
    p.add_argument("--val-every", type=int, default=100, help="held-out puzzle eval interval in steps; 0 disables")
    p.add_argument("--val-per-task", type=int, default=20)
    p.add_argument("--no-weave", action="store_true", help="disable Weave rollout traces")
    p.add_argument("--weave-train-sample-rate", type=float, default=1.0,
                   help="fraction of training rollouts traced in Weave (held-out rollouts are always traced)")
    p.add_argument("--train-images", type=Path, default=C.TRAIN_IMAGES)
    p.add_argument("--val-images", type=Path, default=C.VAL_IMAGES)
    p.add_argument("--output-dir", type=Path, default=None, help="default: $SERVERLESS_OUTPUT/<name>")
    p.add_argument("--dry-run", action="store_true", help="build one step of puzzles and messages; no API calls")
    return p


def summarize(puzzles: list[Puzzle], scores: list[list[Score]]) -> dict[str, float]:
    """Per-task means of reward/accuracy/format plus the share of groups with no reward variance."""
    by_task: dict[str, list[Score]] = defaultdict(list)
    flat = []
    degenerate = 0
    for puzzle, group in zip(puzzles, scores):
        by_task[puzzle.task].extend(group)
        flat.extend(group)
        degenerate += len(group) > 0 and len({s.reward for s in group}) == 1
    out: dict[str, float] = {}

    def add(prefix: str, rows: list[Score]) -> None:
        if rows:
            out[f"{prefix}reward"] = statistics.fmean(s.reward for s in rows)
            out[f"{prefix}accuracy"] = statistics.fmean(s.accuracy for s in rows)
            out[f"{prefix}format"] = statistics.fmean(s.format for s in rows)
            out[f"{prefix}parsed"] = statistics.fmean(s.parsed for s in rows)

    add("", flat)
    for task, rows in sorted(by_task.items()):
        add(f"{task}/", rows)
    out["n_trajectories"] = len(flat)
    out["degenerate_group_frac"] = degenerate / max(1, len(puzzles))
    return out


def message_reasoning(message: Any) -> str | None:
    """Thinking text from the separate reasoning channel ("reasoning" or vLLM's older "reasoning_content")."""
    extra = getattr(message, "model_extra", None) or {}
    value = getattr(message, "reasoning", None) or extra.get("reasoning") or extra.get("reasoning_content")
    if isinstance(value, str):
        return value
    else:
        # or extract from <think> </think> tags
        content = message.content
        match = re.search(r"<think>(.*)</think>", content)
        if match:
            return match.group(1)
        else:
            return None


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def make_dynamic(args: argparse.Namespace, objective: Objective) -> DynamicDifficulty | None:
    if not args.dynamic_difficulty:
        return None
    kwargs = dict(momentum=args.dynamic_momentum, min_accuracy=args.dynamic_min_accuracy)
    if args.dynamic_init is not None:
        return DynamicDifficulty.from_summary(objective.weights, args.dynamic_init, **kwargs)
    return DynamicDifficulty(objective.weights, **kwargs)


def dry_run(args: argparse.Namespace, objective: Objective, out_dir: Path) -> None:
    source = ImageSource(args.train_images)
    dynamic = make_dynamic(args, objective)
    if dynamic is not None:
        print("dynamic difficulty: " + json.dumps(dynamic.state()))
    t0 = time.perf_counter()
    puzzles = objective.batch(source, "train", 0, args.groups_per_step, dynamic.weights() if dynamic else None)
    gen_s = time.perf_counter() - t0
    preview = out_dir / "dry_run"
    preview.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for i, puzzle in enumerate(puzzles):
        messages = puzzle_messages(puzzle)
        size = len(json.dumps(messages))
        total_bytes += size
        for j, img in enumerate(puzzle.images):
            img.save(preview / f"{i:02d}_{puzzle.task}_{j}.jpg", quality=90)
        append_jsonl(preview / "puzzles.jsonl", {"i": i, "task": puzzle.task, "answer": puzzle.answer,
                                                 "text": puzzle.text, "message_bytes": size, "meta": puzzle.meta})
        print(f"[{i:02d}] {puzzle.task:15s} images={len(puzzle.images)} answer={puzzle.answer!r} bytes={size}")
    print(f"generated {len(puzzles)} puzzles in {gen_s:.2f}s; ~{total_bytes * args.group_size / 1e6:.1f} MB of "
          f"messages per step at group size {args.group_size}; preview in {preview}")


async def train(args: argparse.Namespace, objective: Objective, out_dir: Path) -> None:
    import art
    from art.serverless.backend import ServerlessBackend

    train_source = ImageSource(args.train_images)
    val_source = ImageSource(args.val_images) if args.val_every > 0 else None
    backend = ServerlessBackend()
    model = art.TrainableModel(
        name=args.name, run_name=args.name, project=args.project, entity=args.entity, base_model=args.base_model,
        base_path=str(C.SERVERLESS_OUTPUT / "art"),  # ART's local history/trajectories; default is <repo>/.art
    )
    await model.register(backend)
    client = model.openai_client()
    semaphore = asyncio.Semaphore(args.max_concurrency)
    retry_counts: Counter[str] = Counter()
    write_json(out_dir / "model.json", {"id": model.id, "entity": model.entity, "project": model.project,
                                        "name": args.name, "base_model": args.base_model,
                                        "art_version": getattr(art, "__version__", None)})

    if not args.no_weave:
        tracing.init(model.entity, args.project)
    # Serverless LoRA inference only accepts explicit checkpoints (":stepN"), never "latest".
    policy = {"step": await model.get_step()}

    async def complete(messages: list[dict], temperature: float) -> Any:
        """Chat completion with backoff on rate limits (429), server errors, timeouts, and dropped connections."""
        import openai
        from art.openai import consume_chat_completion_stream

        for attempt in range(args.max_retries + 1):
            try:
                async with semaphore:
                    request = dict(
                        model=model.get_inference_name(policy["step"]),
                        messages=messages,
                        max_completion_tokens=args.max_completion_tokens,
                        temperature=temperature,
                        logprobs=True,
                        timeout=args.request_timeout,
                        extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
                    )
                    if not args.stream:
                        return await client.chat.completions.create(**request)
                    # The endpoint sits behind a proxy that drops non-streaming responses after 120 s (HTTP 524);
                    # streaming keeps long thinking alive. ART rebuilds the completion, token ids included.
                    stream = await client.chat.completions.create(
                        **request, stream=True, stream_options={"include_usage": True},
                    )
                    return await consume_chat_completion_stream(stream)
            except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError,
                    openai.APIConnectionError) as exc:
                if attempt == args.max_retries:
                    raise
                retry_counts[type(exc).__name__] += 1
                await asyncio.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")

    async def rollout(
        puzzle: Puzzle, messages: list[dict], temperature: float, sink: list[Score], split: str = "train",
    ) -> art.Trajectory:
        completion = await complete(messages, temperature)
        choice = completion.choices[0]
        text = choice.message.content or ""
        reasoning = message_reasoning(choice.message)
        
        answer = MATCH_ANSWER.search(text).group(1) if MATCH_ANSWER.search(text) else None
        result = objective.score(puzzle, text, reasoning if args.enable_thinking else None)
        sink.append(result)
        usage = completion.usage
        traj = art.Trajectory(
            messages_and_choices=[*messages, choice],
            reward=result.reward,
            metrics={
                "accuracy": result.accuracy,
                "format": result.format,
                "parsed": result.parsed,
                "truncated": choice.finish_reason == "length",
                "completion_tokens": usage.completion_tokens if usage else 0,
            },
            # metrics must be numeric (ART validates them), so the parsed answer string goes in metadata
            metadata={"task": puzzle.task, "puzzle_id": puzzle.meta["puzzle_id"], "answer": puzzle.answer,
                      "predicted": answer,
                      "response": text, "reasoning": reasoning, "policy_step": policy["step"]},
        )
        return traj.finish()

    rollout_train, rollout_val = rollout, rollout
    if not args.no_weave:
        rollout_train = tracing.traced(rollout, "train", args.weave_train_sample_rate)
        rollout_val = tracing.traced(rollout, "val")

    def make_group(puzzle: Puzzle, sink: list[Score]) -> art.TrajectoryGroup:
        messages = puzzle_messages(puzzle)  # encoded once, shared by the group's rollouts
        return art.TrajectoryGroup(
            [rollout_train(puzzle, messages, args.temperature, sink, split="train")
             for _ in range(args.group_size)],
            metadata={"task": puzzle.task, "puzzle_id": puzzle.meta["puzzle_id"]},
        )

    async def evaluate(step: int) -> None:
        n = args.val_per_task * len(objective.tasks)
        puzzles = objective.batch(val_source, "val", 0, n)  # fixed held-out set across evaluations
        sinks: list[list[Score]] = [[] for _ in puzzles]
        trajs = await art.gather_trajectories(
            (rollout_val(p, puzzle_messages(p), 0.0, s, split="val") for p, s in zip(puzzles, sinks)),
            pbar_desc=f"val@{step}",
            max_exceptions=args.max_exceptions,
        )
        ok = [t for t in trajs if isinstance(t, art.Trajectory)]
        summary = summarize(puzzles, sinks)
        summary["failed_rollouts"] = len(trajs) - len(ok)
        summary["request_retries"] = sum(retry_counts.values())
        retry_counts.clear()
        await model.log(ok, split="val", step=step)
        await model.log(metrics={f"puzzle_val/{k}": v for k, v in summary.items()}, step=step)
        append_jsonl(out_dir / "val_metrics.jsonl", {"step": step, **summary})
        with (out_dir / f"val_responses_step{step:05d}.jsonl").open("w") as f:
            for t in ok:
                f.write(json.dumps({**t.metadata, "reward": t.reward, **t.metrics}, default=str) + "\n")
        print(f"val@{step}: " + " ".join(f"{k}={v:.3f}" for k, v in summary.items() if k.endswith("accuracy")))

    start = policy["step"]
    print(f"model {model.entity}/{model.project}/{args.name} at step {start}; training to {args.steps}")
    dynamic = make_dynamic(args, objective)
    dynamic_path = out_dir / "dynamic_difficulty.jsonl"
    if dynamic is not None:
        if not dynamic.load(dynamic_path, start):
            dynamic.save(dynamic_path, start)
        print("dynamic difficulty: " + json.dumps(dynamic.state()))
    if val_source is not None and start % args.val_every == 0:
        await evaluate(start)
    for step in range(start, args.steps):
        t0 = time.perf_counter()
        task_probs = dynamic.weights() if dynamic is not None else None
        puzzles = await asyncio.to_thread(objective.batch, train_source, "train", step, args.groups_per_step,
                                          task_probs)
        sinks: list[list[Score]] = [[] for _ in puzzles]
        groups = await art.gather_trajectory_groups(
            [make_group(p, s) for p, s in zip(puzzles, sinks)],
            pbar_desc=f"train@{step}",
            max_exceptions=args.max_exceptions,
        )
        rollout_s = time.perf_counter() - t0
        result = await backend.train(
            model, groups, learning_rate=args.learning_rate, loss_fn="ppo", kl_penalty_coef=args.kl_penalty,
            scale_rewards=True,
        )
        summary = summarize(puzzles, sinks)
        summary.update(rollout_seconds=rollout_s, train_seconds=time.perf_counter() - t0 - rollout_s,
                       request_retries=sum(retry_counts.values()))
        retry_counts.clear()
        if dynamic is not None:
            summary.update({f"{t}/task_prob": p for t, p in task_probs.items()})
            dynamic.update(summary)
            dynamic.save(dynamic_path, result.step)
        policy["step"] = result.step
        await model.log(groups, metrics=result.metrics, step=result.step, split="train")
        await model.log(metrics={f"puzzle_train/{k}": v for k, v in summary.items()}, step=result.step)
        append_jsonl(out_dir / "train_metrics.jsonl", {"step": result.step, "data_step": step, **summary,
                                                        "artifact": getattr(result, "artifact_name", None)})
        print(f"step {result.step}: reward={summary.get('reward', 0):.3f} acc={summary.get('accuracy', 0):.3f} "
              f"format={summary.get('format', 0):.3f} degenerate={summary['degenerate_group_frac']:.2f}")
        if val_source is not None and result.step % args.val_every == 0:
            await evaluate(result.step)
    await backend.close()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    objective = Objective.from_spec(args.tasks, args.difficulty, args.task_params, args.seed)
    out_dir = args.output_dir or C.SERVERLESS_OUTPUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"run_config_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json",
               {"argv": sys.argv, "args": vars(args), "objective": objective.describe(),
                "baseline": "SSL-R1 arXiv:2604.20705"})
    if args.dry_run:
        dry_run(args, objective, out_dir)
        return
    os.environ.setdefault("WANDB_DIR", str(C.SERVERLESS_OUTPUT))  # keep wandb run files out of the repository
    asyncio.run(train(args, objective, out_dir))


if __name__ == "__main__":
    main()
