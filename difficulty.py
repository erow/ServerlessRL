"""Measure per-task difficulty: run any OpenAI-compatible chat endpoint on COCO puzzles and report accuracy.

Each task gets its own fixed puzzle set (puzzle i of every task uses the same seed, so tasks are compared on the
same images). Each puzzle is sampled `--samples` times; besides mean accuracy, the report gives the solve rate
(accuracy == 1), pass@k over the samples, and the share of puzzles whose samples all score the same (no GRPO signal).

  python -m serverless.difficulty --base-url http://localhost:8000/v1 --api-key EMPTY \\
      --model google/gemma-4-26B-A4B-it --n-per-task 50 --samples 4 --temperature 1.0

Responses are appended to <output-dir>/responses.jsonl; rerunning with the same output dir skips finished samples.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from serverless import config as C
from serverless.images import ImageSource
from serverless.objective import PRESETS, Objective, parse_mixture, puzzle_messages
from serverless.rewards import Score
from serverless.tasks import Puzzle
from serverless.train import message_reasoning, write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="default: $OPENAI_API_KEY")
    p.add_argument("--model", default=C.BASE_MODEL)
    p.add_argument("--tasks", default="ssl-r1", help=f"preset {sorted(PRESETS)} or 'task,...' (weights ignored)")
    p.add_argument("--difficulty", choices=["paper", "easy"], default="paper")
    p.add_argument("--task-params", default=None, help='JSON overrides, e.g. \'{"rotation": {"steps": 8}}\'')
    p.add_argument("--images", type=Path, default=C.VAL_IMAGES, help="COCO image directory or ZIP (default val2017)")
    p.add_argument("--n-per-task", type=int, default=50, help="puzzles per task")
    p.add_argument("--samples", type=int, default=1, help="responses per puzzle")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None, help="sent in extra_body (vLLM/SGLang)")
    p.add_argument("--max-completion-tokens", type=int, default=C.MAX_COMPLETION_TOKENS)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="send chat_template_kwargs.enable_thinking (vLLM/SGLang); omitted unless given")
    p.add_argument("--extra-body", default=None, help="JSON merged into the request's extra_body")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False,
                   help="stream responses (keeps long generations alive behind proxies with idle timeouts)")
    p.add_argument("--max-concurrency", type=int, default=16, help="concurrent requests")
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--max-retries", type=int, default=8, help="per-request retries on 429/5xx/timeouts, with backoff")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: $SERVERLESS_OUTPUT/difficulty/<model>_<difficulty>_T<temperature>")
    return p


class Reply:
    """Minimal stand-in for a chat message rebuilt from a stream (what message_reasoning reads)."""

    def __init__(self, content: str, reasoning: str | None):
        self.content = content
        self.reasoning = reasoning or None
        self.model_extra: dict[str, Any] = {}


async def complete(client: Any, args: argparse.Namespace, messages: list[dict]) -> tuple[Any, str | None, int]:
    """Returns (message, finish_reason, completion_tokens), retrying rate limits, server errors and timeouts."""
    import openai

    request: dict[str, Any] = dict(model=args.model, messages=messages, temperature=args.temperature,
                                   max_completion_tokens=args.max_completion_tokens, timeout=args.request_timeout)
    if args.top_p is not None:
        request["top_p"] = args.top_p
    extra = json.loads(args.extra_body) if args.extra_body else {}
    if args.top_k is not None:
        extra["top_k"] = args.top_k
    if args.enable_thinking is not None:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = args.enable_thinking
    if extra:
        request["extra_body"] = extra
    for attempt in range(args.max_retries + 1):
        try:
            if not args.stream:
                completion = await client.chat.completions.create(**request)
                choice = completion.choices[0]
                tokens = completion.usage.completion_tokens if completion.usage else 0
                return choice.message, choice.finish_reason, tokens
            stream = await client.chat.completions.create(**request, stream=True,
                                                          stream_options={"include_usage": True})
            content, reasoning, finish, tokens = [], [], None, 0
            async for chunk in stream:
                if chunk.usage:
                    tokens = chunk.usage.completion_tokens
                for choice in chunk.choices:
                    delta = choice.delta
                    extra_delta = getattr(delta, "model_extra", None) or {}
                    content.append(delta.content or "")
                    reasoning.append(getattr(delta, "reasoning", None) or extra_delta.get("reasoning")
                                     or extra_delta.get("reasoning_content") or "")
                    finish = choice.finish_reason or finish
            return Reply("".join(content), "".join(reasoning)), finish, tokens
        except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError,
                openai.APIConnectionError):
            if attempt == args.max_retries:
                raise
            await asyncio.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
    raise AssertionError("unreachable")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    prob = 1.0
    for i in range(n - c + 1, n + 1):
        prob *= 1.0 - k / i
    return 1.0 - prob


def summarize(rows: list[dict[str, Any]], samples: int) -> dict[str, dict[str, float]]:
    """Per-task (and overall) accuracy, format, solve rate, pass@k and zero-variance puzzle share."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    out = {}
    for task, task_rows in [*sorted(by_task.items()), ("all", rows)]:
        if not task_rows:
            continue
        acc = [r["accuracy"] for r in task_rows]
        puzzles: dict[str, list[dict]] = defaultdict(list)
        for r in task_rows:
            puzzles[f'{r["task"]}/{r["puzzle_id"]}'].append(r)
        stats = {
            "n_puzzles": len(puzzles),
            "n_responses": len(task_rows),
            "accuracy": statistics.fmean(acc),
            "accuracy_se": statistics.stdev(acc) / len(acc) ** 0.5 if len(acc) > 1 else 0.0,
            "solved": statistics.fmean(a == 1.0 for a in acc),
            "format": statistics.fmean(r["format"] for r in task_rows),
            "parsed": statistics.fmean(r["parsed"] for r in task_rows),
            "truncated": statistics.fmean(r["truncated"] for r in task_rows),
            "reward": statistics.fmean(r["reward"] for r in task_rows),
            "completion_tokens": statistics.fmean(r["completion_tokens"] for r in task_rows),
        }
        if samples > 1:
            groups = list(puzzles.values())
            stats[f"pass@{samples}"] = statistics.fmean(
                pass_at_k(len(g), sum(r["accuracy"] == 1.0 for r in g), samples) if len(g) >= samples else 0.0
                for g in groups)
            stats["zero_variance_frac"] = statistics.fmean(len({r["reward"] for r in g}) == 1 for g in groups)
        out[task] = stats
    return out


def print_table(summary: dict[str, dict[str, float]]) -> None:
    cols = ["n_puzzles", "accuracy", "accuracy_se", "solved", "format", "parsed", "truncated", "completion_tokens"]
    cols += [c for c in next(iter(summary.values())) if c.startswith("pass@") or c == "zero_variance_frac"]
    print(f"{'task':16s}" + "".join(f"{c:>19s}" for c in cols))
    for task, stats in summary.items():
        cells = "".join(f"{stats[c]:>19d}" if isinstance(stats[c], int) else f"{stats[c]:>19.3f}" for c in cols)
        print(f"{task:16s}{cells}")


async def run(args: argparse.Namespace, out_dir: Path) -> None:
    import openai

    names = list(parse_mixture(args.tasks))
    objectives = {name: Objective.from_spec(name, args.difficulty, args.task_params, args.seed) for name in names}
    source = ImageSource(args.images)
    responses_path = out_dir / "responses.jsonl"
    done: set[tuple[str, str, int]] = set()
    if responses_path.exists():
        with responses_path.open() as f:
            done = {(r["task"], r["puzzle_id"], r["sample"]) for r in map(json.loads, f)}
    write_json(out_dir / "config.json", {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                            if k != "api_key"},
                                         "tasks": {k: o.describe() for k, o in objectives.items()}})

    client = openai.AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    requests = asyncio.Semaphore(args.max_concurrency)
    # Bounds how many puzzles hold encoded images at once; a few more than needed to keep requests saturated.
    building = asyncio.Semaphore(max(1, 2 * args.max_concurrency // args.samples))
    lock = asyncio.Lock()
    counts: Counter[str] = Counter()
    total = len(names) * args.n_per_task * args.samples - len(done)
    t0 = time.perf_counter()

    async def sample(puzzle: Puzzle, messages: list[dict], k: int, f: Any) -> None:
        async with requests:
            try:
                message, finish, tokens = await complete(client, args, messages)
            except Exception as exc:  # keep going; failed samples are reported and retried on rerun
                counts["failed"] += 1
                print(f"failed {puzzle.task}/{puzzle.meta['puzzle_id']}#{k}: {type(exc).__name__}: {exc}")
                return
        text = message.content or ""
        message.content = text  # message_reasoning falls back to <think> tags in content, which may be None
        reasoning = message_reasoning(message)
        result: Score = objectives[puzzle.task].score(puzzle, text, reasoning if args.enable_thinking else None)
        row = {"task": puzzle.task, "puzzle_id": puzzle.meta["puzzle_id"], "sample": k, "answer": puzzle.answer,
               "reward": result.reward, "accuracy": result.accuracy, "format": result.format,
               "parsed": result.parsed, "truncated": finish == "length", "completion_tokens": tokens,
               "response": text, "reasoning": reasoning, "meta": puzzle.meta}
        async with lock:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            counts["done"] += 1
            if counts["done"] % 20 == 0 or counts["done"] == total:
                rate = counts["done"] / (time.perf_counter() - t0)
                print(f"{counts['done']}/{total} responses ({rate:.2f}/s, {counts['failed']} failed)", flush=True)

    async def puzzle_job(name: str, i: int, f: Any) -> None:
        todo = [k for k in range(args.samples) if (name, f"val-{i}-0", k) not in done]
        if not todo:
            return
        async with building:
            # puzzle i of every task draws from the same seeded RNG, so tasks see the same images
            puzzle = await asyncio.to_thread(objectives[name].puzzle, source, "val", i, 0, 1)
            messages = await asyncio.to_thread(puzzle_messages, puzzle)
            await asyncio.gather(*(sample(puzzle, messages, k, f) for k in todo))

    print(f"{args.model} @ {args.base_url}: {len(names)} tasks x {args.n_per_task} puzzles x {args.samples} samples "
          f"({total} to run, {len(done)} already done) -> {out_dir}")
    with responses_path.open("a") as f:
        await asyncio.gather(*(puzzle_job(n, i, f) for i in range(args.n_per_task) for n in names))
    await client.close()

    with responses_path.open() as f:
        rows = [r for r in map(json.loads, f) if r["task"] in names and int(r["puzzle_id"].split("-")[1]) <
                args.n_per_task and r["sample"] < args.samples]
    summary = summarize(rows, args.samples)
    write_json(out_dir / "summary.json", summary)
    print_table(summary)
    if counts["failed"]:
        print(f"{counts['failed']} requests failed; rerun the same command to retry them")


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.base_url is None:
        raise SystemExit("--base-url (or $OPENAI_BASE_URL) is required")
    out_dir = args.output_dir or (C.SERVERLESS_OUTPUT / "difficulty" /
                                  f"{args.model.replace('/', '_')}_{args.difficulty}_T{args.temperature:g}")
    out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args, out_dir))


if __name__ == "__main__":
    main()
