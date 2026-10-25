"""Masked-view caption game: self-play RL where one policy captions, forges, and judges.

For each image x the same policy plays three roles:

- captioner: sees the full image and writes a detailed caption (the "real" caption);
- forger: sees x with a salient region masked out and must describe the complete image, guessing what is hidden
  (the "fake" caption);
- judge: sees the full image and one real/fake pair in random order and must say which caption was written from
  the full image.

The real/fake label comes from which view produced the caption, so no annotation is needed, and the fakes are
natural hallucinations (guesses from context). Rewards, per GRPO group:

- judge: 1 for picking the full-image caption, plus the shared 0.2 format bonus;
- captioner: share of judge samples that picked its caption as real;
- forger: share of judge samples it fooled, set to 0 when its caption mentions the mask, hedges, or its length is
  outside [min_len_ratio, max_len_ratio] of the paired real caption (style shortcuts the judge could exploit).

Pairs whose two captions are near-identical (word-set Jaccard above --max-pair-overlap) carry no signal: they are
not judged, and both captions get the neutral reward 0.5.

    WANDB_API_KEY=... python -m serverless.caption_game --name capgame-pilot --steps 200 --val-every 20
    python -m serverless.caption_game --name smoke --dry-run      # masks + prompts only, no API calls
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import numpy as np
from PIL import Image, ImageFilter

from serverless import config as C
from serverless.images import ImageSource, to_data_uri
from serverless.rewards import boxed_contents, format_ok
from serverless.tasks.inpainting import random_box, sample_mask_size
from serverless.train import message_reasoning

Box = tuple[int, int, int, int]
ROLES = ("captioner", "forger", "judge")

CAPTION_PROMPT = (
    "Describe this image in detail in one paragraph of {lo}-{hi} words. Mention the objects, their attributes "
    "(colours, counts, text), and where they are. Write only the description."
)
FORGE_PROMPT = (
    "Part of this image is hidden by a grey box. Describe the complete original image in detail in one paragraph "
    "of {lo}-{hi} words, as if you could see all of it, including what is most likely in the hidden area. "
    "Mention the objects, their attributes (colours, counts, text), and where they are. Do not mention the box, "
    "that anything is hidden, or any uncertainty. Write only the description."
)
JUDGE_PROMPT = (
    "Two people described this image. One saw the full image. The other saw it with a region hidden and had to "
    "guess what was there. Which caption was written by the person who saw the full image?\n\n"
    "Caption A: {a}\n\nCaption B: {b}\n\n"
    "Compare each caption with the image and look for details that are wrong or invented. "
    "Keep your reasoning to a few sentences. The final answer should be a single letter, A or B."
)

# Words that give a forged caption away by style rather than content.
LEAK_PATTERNS = re.compile(
    r"\b(possibly|probably|likely|perhaps|maybe|might|appears? to|seems?|unclear|presumably|"
    r"(?:grey|gray) (?:box|rectangle|square|block|area|region|patch|overlay)|"
    r"hidden (?:area|region|part|portion|section)|masked (?:area|region|part|portion)|"
    r"(?:obscur|block|cover|occlud|hid)(?:ed|ing|es|s)? by (?:a|an|the) (?:grey|gray|large|solid|uniform|blank)|"
    r"can(?:not|'t) (?:see|tell)|not visible)\b",
    re.IGNORECASE,
)
WORD = re.compile(r"[a-z0-9']+")


# ----------------------------------------------------------------------------------------------- masking

def edge_energy(img: Image.Image) -> np.ndarray:
    """Per-pixel edge magnitude, a cheap stand-in for saliency (masks over sky or wall are too easy to guess)."""
    return np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)


def salient_box(rng: random.Random, img: Image.Image, ratio: float, candidates: int, min_side: int) -> Box:
    """Among `candidates` random boxes covering `ratio` of the image, the one with the most edge energy."""
    width, height = img.size
    w, h = sample_mask_size(rng, width, height, ratio, min_side)
    energy = edge_energy(img)
    integral = np.pad(energy.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, best_score = None, -1.0
    for _ in range(max(1, candidates)):
        x0, y0, x1, y1 = random_box(rng, width, height, w, h)
        score = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        if score > best_score:
            best, best_score = (x0, y0, x1, y1), float(score)
    return best


def mask_image(img: Image.Image, box: Box, fill: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    masked = img.copy()
    masked.paste(fill, box)
    return masked


# ----------------------------------------------------------------------------------------------- scoring

def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def clean_caption(text: str) -> str:
    """Caption as shown to the judge: no boxed wrapper, whitespace collapsed."""
    boxes = boxed_contents(text)
    if boxes:
        text = boxes[-1]
    return " ".join(text.split())


@dataclass
class ForgeGate:
    min_len_ratio: float = 0.6
    max_len_ratio: float = 1.6

    def check(self, forged: str, real: str) -> tuple[bool, str]:
        if LEAK_PATTERNS.search(forged):
            return False, "leak"
        n_real, n_fake = len(words(real)), len(words(forged))
        if n_real == 0 or n_fake == 0:
            return False, "empty"
        ratio = n_fake / n_real
        if not self.min_len_ratio <= ratio <= self.max_len_ratio:
            return False, "length"
        return True, "ok"


def parse_judgement(text: str) -> str | None:
    """The judge's letter: last \\boxed{}, else a closing "Final answer: X" or a bare letter on the last line.

    The format bonus still requires \\boxed{}; this only decides whether the verdict counts."""
    boxes = boxed_contents(text)
    if boxes:
        content = boxes[-1]
    else:
        tail = text.strip().splitlines()[-1] if text.strip() else ""
        m = re.search(r"final answer\s*(?:is)?\s*[:\-]?\s*\**\s*\(?\s*(?:caption\s+)?([AB])\b\s*\)?\**\s*\.?\s*$",
                      tail, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        content = tail
    m = re.fullmatch(r"\s*\(?\s*(?:caption\s+)?([AB])\s*\)?\s*\.?\s*", content, re.IGNORECASE)
    return m.group(1).upper() if m else None


# ----------------------------------------------------------------------------------------------- one image

@dataclass
class Rollout:
    """One sampled response in a role; `reward` is filled in once the judge has played."""

    role: str
    messages: list[dict[str, Any]]
    completion: Any
    text: str
    reward: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameResult:
    image: str
    box: Box
    captions: list[Rollout]
    forgeries: list[Rollout]
    judge_groups: list[list[Rollout]]
    stats: dict[str, float]


def text_message(img: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": to_data_uri(img)}},
        {"type": "text", "text": prompt},
    ]}]


def judge_message(img_uri: str, a: str, b: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img_uri}},
        {"type": "text", "text": JUDGE_PROMPT.format(a=a, b=b) + "\n" + C.FORMAT_PROMPT},
    ]}]


Complete = Callable[[list[dict[str, Any]], float, str], Awaitable[Any]]


@dataclass
class GameConfig:
    caption_samples: int = 4      # real and forged captions per image (pairs per image)
    judge_samples: int = 4        # judge responses per pair
    temperature: float = 1.0
    words_lo: int = 50
    words_hi: int = 90
    mask_candidates: int = 8
    min_side: int = 28
    max_pair_overlap: float = 0.9
    caption_slack_lo: float = 0.75   # captioner length window: [words_lo * lo, words_hi * hi]
    caption_slack_hi: float = 1.25
    format_reward: float = C.FORMAT_REWARD
    thinking: bool = C.ENABLE_THINKING
    gate: ForgeGate = field(default_factory=ForgeGate)


async def play(
    cfg: GameConfig, complete: Complete, img: Image.Image, image_name: str, ratio: float, rng: random.Random,
    temperature: float | None = None,
) -> GameResult:
    """Play one image: G real and G forged captions in parallel, then judge each (real_i, forged_i) pair."""
    temp = cfg.temperature if temperature is None else temperature
    box = salient_box(rng, img, ratio, cfg.mask_candidates, cfg.min_side)
    masked = mask_image(img, box)
    cap_msgs = text_message(img, CAPTION_PROMPT.format(lo=cfg.words_lo, hi=cfg.words_hi))
    forge_msgs = text_message(masked, FORGE_PROMPT.format(lo=cfg.words_lo, hi=cfg.words_hi))
    g = cfg.caption_samples

    async def sample(role: str, msgs: list[dict[str, Any]]) -> Rollout:
        completion = await complete(msgs, temp, role)
        text = completion.choices[0].message.content or ""
        return Rollout(role, msgs, completion, clean_caption(text))

    produced = await asyncio.gather(*[sample("captioner", cap_msgs) for _ in range(g)],
                                    *[sample("forger", forge_msgs) for _ in range(g)])
    captions, forgeries = list(produced[:g]), list(produced[g:])

    img_uri = to_data_uri(img)
    judge_jobs: list[tuple[int, str, list[dict[str, Any]]]] = []  # (pair, real letter, messages)
    gate_reasons: Counter[str] = Counter()
    for i, (real, fake) in enumerate(zip(captions, forgeries)):
        passed, reason = cfg.gate.check(fake.text, real.text)
        n_fake = len(words(fake.text))
        if passed and not cfg.words_lo * cfg.caption_slack_lo <= n_fake <= cfg.words_hi * cfg.caption_slack_hi:
            passed, reason = False, "length"  # same absolute window as the captioner
        gate_reasons[reason] += 1
        fake.metrics["gate_passed"] = float(passed)
        fake.metadata["gate"] = reason
        overlap = jaccard(real.text, fake.text)
        for r in (real, fake):
            r.metrics["pair_overlap"] = overlap
            r.metrics["words"] = float(len(words(r.text)))
        # Longer captions are easier to pick as real, so an unbounded captioner drifts to very long captions;
        # one outside the requested word range scores 0 and its pair is not judged.
        n_real = len(words(real.text))
        real_ok = cfg.words_lo * cfg.caption_slack_lo <= n_real <= cfg.words_hi * cfg.caption_slack_hi
        # Forgeries may not hedge, so a hedging real caption would be an easy style tell; gate it the same way.
        real_ok = real_ok and not LEAK_PATTERNS.search(real.text)
        real.metrics["length_ok"] = float(real_ok)
        if not real_ok:
            real.reward, fake.reward = 0.0, (0.5 if passed else 0.0)
            real.metadata["judged"] = fake.metadata["judged"] = False
            continue
        if overlap > cfg.max_pair_overlap or not real.text or not fake.text:
            real.reward = fake.reward = 0.5 if passed else 0.0
            real.metadata["judged"] = fake.metadata["judged"] = False
            continue
        real_first = rng.random() < 0.5
        a, b = (real.text, fake.text) if real_first else (fake.text, real.text)
        judge_jobs.append((i, "A" if real_first else "B", judge_message(img_uri, a, b)))

    async def judge(pair: int, answer: str, msgs: list[dict[str, Any]]) -> Rollout:
        completion = await complete(msgs, temp, "judge")
        message = completion.choices[0].message
        text = message.content or ""
        pred = parse_judgement(text)
        correct = float(pred == answer)
        # With native thinking the reasoning arrives on its own channel; otherwise it must precede the box.
        fmt = format_ok(text, reasoning=message_reasoning(message) if cfg.thinking else None)
        roll = Rollout("judge", msgs, completion, text, correct + (cfg.format_reward if fmt else 0.0))
        roll.metrics.update(accuracy=correct, format=float(fmt), parsed=float(pred is not None),
                            picked_a=float(pred == "A"))
        roll.metadata.update(pair=pair, answer=answer, predicted=pred)
        return roll

    judged = await asyncio.gather(*[judge(i, ans, msgs) for i, ans, msgs in judge_jobs
                                    for _ in range(cfg.judge_samples)])
    judge_groups: list[list[Rollout]] = []
    for k, (i, _, _) in enumerate(judge_jobs):
        group = list(judged[k * cfg.judge_samples:(k + 1) * cfg.judge_samples])
        judge_groups.append(group)
        # Only parsed verdicts count: an unparseable judge answer is not a fooled judge, and counting it would
        # pay the forger for captions that break the judge's output format.
        verdicts = [r.metrics["accuracy"] for r in group if r.metrics["parsed"]]
        p_real = statistics.fmean(verdicts) if verdicts else 0.5
        captions[i].reward = p_real
        forgeries[i].reward = (1.0 - p_real) if forgeries[i].metrics["gate_passed"] else 0.0
        captions[i].metadata["judged"] = forgeries[i].metadata["judged"] = True
        captions[i].metrics["judged_real"] = forgeries[i].metrics["judged_real"] = p_real

    all_judges = [r for grp in judge_groups for r in grp]
    stats = {
        "judge/accuracy": statistics.fmean(r.metrics["accuracy"] for r in all_judges) if all_judges else float("nan"),
        "judge/picked_a": statistics.fmean(r.metrics["picked_a"] for r in all_judges) if all_judges else float("nan"),
        "judge/accuracy_parsed": (statistics.fmean(r.metrics["accuracy"] for r in all_judges if r.metrics["parsed"])
                                  if any(r.metrics["parsed"] for r in all_judges) else float("nan")),
        "judge/parsed": statistics.fmean(r.metrics["parsed"] for r in all_judges) if all_judges else float("nan"),
        "judge/format": statistics.fmean(r.metrics["format"] for r in all_judges) if all_judges else float("nan"),
        "judged_pair_frac": len(judge_jobs) / max(1, g),
        "forger/gate_pass": gate_reasons["ok"] / max(1, g),
        "forger/gate_leak": gate_reasons["leak"] / max(1, g),
        "forger/gate_length": gate_reasons["length"] / max(1, g),
        "captioner/reward": statistics.fmean(r.reward for r in captions),
        "forger/reward": statistics.fmean(r.reward for r in forgeries),
        "captioner/length_ok": statistics.fmean(r.metrics["length_ok"] for r in captions),
        "captioner/words": statistics.fmean(len(words(r.text)) for r in captions),
        "forger/words": statistics.fmean(len(words(r.text)) for r in forgeries),
        "pair_overlap": statistics.fmean(r.metrics["pair_overlap"] for r in captions),
        "captioner/leak": statistics.fmean(bool(LEAK_PATTERNS.search(r.text)) for r in captions),
    }
    return GameResult(image_name, box, captions, forgeries, judge_groups, stats)


# ----------------------------------------------------------------------------------------------- difficulty

@dataclass
class MaskController:
    """Steers the masked fraction so judge accuracy stays near `target` (bigger masks make fakes easier to spot)."""

    ratio: float = 0.25
    target: float = 0.7
    gain: float = 0.5
    lo: float = 0.1
    hi: float = 0.6
    enabled: bool = True

    def update(self, judge_accuracy: float) -> None:
        if self.enabled and judge_accuracy == judge_accuracy:  # skip NaN
            self.ratio = min(self.hi, max(self.lo, self.ratio - self.gain * (judge_accuracy - self.target) * self.ratio))

    def save(self, path: Path, step: int) -> None:
        with path.open("a") as f:
            f.write(json.dumps({"step": step, "ratio": self.ratio}) + "\n")

    def load(self, path: Path, step: int) -> bool:
        """Restore the ratio recorded at `step` (resume); False when there is none."""
        if not path.exists():
            return False
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["step"] == step:
                self.ratio = row["ratio"]
                return True
        return False


# ----------------------------------------------------------------------------------------------- training loop

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--project", default=C.WANDB_PROJECT)
    p.add_argument("--entity", default=C.WANDB_ENTITY)
    p.add_argument("--base-model", default=C.BASE_MODEL)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--images-per-step", type=int, default=8)
    p.add_argument("--caption-samples", type=int, default=4, help="real and forged captions per image")
    p.add_argument("--judge-samples", type=int, default=4, help="judge responses per (real, forged) pair")
    p.add_argument("--train-roles", default="captioner,forger,judge",
                   help="roles whose trajectories are trained; e.g. 'judge' for the judge-only pilot")
    p.add_argument("--freeze-captions", action="store_true",
                   help="captions and forgeries always come from the base checkpoint (step 0), not the policy")
    p.add_argument("--mask-ratio", type=float, default=0.25, help="initial masked fraction of the image")
    p.add_argument("--fixed-mask", action="store_true", help="keep --mask-ratio instead of steering it")
    p.add_argument("--judge-target", type=float, default=0.7, help="judge accuracy the mask controller aims for")
    p.add_argument("--words-lo", type=int, default=50)
    p.add_argument("--words-hi", type=int, default=90)
    p.add_argument("--min-len-ratio", type=float, default=0.6)
    p.add_argument("--max-len-ratio", type=float, default=1.6)
    p.add_argument("--max-pair-overlap", type=float, default=0.9)
    p.add_argument("--learning-rate", type=float, default=C.LEARNING_RATE)
    p.add_argument("--kl-penalty", type=float, default=C.KL_PENALTY)
    p.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    p.add_argument("--top-p", type=float, default=C.TOP_P)
    p.add_argument("--top-k", type=int, default=C.TOP_K)
    p.add_argument("--max-completion-tokens", type=int, default=C.MAX_COMPLETION_TOKENS)
    p.add_argument("--judge-max-tokens", type=int, default=768,
                   help="judge completion cap; a truncated judgement has no answer and scores 0")
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=C.ENABLE_THINKING)
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--val-every", type=int, default=20, help="0 disables")
    p.add_argument("--val-images-count", type=int, default=32)
    p.add_argument("--train-images", type=Path, default=C.TRAIN_IMAGES)
    p.add_argument("--val-images", type=Path, default=C.VAL_IMAGES)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="save masks and prompts for one step; no API calls")
    return p


def game_config(args: argparse.Namespace) -> GameConfig:
    return GameConfig(
        caption_samples=args.caption_samples, judge_samples=args.judge_samples, temperature=args.temperature,
        words_lo=args.words_lo, words_hi=args.words_hi, max_pair_overlap=args.max_pair_overlap,
        thinking=args.enable_thinking, gate=ForgeGate(args.min_len_ratio, args.max_len_ratio),
    )


def step_images(source: ImageSource, seed: int, split: str, step: int, n: int) -> list[tuple[str, Image.Image, random.Random]]:
    out = []
    for slot in range(n):
        rng = random.Random(f"{seed}:capgame:{split}:{step}:{slot}")
        idx, img = source.sample(rng)
        out.append((source.names[idx], img, rng))
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def mean_stats(results: list[GameResult]) -> dict[str, float]:
    keys = sorted({k for r in results for k in r.stats})
    out = {}
    for k in keys:
        vals = [r.stats[k] for r in results if k in r.stats and r.stats[k] == r.stats[k]]
        if vals:
            out[k] = statistics.fmean(vals)
    return out


def dry_run(args: argparse.Namespace, out_dir: Path) -> None:
    cfg = game_config(args)
    preview = out_dir / "dry_run"
    preview.mkdir(parents=True, exist_ok=True)
    for i, (name, img, rng) in enumerate(step_images(ImageSource(args.train_images), args.seed, "train", 0,
                                                      args.images_per_step)):
        box = salient_box(rng, img, args.mask_ratio, cfg.mask_candidates, cfg.min_side)
        img.save(preview / f"{i:02d}_full.jpg", quality=90)
        mask_image(img, box).save(preview / f"{i:02d}_masked.jpg", quality=90)
        append_jsonl(preview / "games.jsonl", {"i": i, "image": name, "box": box})
        print(f"[{i:02d}] {name} box={box}")
    (preview / "prompts.txt").write_text(
        "CAPTION:\n" + CAPTION_PROMPT.format(lo=cfg.words_lo, hi=cfg.words_hi) + "\n\nFORGE:\n"
        + FORGE_PROMPT.format(lo=cfg.words_lo, hi=cfg.words_hi) + "\n\nJUDGE:\n"
        + JUDGE_PROMPT.format(a="<real or forged>", b="<forged or real>") + "\n" + C.FORMAT_PROMPT + "\n")
    print(f"preview in {preview}")


def to_trajectory(art: Any, roll: Rollout, policy_step: int) -> Any:
    choice = roll.completion.choices[0]
    usage = getattr(roll.completion, "usage", None)
    metrics = {k: float(v) for k, v in roll.metrics.items()}
    metrics.update(truncated=float(choice.finish_reason == "length"),
                   completion_tokens=float(usage.completion_tokens if usage else 0))
    return art.Trajectory(
        messages_and_choices=[*roll.messages, choice], reward=roll.reward, metrics=metrics,
        metadata={"role": roll.role, "response": roll.text, "policy_step": policy_step,
                  **{k: v for k, v in roll.metadata.items() if isinstance(v, (str, int, float, bool)) or v is None}},
    ).finish()


def build_groups(art: Any, results: list[GameResult], roles: set[str], policy_step: int) -> list[Any]:
    """One GRPO group per image for captions and for forgeries, one per judged pair; groups of one are dropped."""
    groups = []
    for res in results:
        for role, rolls in (("captioner", res.captions), ("forger", res.forgeries)):
            if role in roles and len(rolls) > 1:
                groups.append(art.TrajectoryGroup([to_trajectory(art, r, policy_step) for r in rolls],
                                                  metadata={"role": role, "image": res.image}))
        if "judge" in roles:
            for grp in res.judge_groups:
                if len(grp) > 1:
                    groups.append(art.TrajectoryGroup([to_trajectory(art, r, policy_step) for r in grp],
                                                      metadata={"role": "judge", "image": res.image}))
    return groups


async def train(args: argparse.Namespace, out_dir: Path) -> None:
    import art
    import openai
    from art.openai import consume_chat_completion_stream
    from art.serverless.backend import ServerlessBackend

    roles = {r.strip() for r in args.train_roles.split(",") if r.strip()}
    if not roles <= set(ROLES):
        raise SystemExit(f"--train-roles must be a subset of {ROLES}")
    cfg = game_config(args)
    train_source = ImageSource(args.train_images)
    val_source = ImageSource(args.val_images) if args.val_every > 0 else None
    backend = ServerlessBackend()
    model = art.TrainableModel(name=args.name, run_name=args.name, project=args.project, entity=args.entity,
                               base_model=args.base_model, base_path=str(C.SERVERLESS_OUTPUT / "art"))
    await model.register(backend)
    client = model.openai_client()
    semaphore = asyncio.Semaphore(args.max_concurrency)
    retries: Counter[str] = Counter()
    policy = {"step": await model.get_step()}
    base_step = 0
    (out_dir / "model.json").write_text(json.dumps({"id": model.id, "entity": model.entity, "project": model.project,
                                                    "name": args.name, "base_model": args.base_model}, indent=2))

    async def complete(messages: list[dict], temperature: float, role: str) -> Any:
        step = base_step if (args.freeze_captions and role != "judge") else policy["step"]
        for attempt in range(args.max_retries + 1):
            try:
                async with semaphore:
                    extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}
                    request = dict(model=model.get_inference_name(step), messages=messages,
                                   max_completion_tokens=(args.judge_max_tokens if role == "judge" else args.max_completion_tokens),
                                   temperature=temperature,
                                   logprobs=True, timeout=args.request_timeout, extra_body=extra_body)
                    if temperature > 0:
                        if args.top_p is not None:
                            request["top_p"] = args.top_p
                        if args.top_k is not None:
                            extra_body["top_k"] = args.top_k
                    if not args.stream:
                        return await client.chat.completions.create(**request)
                    stream = await client.chat.completions.create(**request, stream=True,
                                                                  stream_options={"include_usage": True})
                    return await consume_chat_completion_stream(stream)
            except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError,
                    openai.APIConnectionError) as exc:
                if attempt == args.max_retries:
                    raise
                retries[type(exc).__name__] += 1
                await asyncio.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")

    # Greedy evaluation: one caption, one forgery and one judgement per image (more samples would be identical).
    val_cfg = dataclasses.replace(cfg, caption_samples=1, judge_samples=1)

    async def play_all(images: list[tuple[str, Image.Image, random.Random]], ratio: float,
                       temperature: float | None = None, game: GameConfig = cfg) -> list[GameResult]:
        outcomes = await asyncio.gather(*[play(game, complete, img, name, ratio, rng, temperature)
                                          for name, img, rng in images], return_exceptions=True)
        failed = [o for o in outcomes if isinstance(o, BaseException)]
        for exc in failed[:3]:
            print(f"game failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if len(failed) > max(1, len(outcomes) // 4):
            raise RuntimeError(f"{len(failed)}/{len(outcomes)} games failed") from failed[0]
        return [o for o in outcomes if not isinstance(o, BaseException)]

    async def train_with_retry(groups: list[Any]) -> Any:
        """backend.train with backoff on 5xx/timeouts (e.g. a 524 from the proxy while the service is slow).

        A failed call may still have trained server-side, so before retrying we check whether the checkpoint
        advanced; if it did, that step is kept instead of training the same groups twice."""
        before = policy["step"]
        for attempt in range(args.max_retries + 1):
            try:
                return await backend.train(model, groups, learning_rate=args.learning_rate, loss_fn="ppo",
                                           kl_penalty_coef=args.kl_penalty, scale_rewards=True)
            except (openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError) as exc:
                if attempt == args.max_retries:
                    raise
                retries[f"train_{type(exc).__name__}"] += 1
                print(f"backend.train failed ({type(exc).__name__}); retry {attempt + 1}", file=sys.stderr)
                await asyncio.sleep(min(300.0, 30.0 * 2 ** attempt))
                step = await model.get_step()
                if step > before:
                    print(f"checkpoint advanced to step {step} despite the error; keeping it", file=sys.stderr)
                    return SimpleNamespace(step=step, metrics={})
        raise AssertionError("unreachable")

    controller = MaskController(ratio=args.mask_ratio, target=args.judge_target, enabled=not args.fixed_mask)
    controller_path = out_dir / "mask_ratio.jsonl"
    start = policy["step"]
    if not controller.load(controller_path, start):
        controller.save(controller_path, start)

    async def evaluate(step: int) -> None:
        # Fixed val images and a fixed mask ratio, so evaluations are comparable across steps.
        images = step_images(val_source, args.seed, "val", 0, args.val_images_count)
        results = await play_all(images, args.mask_ratio, temperature=0.0, game=val_cfg)
        summary = mean_stats(results)
        await model.log(metrics={f"capgame_val/{k}": v for k, v in summary.items()}, step=step)
        append_jsonl(out_dir / "val_metrics.jsonl", {"step": step, **summary})
        with (out_dir / f"val_games_step{step:05d}.jsonl").open("w") as f:
            for res in results:
                f.write(json.dumps({"image": res.image, "box": res.box,
                                    "real": [r.text for r in res.captions],
                                    "forged": [r.text for r in res.forgeries],
                                    "gate": [r.metadata.get("gate") for r in res.forgeries],
                                    "judge": [[j.text for j in grp] for grp in res.judge_groups],
                                    "stats": res.stats}, default=str) + "\n")
        print(f"val@{step}: judge_acc={summary.get('judge/accuracy', float('nan')):.3f} "
              f"gate_pass={summary.get('forger/gate_pass', 0):.2f}")

    print(f"model {model.entity}/{model.project}/{args.name} at step {start}; training {sorted(roles)} to {args.steps}")
    if val_source is not None and start % args.val_every == 0:
        await evaluate(start)
    for step in range(start, args.steps):
        t0 = time.perf_counter()
        images = await asyncio.to_thread(step_images, train_source, args.seed, "train", step, args.images_per_step)
        results = await play_all(images, controller.ratio)
        groups = build_groups(art, results, roles, policy["step"])
        rollout_s = time.perf_counter() - t0
        result = await train_with_retry(groups)
        summary = mean_stats(results)
        summary.update(mask_ratio=controller.ratio, n_groups=len(groups), rollout_seconds=rollout_s,
                       train_seconds=time.perf_counter() - t0 - rollout_s, request_retries=sum(retries.values()),
                       degenerate_group_frac=statistics.fmean(
                           len({t.reward for t in g.trajectories}) <= 1 for g in groups) if groups else 1.0)
        retries.clear()
        controller.update(summary.get("judge/accuracy_parsed", float("nan")))  # unparsed answers are not "fooled"
        controller.save(controller_path, result.step)
        policy["step"] = result.step
        await model.log(groups, metrics=result.metrics, step=result.step, split="train")
        await model.log(metrics={f"capgame_train/{k}": v for k, v in summary.items()}, step=result.step)
        append_jsonl(out_dir / "train_metrics.jsonl", {"step": result.step, **summary})
        print(f"step {result.step}: judge_acc={summary.get('judge/accuracy', float('nan')):.3f} "
              f"cap_r={summary.get('captioner/reward', 0):.2f} forge_r={summary.get('forger/reward', 0):.2f} "
              f"gate={summary.get('forger/gate_pass', 0):.2f} mask={summary['mask_ratio']:.2f} "
              f"degenerate={summary['degenerate_group_frac']:.2f}")
        if val_source is not None and result.step % args.val_every == 0:
            await evaluate(result.step)
    await backend.close()


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    out_dir = args.output_dir or C.SERVERLESS_OUTPUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"run_config_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json").write_text(
        json.dumps({"argv": sys.argv, "args": vars(args), "objective": "masked-view caption game",
                    "prompts": {"caption": CAPTION_PROMPT, "forge": FORGE_PROMPT, "judge": JUDGE_PROMPT,
                                "format": C.FORMAT_PROMPT}}, indent=2, default=str))
    if args.dry_run:
        dry_run(args, out_dir)
        return
    os.environ.setdefault("WANDB_DIR", str(C.SERVERLESS_OUTPUT))
    asyncio.run(train(args, out_dir))


if __name__ == "__main__":
    main()
