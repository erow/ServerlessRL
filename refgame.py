"""Referring game: a speaker describes a boxed region, a frozen listener must pick that region among look-alikes.

One round: an image with a red box drawn around a target region (the speaker's view), and K crops of the same size
(the listener's view): the target and K-1 distractors (other regions of the same image, regions of other images,
optionally COCO-label-matched or colour/sharpness-matched ones; see `tasks/inpainting.py` for the kinds). The crops
carry no position, so only what is *inside* the region can identify it.

  speaker (trained policy): image with box -> description of at most `max_words` words
  listener (frozen base):   K crops + description -> letter; we read its probabilities over the K letters

Reward = listener's probability of the right crop, averaged over `permutations` cyclic orders of the candidates
(cancels letter bias), minus `length_penalty` per word beyond `max_words`. There is no recall or content term:
an invented detail cannot help the listener, and one that happens to match a distractor lowers the reward.

The listener helper `letter_probs` works with any Hugging Face VLM + processor (Qwen3-VL here). Phase 0 (no
training) checks the base model before any run: blind and colour-only listeners (shortcut floors), greedy speaker
accuracy, and whether sampled descriptions give GRPO signal (reward spread within a group):

  python -m serverless.refgame phase0 --images /path/val2017 --out-dir out/refgame_phase0 --n 200 \\
      --scheme same='{"distractors": "same-image"}' --scheme mixed='{"distractors": "same-image,random,random"}'
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from serverless.images import ImageSource
from serverless.tasks.inpainting import InpaintingTask, iou, parse_kinds, random_box

LETTERS = "ABCDEFGH"
Box = tuple[int, int, int, int]

SPEAKER_PROMPT = (
    "Look at the region inside the red box. Someone will be shown {k} small image crops: this region and {d} "
    "similar-looking regions, some from this photo and some from other photos, in random order and without the rest "
    "of the photo. Write a description of at most {w} words that lets them pick out this region. Describe only what "
    "is visible inside the red box; its position in the photo will not help. Reply with the description only."
)
LISTENER_PROMPT = "Description: \"{desc}\"\nWhich crop does the description refer to? Answer with a single letter."
BLIND_DESC = "(no description was given; guess)"


@dataclass
class RefGameConfig:
    choices: int = 4
    ratio: tuple[float, float] = (0.05, 0.15)  # target area as a share of the image, uniform in this range
    min_side: int = 48
    salient_candidates: int = 8  # keep the box with the most edge energy among this many (1 = uniform)
    distractors: str = "same-image"  # inpainting kinds, one for all slots or a comma list per slot
    oracle_index: str | None = None
    color_select: int = 1  # >1: among this many cross-image candidates keep the most similar (select_by)
    select_by: str = "stats"
    crop_side: int = 224  # every candidate is resized so its longer side is this
    box_width: int = 3
    max_words: int = 20
    length_penalty: float = 0.05  # per word beyond max_words
    permutations: int = 2  # listener orders per description (cyclic shifts)
    reward: str = "prob"  # "prob" = p(correct letter), "acc" = argmax is correct

    max_overlap: float = 0.1  # same-image distractors: IoU with the target and each other below this

    def kinds(self) -> list[str]:
        return parse_kinds(self.distractors, self.choices - 1)

    def cross_task(self) -> InpaintingTask | None:
        """Inpainting's distractor miner for the cross-image slots (same-image slots are placed here, with less
        overlap than inpainting allows: overlapping crops share content and would be false negatives)."""
        cross = [k for k in self.kinds() if k != "same-image"]
        if not cross:
            return None
        return InpaintingTask(choices=len(cross) + 1, distractors=",".join(cross), oracle_index=self.oracle_index,
                              boundary_filter=None, color_select=self.color_select, select_by=self.select_by)


@dataclass
class Round:
    image: str
    box: Box
    speaker_view: Image.Image
    candidates: list[Image.Image]  # shuffled; candidates[answer] is the target
    answer: int
    meta: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------- rounds

def edge_energy(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)


def box_size(rng: random.Random, width: int, height: int, ratio: float, min_side: int,
             max_aspect: float = 2.0) -> tuple[int, int]:
    """w x h ~= ratio * W * H with aspect w/h log-uniform in [1/max_aspect, max_aspect] (no thin strips)."""
    area = ratio * width * height
    aspect = math.exp(rng.uniform(-math.log(max_aspect), math.log(max_aspect)))
    w = min(width, max(min_side, round(math.sqrt(area * aspect))))
    h = min(height, max(min_side, round(area / w)))
    return w, h


def salient_box(rng: random.Random, img: Image.Image, ratio: float, candidates: int, min_side: int) -> Box:
    """Among `candidates` random boxes of the same size, the one with the most edge energy (featureless sky or wall
    cannot be described apart from its neighbours)."""
    width, height = img.size
    w, h = box_size(rng, width, height, ratio, min_side)
    integral = np.pad(edge_energy(img).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, best_score = None, -1.0
    for _ in range(max(1, candidates)):
        x0, y0, x1, y1 = random_box(rng, width, height, w, h)
        score = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        if score > best_score:
            best, best_score = (x0, y0, x1, y1), float(score)
    return best


def resize_long(img: Image.Image, side: int) -> Image.Image:
    s = side / max(img.size)
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.Resampling.BICUBIC)


def draw_box(img: Image.Image, box: Box, width: int) -> Image.Image:
    out = img.copy()
    x0, y0, x1, y1 = box
    ImageDraw.Draw(out).rectangle((x0 - width, y0 - width, x1 + width - 1, y1 + width - 1), outline=(255, 0, 0),
                                  width=width)
    return out


def same_image_boxes(rng: random.Random, width: int, height: int, gt: Box, count: int, max_overlap: float) -> list[Box]:
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    for limit in (max_overlap, max(max_overlap, 0.25)):
        boxes: list[Box] = []
        for _ in range(300):
            box = random_box(rng, width, height, w, h)
            if all(iou(box, other) < limit for other in [gt, *boxes]):
                boxes.append(box)
                if len(boxes) == count:
                    return boxes
    raise ValueError("no room for distinct same-image regions")


def make_round(cfg: RefGameConfig, rng: random.Random, source: ImageSource) -> Round:
    kinds, cross = cfg.kinds(), cfg.cross_task()
    n_same = kinds.count("same-image")
    for _ in range(10):
        idx, img = source.sample(rng)
        try:
            box = salient_box(rng, img, rng.uniform(*cfg.ratio), cfg.salient_candidates, cfg.min_side)
            same = same_image_boxes(rng, img.width, img.height, box, n_same, cfg.max_overlap) if n_same else []
            if cross is not None:
                cross_patches, info = cross._distractors(rng, source, idx, img, box, box)
            else:
                cross_patches, info = [], {}
        except ValueError:
            continue
        break
    else:
        raise RuntimeError("refgame: no image admitted distinct distractors")
    same_it, cross_it, cross_src = iter(same), iter(cross_patches), iter(info.get("distractors", []))
    patches, sources = [], []
    for k in kinds:
        if k == "same-image":
            b = next(same_it)
            patches.append(img.crop(b)); sources.append({"kind": k, "box": list(b)})
        else:
            patches.append(next(cross_it)); sources.append(next(cross_src))
    truth = img.crop(box)
    order = list(range(len(patches) + 1))
    rng.shuffle(order)
    pool = [truth] + patches
    candidates = [resize_long(pool[i], cfg.crop_side) for i in order]
    meta = {"distractors": sources, "kinds": kinds}
    return Round(source.names[idx], box, draw_box(img, box, cfg.box_width), candidates, order.index(0), meta)


def step_rounds(cfg: RefGameConfig, source: ImageSource, seed: int, split: str, step: int, n: int) -> list[Round]:
    """Deterministic rounds: step s, slot i always gives the same round for a seed (resume-safe)."""
    return [make_round(cfg, random.Random(f"{seed}:refgame:{split}:{step}:{slot}"), source) for slot in range(n)]


# ------------------------------------------------------------------------------------------------ prompts

Content = list[dict[str, str]]  # chat content with {"type": "image"} placeholders, images passed separately


def speaker_item(cfg: RefGameConfig, rnd: Round) -> tuple[Content, list[Image.Image]]:
    text = SPEAKER_PROMPT.format(k=cfg.choices, d=cfg.choices - 1, w=cfg.max_words)
    return [{"type": "image"}, {"type": "text", "text": text}], [rnd.speaker_view]


def orders(cfg: RefGameConfig) -> list[list[int]]:
    """Cyclic shifts of the round's candidate order, spread evenly (P=2 with K=4: shifts 0 and 2)."""
    k = cfg.choices
    shifts = sorted({p * k // max(1, cfg.permutations) for p in range(max(1, cfg.permutations))})
    return [[(i + s) % k for i in range(k)] for s in shifts]


def listener_item(rnd: Round, desc: str, order: list[int]) -> tuple[Content, list[Image.Image], int]:
    """Listener prompt for one candidate order; returns the index of the right letter too."""
    content: Content = [{"type": "text", "text": f"Here are {len(order)} image crops.\n"}]
    images = []
    for slot, i in enumerate(order):
        content += [{"type": "text", "text": f"({LETTERS[slot]}) "}, {"type": "image"}, {"type": "text", "text": "\n"}]
        images.append(rnd.candidates[i])
    content.append({"type": "text", "text": LISTENER_PROMPT.format(desc=desc)})
    return content, images, order.index(rnd.answer)


def clean_description(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"^\s*(description|answer)\s*:\s*", "", t, flags=re.I)
    t = t.strip().splitlines()[0] if t.strip() else ""
    return t.strip().strip('"').strip()


def n_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def score(cfg: RefGameConfig, desc: str, probs: list[list[float]], targets: list[int]) -> dict[str, float]:
    """Reward from the listener's letter distributions (one per order) for one description."""
    p = statistics.mean(pr[t] for pr, t in zip(probs, targets))
    acc = statistics.mean(float(int(np.argmax(pr)) == t) for pr, t in zip(probs, targets))
    words = n_words(desc)
    base = p if cfg.reward == "prob" else acc
    reward = 0.0 if words == 0 else base - cfg.length_penalty * max(0, words - cfg.max_words)
    return {"reward": reward, "p_correct": p, "accuracy": acc, "words": float(words)}


# ---------------------------------------------------------------------------------- colour-only shortcut

COLOURS = {"black": (20, 20, 20), "white": (235, 235, 235), "grey": (128, 128, 128), "red": (200, 30, 30),
           "orange": (230, 130, 30), "yellow": (230, 210, 40), "green": (50, 150, 50), "blue": (40, 80, 200),
           "light blue": (140, 190, 235), "brown": (120, 80, 40), "pink": (230, 150, 180), "purple": (120, 50, 150),
           "beige": (215, 195, 160), "dark green": (25, 70, 30)}


def colour_description(crop: Image.Image) -> str:
    """'mostly X and Y': what a speaker that only reports colours could say (shortcut probe)."""
    px = np.asarray(crop.convert("RGB").resize((32, 32)), dtype=np.float32).reshape(-1, 3)
    names = list(COLOURS)
    ref = np.asarray([COLOURS[n] for n in names], dtype=np.float32)
    nearest = ((px[:, None] - ref[None]) ** 2).sum(-1).argmin(1)
    counts = np.bincount(nearest, minlength=len(names))
    top = [names[i] for i in counts.argsort()[::-1][:2] if counts[i] > 0]
    return "A region that is mostly " + " and ".join(top) + "."


# ---------------------------------------------------------------------------------------- model helpers

def chat_text(processor, content: Content) -> str:
    return processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)


def letter_token_ids(tokenizer, k: int) -> list[int]:
    ids = []
    for letter in LETTERS[:k]:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"letter {letter!r} is not one token: {enc}")
        ids.append(enc[0])
    return ids


def letter_probs(model, processor, items: list[tuple[Content, list[Image.Image]]], k: int, device="cuda",
                 batch: int = 8, autocast_dtype=None) -> list[list[float]]:
    """Listener: softmax over the K letter tokens at the first answer position, one forward pass per prompt."""
    import contextlib

    import torch

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tok.padding_side = "left"
    letter_ids = letter_token_ids(tok, k)
    out: list[list[float]] = []
    amp = (torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None
           else contextlib.nullcontext())
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        inputs = processor(text=[chat_text(processor, c) for c, _ in chunk],
                           images=[im for _, ims in chunk for im in ims], padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), amp:
            logits = model(**inputs).logits[:, -1, :]
        out += torch.softmax(logits[:, letter_ids].float(), -1).cpu().tolist()
    return out


def listen(cfg: RefGameConfig, model, processor, rounds: list[Round], descs: list[str], **kw) -> list[dict]:
    """Score one description per round (all orders batched together)."""
    items, targets, owner = [], [], []
    for j, (rnd, d) in enumerate(zip(rounds, descs)):
        for order in orders(cfg):
            content, images, t = listener_item(rnd, d or BLIND_DESC, order)
            items.append((content, images)); targets.append(t); owner.append(j)
    probs = letter_probs(model, processor, items, cfg.choices, **kw)
    per = [([], []) for _ in rounds]
    for j, pr, t in zip(owner, probs, targets):
        per[j][0].append(pr); per[j][1].append(t)
    return [score(cfg, d, pr, t) for d, (pr, t) in zip(descs, per)]


# --------------------------------------------------------------------------------------------- phase 0

def load_model(name: str, device: str):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(name)
    config = AutoConfig.from_pretrained(name)
    qc = getattr(config, "quantization_config", None)
    if qc is not None:  # unsloth bnb checkpoint: keep the vision tower unquantized (see kaggle/eval_adapter.py)
        skip = list((qc.get("llm_int8_skip_modules") if isinstance(qc, dict) else qc.llm_int8_skip_modules) or [])
        if "model.visual" not in skip:
            skip.append("model.visual")
            if isinstance(qc, dict):
                qc["llm_int8_skip_modules"] = skip
            else:
                qc.llm_int8_skip_modules = skip
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(name, config=config, device_map={"": device}, dtype=dtype)
    return model.eval(), processor


def generate(model, processor, items, n: int, max_new_tokens: int, temperature: float, device: str,
             batch: int = 8) -> list[list[str]]:
    """n samples per item (temperature 0 = greedy, n must be 1)."""
    import torch

    tok = processor.tokenizer
    tok.padding_side = "left"
    stops = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    kw.update(do_sample=False) if temperature == 0 else kw.update(do_sample=True, temperature=temperature, top_p=0.95)
    per = max(1, batch // n)
    out = []
    for i in range(0, len(items), per):
        chunk = items[i : i + per]
        inputs = processor(text=[chat_text(processor, c) for c, _ in chunk],
                           images=[im for _, ims in chunk for im in ims], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, num_return_sequences=n, **kw)
        plen = inputs["input_ids"].shape[1]
        texts = []
        for row in gen[:, plen:]:
            ids = row.tolist()
            cut = next((k for k, t in enumerate(ids) if t in stops), len(ids))
            texts.append(tok.decode(ids[:cut], skip_special_tokens=True))
        out += [texts[j * n : (j + 1) * n] for j in range(len(chunk))]
    return out


def contact_sheet(rounds: list[Round], descs: list[str], thumb: int = 160) -> Image.Image:
    """One row per round: speaker view, then the candidates (target framed green), description below."""
    rows = []
    for rnd, d in zip(rounds, descs):
        tiles = [resize_long(rnd.speaker_view, thumb * 2)] + [resize_long(c, thumb) for c in rnd.candidates]
        w = sum(t.width for t in tiles) + 10 * len(tiles)
        row = Image.new("RGB", (max(w, 900), thumb * 2 + 40), "white")
        x = 0
        for j, t in enumerate(tiles):
            row.paste(t, (x, 0))
            if j - 1 == rnd.answer:
                ImageDraw.Draw(row).rectangle((x, 0, x + t.width - 1, t.height - 1), outline=(0, 200, 0), width=4)
            x += t.width + 10
        ImageDraw.Draw(row).text((4, thumb * 2 + 8), d[:160], fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    return sheet


def phase0(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ImageSource(Path(args.images))
    model, processor = load_model(args.model, args.device)
    kw = dict(device=args.device, batch=args.batch)
    schemes = dict(s.split("=", 1) for s in args.scheme) or {"same": "{}"}
    summary = {}
    for name, spec in schemes.items():
        cfg = replace(RefGameConfig(), **json.loads(spec))
        t0 = time.time()
        rounds = step_rounds(cfg, source, args.seed, "val", 0, args.n)
        spk = [speaker_item(cfg, r) for r in rounds]
        greedy = [clean_description(g[0]) for g in generate(model, processor, spk, 1, args.max_new_tokens, 0.0,
                                                               **kw)]
        sampled = [[clean_description(t) for t in g] for g in
                   generate(model, processor, spk, args.group, args.max_new_tokens, 1.0, **kw)]
        colour = [colour_description(r.candidates[r.answer]) for r in rounds]
        res = {"blind": listen(cfg, model, processor, rounds, [""] * len(rounds), **kw),
               "colour": listen(cfg, model, processor, rounds, colour, **kw),
               "greedy": listen(cfg, model, processor, rounds, greedy, **kw)}
        flat = [d for g in sampled for d in g]
        rep = [r for r in rounds for _ in range(args.group)]
        s_flat = listen(cfg, model, processor, rep, flat, **kw)
        groups = [s_flat[i * args.group : (i + 1) * args.group] for i in range(len(rounds))]
        stds = [statistics.pstdev([s["reward"] for s in g]) for g in groups]
        row = {k: {m: statistics.mean(s[m] for s in v) for m in ("accuracy", "p_correct", "words")}
               for k, v in res.items()}
        row["sampled"] = {m: statistics.mean(s[m] for s in s_flat) for m in ("reward", "accuracy", "p_correct", "words")}
        row["sampled"].update(group_std=statistics.mean(stds),
                              groups_with_signal=sum(s > 0.05 for s in stds) / len(stds),
                              over_max_words=sum(s["words"] > cfg.max_words for s in s_flat) / len(s_flat))
        row["config"] = asdict(cfg)
        row["seconds"] = time.time() - t0
        summary[name] = row
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r, g, gs, c, rb, rc, rg, samp in zip(rounds, greedy, groups, colour, res["blind"], res["colour"],
                                                      res["greedy"], sampled):
                f.write(json.dumps({"image": r.image, "box": r.box, "answer": r.answer, "meta": r.meta,
                                    "greedy": g, "greedy_score": rg, "colour": c, "colour_score": rc,
                                    "blind_score": rb, "samples": samp, "sample_scores": gs}, default=str) + "\n")
        contact_sheet(rounds[: args.sheet], greedy[: args.sheet]).save(out_dir / f"{name}_sheet.jpg", quality=85)
        print(name, json.dumps({k: v for k, v in row.items() if k != "config"}, indent=1), flush=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out_dir / "summary.json")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("phase0", help="base-model checks before training")
    q.add_argument("--images", required=True, help="val2017 directory or ZIP")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    q.add_argument("--device", default="cuda")
    q.add_argument("--n", type=int, default=200)
    q.add_argument("--group", type=int, default=8, help="sampled descriptions per round (GRPO group)")
    q.add_argument("--max-new-tokens", type=int, default=64)
    q.add_argument("--batch", type=int, default=16)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--sheet", type=int, default=12, help="rounds on the preview sheet")
    q.add_argument("--scheme", action="append", default=[], help="name=JSON overrides of RefGameConfig")
    args = p.parse_args(argv)
    if args.cmd == "phase0":
        phase0(args)


if __name__ == "__main__":
    main()
