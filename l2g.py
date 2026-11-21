"""Local-to-global caption game: from a small crop, guess the caption of the whole image.

Borrowed from DINO's multi-crop self-distillation, moved into the VLM's text output. The frozen base model captions
each whole image once, with an open prompt and no length limit; that caption is the fixed reference answer. The
policy sees only a small crop (10-30% of the image) and writes

    Visible: <what the crop shows, <= 25 words>
    Scene: <a guess at the whole photo, <= 50 words>

Scoring (design doc https://claude.ai/artifact/UBfwevBPjHPdm549hBkWiU):

  * the reference is split once into atomic facts tagged [subject] / [scene] / [detail] by a small text judge
    (~4B), and each fact is checked against the whole image by the frozen VLM (facts it rejects are dropped:
    long base captions contain the base model's own hallucinations);
  * judge recall: per fact, the judge's p(Yes) that the guess commits to it (hedges and "A or B" do not count);
    facts are weighted by type (subject/scene 1.0, detail 0.3) x (1 - p_blind), where p_blind is how often
    guesses made from a crop of a *different* image hit that fact (common-sense facts such as "indoors" or "a
    person" earn little);
  * judge precision / contradictions: the judge splits the guess into claims and labels each SUPPORTED,
    CONTRADICTED or UNSUPPORTED by the reference;
  * Visible precision: the frozen VLM's p(Yes) that the Visible line is accurate for the crop;
  * sentence-embedding similarity (comparison score): mean over reference sentences of the best cosine with a
    Scene sentence (bge-base);
  * retrieval (independent check): SigLIP picks the true image among it and 7 look-alikes (closest COCO label
    histograms) from the Scene text.

Phase 0 runs the base model only (no training) and reports, per crop scale: these scores for greedy and sampled
guesses, GRPO signal (reward spread within a crop's group), and the baselines (crop of another image, no image,
Visible only, a fixed generic sentence, the reference itself); plus judge-vs-embedding agreement and a sheet of 50
guesses for human labelling.

  python -m serverless.l2g phase0 --images /path/val2017 --index val2017_labels.npz --out-dir out/l2g_phase0 \\
      --n 200 --scales 0.1,0.2,0.3 --group 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from serverless.images import ImageSource
from serverless.refgame import box_size, chat_text, resize_long
from serverless.tasks.inpainting import random_box

Box = tuple[int, int, int, int]


def generate(model, processor, items, n: int, max_new_tokens: int, temperature: float, device: str,
             batch: int = 8) -> list[list[str]]:
    """refgame.generate (n samples per item via num_return_sequences), plus: text-only items (no images), and a
    one-by-one fallback when a VLM's generate cannot expand pixel inputs for num_return_sequences > 1."""
    import torch

    tok = processor.tokenizer
    tok.padding_side = "left"
    stops = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    kw.update(do_sample=False) if temperature == 0 else kw.update(do_sample=True, temperature=temperature, top_p=0.95)

    def decode(gen, plen):
        texts = []
        for row in gen[:, plen:]:
            ids = row.tolist()
            cut = next((k for k, t in enumerate(ids) if t in stops), len(ids))
            texts.append(tok.decode(ids[:cut], skip_special_tokens=True))
        return texts

    per = max(1, batch // n)
    out = []
    t0 = time.time()
    for i in range(0, len(items), per):
        chunk = items[i : i + per]
        ims = [im for _, ims in chunk for im in ims]
        enc = dict(text=[chat_text(processor, c) for c, _ in chunk], padding=True, return_tensors="pt")
        if ims:
            enc["images"] = ims
        inputs = processor(**enc).to(device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            try:
                texts = decode(model.generate(**inputs, num_return_sequences=n, **kw), plen)
            except Exception as e:  # noqa: BLE001
                if n == 1:
                    raise
                print("batched generate failed, sampling one at a time:", repr(e)[:200], flush=True)
                runs = [decode(model.generate(**inputs, **kw), plen) for _ in range(n)]
                texts = [runs[s][j] for j in range(len(chunk)) for s in range(n)]
        out += [texts[j * n : (j + 1) * n] for j in range(len(chunk))]
        if (i // per) % 20 == 0:
            print(f"  generate {min(i + per, len(items))}/{len(items)} items x {n} ({time.time() - t0:.0f}s)",
                  flush=True)
    return out

REF_PROMPT = "Describe this image in detail."
POLICY_PROMPT = (
    "This is a small crop cut out of a larger photo. First describe what is visible in the crop. Then guess what "
    "the whole photo shows: the main subjects, the setting, and what is happening. Use exactly this format:\n"
    "Visible: <at most 25 words>\nScene: <at most 50 words>"
)
NO_IMAGE_PROMPT = (
    "You cannot see the photo. Guess what a typical everyday photo shows: the main subjects, the setting, and what "
    "is happening. Use exactly this format:\nVisible: nothing\nScene: <at most 50 words>"
)
GENERIC_SCENE = "A photo of an everyday scene with people and objects in an indoor or outdoor setting."

FACT_PROMPT = (
    "Below is a detailed description of a photo. Split it into short atomic facts, one per line. Each fact states "
    "one thing: an object, an attribute, a relation, an action, or the setting. Start each line with a tag: "
    "[subject] for the main subjects and what they are doing, [scene] for the place, setting, time or overall "
    "situation, [detail] for colours, text, counts, small objects and other fine details. Write at most {k} facts, "
    "the most important first. Do not add anything that is not in the description. Reply with the lines only.\n\n"
    "Description:\n{ref}"
)
VERIFY_PROMPT = 'Is this statement about the image true? "{fact}" Answer Yes or No.'
HIT_PROMPT = (
    "A guess about a photo:\n\"{guess}\"\n\nFact: \"{fact}\"\n\nDoes the guess state this fact, or something that "
    "clearly implies it? Synonyms and more specific wording count (\"a golden retriever\" implies \"a dog\"). The "
    "guess must commit to it: alternatives such as \"a dog or a cat\" and hedges such as \"possibly\" or \"maybe\" "
    "do not count. Answer Yes or No."
)
CLAIM_PROMPT = (
    "Reference description of a photo:\n\"\"\"{ref}\"\"\"\n\nA guess about the same photo:\n\"{guess}\"\n\n"
    "Split the guess into its separate claims, one per line, and end each line with exactly one label: "
    "[SUPPORTED] if the reference states or clearly implies it, [CONTRADICTED] if the reference says otherwise, "
    "[UNSUPPORTED] if the reference does not mention it. Reply with the lines only."
)
VISIBLE_PROMPT = (
    'Is this description of the image accurate, with nothing in it that is not visible? "{visible}" '
    "Answer Yes or No."
)

TYPE_WEIGHT = {"subject": 1.0, "scene": 1.0, "detail": 0.3}
UNTAGGED_WEIGHT = 0.5


@dataclass
class L2GConfig:
    scales: tuple[float, ...] = (0.1, 0.2, 0.3)  # crop area as a share of the image
    blind_scale: float = 0.2  # crop scale for the other-image baseline
    min_side: int = 64
    crop_tries: int = 16  # candidate boxes per crop; keep the most salient one that shows a COCO thing
    min_thing_share: float = 0.05  # share of the crop's label cells that must be a COCO thing (index needed)
    crop_side: int = 448  # crops are resized so the longer side is this
    max_facts: int = 30
    verify_threshold: float = 0.5
    max_visible_words: int = 25
    max_scene_words: int = 50
    length_penalty: float = 0.02  # per Scene word beyond max_scene_words
    format_penalty: float = 0.5  # Visible/Scene lines missing
    w_precision: float = 0.3
    w_contra: float = 0.5
    w_visible: float = 0.2
    negatives: int = 7  # retrieval look-alikes per image


# ------------------------------------------------------------------------------------------------- text utils

def n_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_guess(text: str) -> dict[str, Any]:
    """'Visible: ... / Scene: ...' -> fields; format_ok when both lines are there and non-empty."""
    t = strip_think(text)
    vis = re.search(r"visible\s*:\s*(.*?)(?=\n\s*scene\s*:|$)", t, flags=re.I | re.S)
    scn = re.search(r"scene\s*:\s*(.*)", t, flags=re.I | re.S)
    visible = " ".join(vis.group(1).split()) if vis else ""
    scene = " ".join(scn.group(1).split()) if scn else ""
    return {"raw": text, "visible": visible, "scene": scene, "format_ok": bool(visible and scene)}


def parse_facts(text: str, k: int) -> list[dict[str, str]]:
    facts = []
    for line in strip_think(text).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        m = re.match(r"\[(subject|scene|detail)\]\s*[:\-]?\s*(.+)", line, flags=re.I)
        kind, fact = (m.group(1).lower(), m.group(2).strip()) if m else ("untagged", line)
        if n_words(fact) >= 2:
            facts.append({"type": kind, "fact": fact})
    return facts[:k]


def parse_claims(text: str) -> dict[str, int]:
    counts = {"supported": 0, "contradicted": 0, "unsupported": 0}
    for line in strip_think(text).splitlines():
        m = re.search(r"\[(SUPPORTED|CONTRADICTED|UNSUPPORTED)\]", line, flags=re.I)
        if m:
            counts[m.group(1).lower()] += 1
    return counts


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if n_words(s) >= 3]
    return parts or ([text.strip()] if text.strip() else [])


# ---------------------------------------------------------------------------------------------------- crops

def thing_share(index, name: str, box: Box) -> float | None:
    """Share of the box's label cells that are COCO things (ids 1-90); None without an index entry."""
    if index is None:
        return None
    from serverless.hardneg import cell_box, histogram

    grid = index.grid(name)
    if grid is None:
        return None
    h = histogram(grid, cell_box(box))
    return float(h[1:91].sum())


def pick_crop(rng: random.Random, img: Image.Image, scale: float, cfg: L2GConfig, index=None, name: str = "") -> Box:
    """Most edge-energetic of `crop_tries` random boxes of area `scale`, preferring boxes that show a COCO thing."""
    from serverless.refgame import edge_energy

    width, height = img.size
    w, h = box_size(rng, width, height, scale, cfg.min_side)
    integral = np.pad(edge_energy(img).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, best_key = None, None
    for _ in range(cfg.crop_tries):
        x0, y0, x1, y1 = random_box(rng, width, height, w, h)
        energy = float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
        share = thing_share(index, name, (x0, y0, x1, y1))
        ok = share is None or share >= cfg.min_thing_share
        key = (ok, energy)
        if best_key is None or key > best_key:
            best, best_key = (x0, y0, x1, y1), key
    return best


def label_hist(index, name: str) -> np.ndarray | None:
    from serverless.hardneg import histogram

    grid = None if index is None else index.grid(name)
    if grid is None:
        return None
    return histogram(grid, (0, 0, grid.shape[1], grid.shape[0]))


def lookalikes(index, names: list[str], pool: list[str], k: int, rng: random.Random) -> dict[str, list[str]]:
    """For each image, the k pool images with the closest whole-image label histogram (random without an index)."""
    from serverless.hardneg import js_divergence

    hists = {n: label_hist(index, n) for n in set(names) | set(pool)}
    pool_ok = [n for n in pool if hists[n] is not None]
    mat = np.stack([hists[n] for n in pool_ok]) if pool_ok else None
    out = {}
    for n in names:
        if hists[n] is None or mat is None:
            others = [p for p in pool if p != n]
            out[n] = rng.sample(others, min(k, len(others)))
            continue
        d = js_divergence(hists[n], mat)
        order = [pool_ok[i] for i in np.argsort(d) if pool_ok[i] != n]
        out[n] = order[:k]
    return out


# ---------------------------------------------------------------------------------------------- model helpers

def load_vlm(name: str, device: str, loader: str):
    """Frozen VLM. 'unsloth' is the path the Kaggle trainer uses on T4s (fp16-safe); 'hf' is plain transformers."""
    if loader == "unsloth":
        import os

        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")  # the Yes/No scores read logits from forward()
        import unsloth  # noqa: F401  (must come before transformers)
        from unsloth import FastVisionModel

        model, processor = FastVisionModel.from_pretrained(name, load_in_4bit=True, device_map={"": device})
        FastVisionModel.for_inference(model)
        return model, processor
    from serverless.refgame import load_model

    return load_model(name, device)


def load_text_model(name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map={"": device})
    return model.eval(), tok


def judge_text(tok, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def yes_no_ids(tok) -> tuple[int, int]:
    ids = []
    for word in ("Yes", "No"):
        enc = tok.encode(word, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"{word!r} is not one token: {enc}")
        ids.append(enc[0])
    return ids[0], ids[1]


def text_yes_probs(model, tok, prompts: list[str], device: str, batch: int) -> list[float]:
    """p(Yes) / (p(Yes) + p(No)) at the first answer position of a text-only judge."""
    import torch

    y, n = yes_no_ids(tok)
    out = []
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        enc = tok([judge_text(tok, p) for p in prompts[i : i + batch]], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        out += torch.softmax(logits[:, [y, n]], -1)[:, 0].cpu().tolist()
        if (i // batch) % 50 == 0:
            print(f"  text_yes_probs {i + batch}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    return [p if math.isfinite(p) else 0.5 for p in out]


def text_generate(model, tok, prompts: list[str], max_new_tokens: int, device: str, batch: int) -> list[str]:
    import torch

    out = []
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        enc = tok([judge_text(tok, p) for p in prompts[i : i + batch]], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if (i // batch) % 50 == 0:
            print(f"  text_generate {i + batch}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def vlm_yes_probs(model, processor, items: list[tuple[str, Image.Image]], device: str, batch: int) -> list[float]:
    """p(Yes) / (p(Yes) + p(No)) for (question, image) pairs on the frozen VLM."""
    import torch

    tok = processor.tokenizer
    tok.padding_side = "left"
    y, n = yes_no_ids(tok)
    out = []
    t0 = time.time()
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        texts = [chat_text(processor, [{"type": "image"}, {"type": "text", "text": q}]) for q, _ in chunk]
        inputs = processor(text=texts, images=[im for _, im in chunk], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float()
        out += torch.softmax(logits[:, [y, n]], -1)[:, 0].cpu().tolist()
        if (i // batch) % 50 == 0:
            print(f"  vlm_yes_probs {i + batch}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    return [p if math.isfinite(p) else 0.5 for p in out]


class Embedder:
    """Sentence embeddings with CLS pooling (bge)."""

    def __init__(self, name: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device = device

    def __call__(self, texts: list[str], batch: int = 64) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i : i + batch], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                cls = self.model(**enc).last_hidden_state[:, 0].float()
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))


def emb_similarity(embed: Embedder, refs: list[str], guesses: list[str]) -> list[float]:
    """Mean over reference sentences of the best cosine with any guess sentence."""
    ref_s = [sentences(r) for r in refs]
    gss_s = [sentences(g) for g in guesses]
    flat = sorted({s for ss in ref_s + gss_s for s in ss})
    vec = dict(zip(flat, embed(flat))) if flat else {}
    out = []
    for rs, gs in zip(ref_s, gss_s):
        if not rs or not gs:
            out.append(0.0)
            continue
        sim = np.stack([vec[s] for s in rs]) @ np.stack([vec[s] for s in gs]).T
        out.append(float(sim.max(1).mean()))
    return out


def _features(out):
    """get_*_features returns a tensor in transformers 4.x and an output object (pooler_output) in 5.x."""
    import torch

    if torch.is_tensor(out):
        return out
    for key in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, key, None)
        if torch.is_tensor(v):
            return v
    raise TypeError(f"no feature tensor in {type(out).__name__}")


class Retriever:
    """SigLIP text -> image retrieval among the true image and its look-alikes."""

    def __init__(self, name: str, device: str):
        from transformers import AutoModel, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device = device
        self.image_vecs: dict[str, np.ndarray] = {}

    def add_images(self, named: list[tuple[str, Image.Image]], batch: int = 32) -> None:
        import torch

        for i in range(0, len(named), batch):
            chunk = named[i : i + batch]
            inp = self.proc(images=[im for _, im in chunk], return_tensors="pt").to(self.device)
            with torch.no_grad():
                v = _features(self.model.get_image_features(**inp)).float()
            v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
            self.image_vecs.update({n: x for (n, _), x in zip(chunk, v)})

    def text_vecs(self, texts: list[str], batch: int = 64) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(texts), batch):
            inp = self.proc(text=[t or "." for t in texts[i : i + batch]], padding="max_length", max_length=64,
                            truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                v = _features(self.model.get_text_features(**inp)).float()
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def score(self, texts: list[str], truths: list[str], negatives: dict[str, list[str]]) -> list[dict[str, float]]:
        tv = self.text_vecs(texts)
        out = []
        for t, truth in zip(tv, truths):
            cands = [truth] + [n for n in negatives[truth] if n in self.image_vecs]
            sims = np.stack([self.image_vecs[c] for c in cands]) @ t
            logits = sims * 100.0  # CLIP-style temperature; only the ranking matters for top-1
            p = np.exp(logits - logits.max())
            p /= p.sum()
            out.append({"retr_top1": float(int(np.argmax(sims)) == 0), "retr_p": float(p[0])})
        return out


# ------------------------------------------------------------------------------------------------- scoring

def reward_of(cfg: L2GConfig, s: dict[str, float]) -> float:
    r = s["wrecall"] + cfg.w_precision * s["precision"] - cfg.w_contra * s["contra"] + cfg.w_visible * s["vis_prec"]
    r -= cfg.length_penalty * max(0, s["words"] - cfg.max_scene_words)
    if not s["format_ok"]:
        r -= cfg.format_penalty
    return r


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float("nan")


# ------------------------------------------------------------------------------------------------- phase 0

def jsonl_cache(path: Path, build):
    """Load a stage's rows if its file exists (resume), else build and save them."""
    if path.exists():
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"loaded {len(rows)} rows from {path}", flush=True)
        return rows
    t0 = time.time()
    rows = build()
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"wrote {len(rows)} rows to {path} in {time.time() - t0:.0f}s", flush=True)
    return rows


def phase0(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = L2GConfig(scales=tuple(float(s) for s in args.scales.split(",")))
    (out / "config.json").write_text(json.dumps({"args": vars(args), "cfg": asdict(cfg)}, indent=1, default=str))
    source = ImageSource(Path(args.images))
    index = None
    if args.index:
        from serverless.hardneg import LabelIndex

        index = LabelIndex.load(args.index)
    rng = random.Random(args.seed)
    usable = [n for n in source.names if index is None or index.grid(n) is not None]
    names = sorted(rng.sample(usable, min(args.n, len(usable))))
    images = {n: source.load(source.index_of(n)) for n in names}
    vlm_dev, judge_dev = args.vlm_device, args.judge_device
    if judge_dev.startswith("cuda:") and judge_dev != vlm_dev:
        import torch

        if torch.cuda.device_count() <= int(judge_dev.split(":")[1]):
            print(f"only {torch.cuda.device_count()} GPU(s): judge models go on {vlm_dev}", flush=True)
            judge_dev = vlm_dev
    gkw = dict(device=vlm_dev, batch=args.gen_batch)
    G = args.group

    model, processor = load_vlm(args.model, vlm_dev, args.loader)

    # 1. reference captions: open prompt, no length instruction (max_ref_tokens is only a compute cap).
    def build_refs():
        items = [([{"type": "image"}, {"type": "text", "text": REF_PROMPT}], [images[n]]) for n in names]
        caps = generate(model, processor, items, 1, args.max_ref_tokens, 0.0, **gkw)
        return [{"image": n, "reference": strip_think(c[0])} for n, c in zip(names, caps)]

    refs = {r["image"]: r["reference"] for r in jsonl_cache(out / "references.jsonl", build_refs)}

    # 2. crops: one per image and scale, plus the other-image crop used as the blind baseline.
    crops: dict[tuple[str, float], Box] = {}
    for n in names:
        for s in cfg.scales:
            crops[(n, s)] = pick_crop(random.Random(f"{args.seed}:l2g:{n}:{s}"), images[n], s, cfg, index, n)
    other = {n: names[(i + 1 + len(names) // 2) % len(names)] for i, n in enumerate(names)}  # a far, fixed partner
    blind_crop = {n: pick_crop(random.Random(f"{args.seed}:l2g:{other[n]}:blind"), images[other[n]],
                               cfg.blind_scale, cfg, index, other[n]) for n in names}

    def crop_img(name: str, box: Box) -> Image.Image:
        return resize_long(images[name].crop(box), cfg.crop_side)

    # 3. policy samples (base model): greedy + G sampled per crop; blind: G per image from the other image's crop;
    #    no image: 2 per image.
    def build_samples():
        rows = []
        pol = [([{"type": "image"}, {"type": "text", "text": POLICY_PROMPT}], [crop_img(n, crops[(n, s)])])
               for n in names for s in cfg.scales]
        keys = [(n, s) for n in names for s in cfg.scales]
        greedy = generate(model, processor, pol, 1, args.max_new_tokens, 0.0, **gkw)
        sampled = generate(model, processor, pol, G, args.max_new_tokens, 1.0, **gkw)
        for (n, s), g, ss in zip(keys, greedy, sampled):
            rows.append({"image": n, "kind": "greedy", "scale": s, "box": crops[(n, s)], **parse_guess(g[0])})
            rows += [{"image": n, "kind": "sampled", "scale": s, "box": crops[(n, s)], "slot": j, **parse_guess(t)}
                     for j, t in enumerate(ss)]
        blind_items = [([{"type": "image"}, {"type": "text", "text": POLICY_PROMPT}],
                        [crop_img(other[n], blind_crop[n])]) for n in names]
        blind = generate(model, processor, blind_items, G, args.max_new_tokens, 1.0, **gkw)
        for n, ss in zip(names, blind):
            rows += [{"image": n, "kind": "other_crop", "scale": cfg.blind_scale, "box": blind_crop[n],
                      "crop_of": other[n], "slot": j, **parse_guess(t)} for j, t in enumerate(ss)]
        noimg = generate(model, processor, [([{"type": "text", "text": NO_IMAGE_PROMPT}], [])] * len(names), 2,
                         args.max_new_tokens, 1.0, **gkw)
        for n, ss in zip(names, noimg):
            rows += [{"image": n, "kind": "no_image", "slot": j, **parse_guess(t)} for j, t in enumerate(ss)]
        return rows

    samples = jsonl_cache(out / "samples.jsonl", build_samples)

    # 4. Visible precision on the crop (frozen VLM), for rows that have a crop.
    def build_vis():
        idx = [i for i, r in enumerate(samples) if r["kind"] in ("greedy", "sampled", "other_crop")]
        items = []
        for i in idx:
            r = samples[i]
            src = r.get("crop_of", r["image"])
            items.append((VISIBLE_PROMPT.format(visible=r["visible"] or "(empty)"), crop_img(src, tuple(r["box"]))))
        probs = vlm_yes_probs(model, processor, items, vlm_dev, args.score_batch)
        return [{"row": i, "vis_prec": p if samples[i]["visible"] else 0.0} for i, p in zip(idx, probs)]

    vis = {v["row"]: v["vis_prec"] for v in jsonl_cache(out / "visible.jsonl", build_vis)}

    # 5. facts: judge splits each reference; the VLM checks each fact on the whole image.
    judge, jtok = load_text_model(args.judge, judge_dev)

    def build_facts():
        raw = text_generate(judge, jtok, [FACT_PROMPT.format(k=cfg.max_facts, ref=refs[n]) for n in names],
                            args.max_fact_tokens, judge_dev, args.judge_gen_batch)
        per = {n: parse_facts(t, cfg.max_facts) for n, t in zip(names, raw)}
        flat = [(n, f) for n in names for f in per[n]]
        probs = vlm_yes_probs(model, processor, [(VERIFY_PROMPT.format(fact=f["fact"]), images[n]) for n, f in flat],
                              vlm_dev, args.score_batch)
        for (n, f), p in zip(flat, probs):
            f["verify_p"] = p
            f["kept"] = p >= cfg.verify_threshold
        return [{"image": n, "facts": per[n], "raw": t} for n, t in zip(names, raw)]

    facts = {r["image"]: [f for f in r["facts"] if f["kept"]] for r in jsonl_cache(out / "facts.jsonl", build_facts)}
    del model
    _free()

    # 6. judge hits: p(Yes) per (guess, fact). Baseline texts: Visible-only (greedy rows), generic, reference.
    texts: list[tuple[str, str, str]] = []  # (image, label, scene text)
    for i, r in enumerate(samples):
        texts.append((r["image"], f"row{i}", r["scene"]))
    for i, r in enumerate(samples):
        if r["kind"] == "greedy":
            texts.append((r["image"], f"visonly{i}", r["visible"]))
    for n in names:
        texts.append((n, "generic", GENERIC_SCENE))
        texts.append((n, "reference", refs[n]))

    def build_hits():
        pairs = [(k, j, HIT_PROMPT.format(guess=t or "(empty)", fact=f["fact"]))
                 for k, (n, _, t) in enumerate(texts) for j, f in enumerate(facts[n])]
        probs = text_yes_probs(judge, jtok, [p for _, _, p in pairs], judge_dev, args.judge_batch)
        hit: dict[int, list[float]] = {}
        for (k, j, _), p in zip(pairs, probs):
            hit.setdefault(k, [0.0] * len(facts[texts[k][0]]))[j] = p
        return [{"image": n, "label": lab, "hits": hit.get(k, [])} for k, (n, lab, _) in enumerate(texts)]

    hits = {(r["image"], r["label"]): r["hits"] for r in jsonl_cache(out / "hits.jsonl", build_hits)}

    # 7. claims of the guess vs the reference (precision, contradictions).
    def build_claims():
        prompts = [CLAIM_PROMPT.format(ref=refs[n], guess=t or "(empty)") for n, _, t in texts]
        raw = text_generate(judge, jtok, prompts, args.max_claim_tokens, judge_dev, args.judge_gen_batch)
        return [{"image": n, "label": lab, **parse_claims(c), "raw": c} for (n, lab, _), c in zip(texts, raw)]

    claims = {(r["image"], r["label"]): r for r in jsonl_cache(out / "claims.jsonl", build_claims)}
    del judge
    _free()

    # 8. p_blind per fact from the other-image crops, then weights.
    blind_hits: dict[str, list[list[float]]] = {}
    for i, r in enumerate(samples):
        if r["kind"] == "other_crop":
            blind_hits.setdefault(r["image"], []).append(hits[(r["image"], f"row{i}")])
    weights = {}
    for n in names:
        hs = blind_hits.get(n, [])
        pb = [mean(h[j] for h in hs) if hs else 0.0 for j in range(len(facts[n]))]
        weights[n] = [TYPE_WEIGHT.get(f["type"], UNTAGGED_WEIGHT) * (1 - p) for f, p in zip(facts[n], pb)]
        for f, p, w in zip(facts[n], pb, weights[n]):
            f["p_blind"], f["weight"] = p, w

    # 9. embeddings and retrieval.
    embed = Embedder(args.embedder, judge_dev)
    emb = dict(zip([(n, lab) for n, lab, _ in texts],
                   emb_similarity(embed, [refs[n] for n, _, _ in texts], [t for _, _, t in texts])))
    del embed
    retr = {}
    if not args.no_retrieval:
        retriever = Retriever(args.retriever, judge_dev)
        pool = sorted(set(names) | set(random.Random(args.seed + 1).sample(usable, min(args.pool, len(usable)))))
        negs = lookalikes(index, names, pool, cfg.negatives, random.Random(args.seed))
        need = sorted(set(names) | {m for n in names for m in negs[n]})
        for i in range(0, len(need), 256):
            retriever.add_images([(m, source.load(source.index_of(m))) for m in need[i : i + 256]])
        rs = retriever.score([t for _, _, t in texts], [n for n, _, _ in texts], negs)
        retr = {(n, lab): r for (n, lab, _), r in zip(texts, rs)}
        (out / "lookalikes.json").write_text(json.dumps(negs, indent=1))
        del retriever
    _free()

    # 10. per-text scores and the reward.
    def scores(n: str, lab: str, text: str, fmt_ok: bool, vis_p: float) -> dict[str, float]:
        h, w = hits[(n, lab)], weights[n]
        c = claims[(n, lab)]
        n_claims = c["supported"] + c["contradicted"] + c["unsupported"]
        s = {
            "recall": mean(h) if h else 0.0,
            "wrecall": float(np.dot(h, w) / sum(w)) if h and sum(w) > 0 else 0.0,
            "precision": c["supported"] / n_claims if n_claims else 0.0,
            "contra": c["contradicted"] / n_claims if n_claims else 0.0,
            "n_claims": float(n_claims),
            "vis_prec": vis_p,
            "emb": emb[(n, lab)],
            "words": float(n_words(text)),
            "format_ok": fmt_ok,
        }
        s.update(retr.get((n, lab), {}))
        s["reward"] = reward_of(cfg, s)
        return s

    scored = []
    for i, r in enumerate(samples):
        scored.append({**r, **scores(r["image"], f"row{i}", r["scene"], r["format_ok"], vis.get(i, 0.0))})
    extra = []
    for i, r in enumerate(samples):
        if r["kind"] == "greedy":
            extra.append({"image": r["image"], "kind": "visible_only", "scale": r["scale"], "scene": r["visible"],
                          **scores(r["image"], f"visonly{i}", r["visible"], True, 0.0)})
    for n in names:
        extra.append({"image": n, "kind": "generic", "scene": GENERIC_SCENE,
                      **scores(n, "generic", GENERIC_SCENE, True, 0.0)})
        extra.append({"image": n, "kind": "reference", "scene": refs[n], **scores(n, "reference", refs[n], True, 0.0)})
    with open(out / "scores.jsonl", "w") as f:
        for r in scored + extra:
            f.write(json.dumps(r, default=str) + "\n")
    with open(out / "facts_weighted.jsonl", "w") as f:
        for n in names:
            f.write(json.dumps({"image": n, "reference": refs[n], "facts": facts[n]}) + "\n")

    summary = summarize(cfg, scored + extra, refs, facts, G)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    label_sheet(out, scored, refs, images, args.label_n, args.seed, cfg)
    print("wrote", out / "summary.json")


METRICS = ("reward", "wrecall", "recall", "precision", "contra", "vis_prec", "emb", "retr_top1", "retr_p", "words",
           "format_ok")


def summarize(cfg: L2GConfig, rows: list[dict], refs: dict, facts: dict, G: int) -> dict:
    def agg(rs):
        return {m: mean(float(r[m]) for r in rs if m in r) for m in METRICS} | {"n": len(rs)}

    summ: dict[str, Any] = {"facts": {
        "per_image": mean(len(v) for v in facts.values()),
        "ref_words": mean(n_words(r) for r in refs.values()),
        "share_by_type": {t: mean(sum(f["type"] == t for f in v) / max(1, len(v)) for v in facts.values())
                          for t in ("subject", "scene", "detail", "untagged")},
        "mean_p_blind": mean(f["p_blind"] for v in facts.values() for f in v),
    }}
    for kind in ("other_crop", "no_image", "generic", "reference"):
        summ[kind] = agg([r for r in rows if r["kind"] == kind])
    for s in cfg.scales:
        key = f"scale_{s:g}"
        g = [r for r in rows if r["kind"] == "greedy" and r.get("scale") == s]
        smp = [r for r in rows if r["kind"] == "sampled" and r.get("scale") == s]
        vo = [r for r in rows if r["kind"] == "visible_only" and r.get("scale") == s]
        groups: dict[str, list[dict]] = {}
        for r in smp:
            groups.setdefault(r["image"], []).append(r)
        stds = [statistics.pstdev([x["reward"] for x in grp]) for grp in groups.values() if len(grp) > 1]
        emb_stds = [statistics.pstdev([x["emb"] for x in grp]) for grp in groups.values() if len(grp) > 1]
        within = [spearman([x["reward"] for x in grp], [x["emb"] for x in grp]) for grp in groups.values()]
        summ[key] = {
            "greedy": agg(g), "sampled": agg(smp), "visible_only": agg(vo),
            "group_reward_std": mean(stds), "groups_with_signal": mean(float(x > 0.05) for x in stds),
            "group_emb_std": mean(emb_stds),
            "spearman_wrecall_emb_all": spearman([r["wrecall"] for r in smp], [r["emb"] for r in smp]),
            "spearman_reward_emb_within_group": mean(within),
            "gap_vs_other_crop": agg(smp)["wrecall"] - summ["other_crop"]["wrecall"],
            "gap_vs_visible_only": agg(smp)["wrecall"] - agg(vo)["wrecall"],
        }
    return summ


def label_sheet(out: Path, scored: list[dict], refs: dict, images: dict, n: int, seed: int, cfg: L2GConfig) -> None:
    """CSV + crops of n random sampled guesses for human recall/precision labels (judge calibration)."""
    rows = [r for r in scored if r["kind"] == "sampled"]
    pick = random.Random(seed + 2).sample(rows, min(n, len(rows)))
    d = out / "label_crops"
    d.mkdir(exist_ok=True)
    with open(out / "label_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "scale", "crop_file", "reference", "visible", "scene", "judge_wrecall",
                    "judge_recall", "judge_precision", "judge_contra", "emb", "human_recall_0to1",
                    "human_contradiction_yes_no", "notes"])
        for i, r in enumerate(pick):
            fn = f"{i:02d}_{Path(r['image']).stem}_{r['scale']:g}.jpg"
            resize_long(images[r["image"]].crop(tuple(r["box"])), cfg.crop_side).save(d / fn, quality=90)
            w.writerow([i, r["image"], r["scale"], fn, refs[r["image"]], r["visible"], r["scene"],
                        f"{r['wrecall']:.3f}", f"{r['recall']:.3f}", f"{r['precision']:.3f}", f"{r['contra']:.3f}",
                        f"{r['emb']:.3f}", "", "", ""])


def rescore_outcrop(args: argparse.Namespace) -> None:
    """Phase 0b: which reference facts are visible in each crop? Re-score from phase-0 caches with recall over the
    facts the crop does NOT show (owrecall), so copying the visible content into Scene earns nothing.

    Needs references/samples/hits/scores/facts_weighted from `phase0` in --out-dir; adds crop_facts.jsonl and
    summary_outcrop.json. One VLM yes/no per (crop, fact)."""
    out = Path(args.out_dir)
    cfg = L2GConfig()
    load = lambda name: [json.loads(l) for l in (out / name).read_text().splitlines() if l.strip()]  # noqa: E731
    samples, scores_rows = load("samples.jsonl"), load("scores.jsonl")
    facts = {r["image"]: r["facts"] for r in load("facts_weighted.jsonl")}
    hits = {(r["image"], r["label"]): r["hits"] for r in load("hits.jsonl")}
    scored = scores_rows[: len(samples)]
    source = ImageSource(Path(args.images))

    crops = sorted({(r["image"], r["image"], tuple(r["box"])) for r in samples
                    if "box" in r and r["kind"] != "other_crop"})

    def build():
        model, processor = load_vlm(args.model, args.vlm_device, args.loader)
        items, keys = [], []
        cache: dict[str, Image.Image] = {}
        for src, img_name, box in crops:
            if src not in cache:
                cache.clear()
                cache[src] = source.load(source.index_of(src))
            crop = resize_long(cache[src].crop(box), cfg.crop_side)
            for j, f in enumerate(facts[img_name]):
                items.append((VERIFY_PROMPT.format(fact=f["fact"]), crop)); keys.append((src, img_name, box, j))
        probs = vlm_yes_probs(model, processor, items, args.vlm_device, args.score_batch)
        per: dict[tuple, list[float]] = {}
        for (src, img_name, box, j), p in zip(keys, probs):
            per.setdefault((src, img_name, box), [0.0] * len(facts[img_name]))[j] = p
        return [{"crop_of": k[0], "image": k[1], "box": list(k[2]), "in_crop_p": v} for k, v in per.items()]

    crop_facts = {(r["crop_of"], r["image"], tuple(r["box"])): r["in_crop_p"]
                  for r in jsonl_cache(out / "crop_facts.jsonl", build)}

    def outcrop(img_name: str, h: list[float], in_p: list[float]) -> dict[str, float]:
        w = [f["weight"] for f in facts[img_name]]
        out_mask = [p < cfg.verify_threshold for p in in_p]
        wo = [wi for wi, m in zip(w, out_mask) if m]
        ho = [hi for hi, m in zip(h, out_mask) if m]
        wi_ = [wi for wi, m in zip(w, out_mask) if not m]
        hi_ = [hi for hi, m in zip(h, out_mask) if not m]
        return {"owrecall": float(np.dot(ho, wo) / sum(wo)) if wo and sum(wo) > 0 else 0.0,
                "orecall": mean(ho) if ho else 0.0,
                "inrecall": float(np.dot(hi_, wi_) / sum(wi_)) if wi_ and sum(wi_) > 0 else 0.0,
                "n_out": float(len(wo)), "n_in": float(len(wi_))}

    rows = []
    greedy_vis = {}
    for i, (r, sc) in enumerate(zip(samples, scored)):
        if "box" not in r:
            continue
        key = (r.get("crop_of", r["image"]), r["image"], tuple(r["box"]))
        # other_crop rows: what the guess hits among the facts that the *true* image's 0.2 crop does not show
        if r["kind"] == "other_crop":
            ref_key = next(((r["image"], r["image"], tuple(x["box"])) for x in samples
                            if x["image"] == r["image"] and x["kind"] == "greedy" and x["scale"] == cfg.blind_scale),
                           None)
            in_p = crop_facts.get(ref_key) if ref_key else None
            if in_p is None:
                continue
        else:
            in_p = crop_facts[key]
        o = outcrop(r["image"], hits[(r["image"], f"row{i}")], in_p)
        row = {"kind": r["kind"], "image": r["image"], "scale": r.get("scale"), **o,
               "precision": sc["precision"], "contra": sc["contra"], "vis_prec": sc["vis_prec"]}
        rows.append(row)
        if r["kind"] == "greedy":
            v = outcrop(r["image"], hits[(r["image"], f"visonly{i}")], in_p)
            vrow = {"kind": "visible_only", "image": r["image"], "scale": r["scale"], **v, "vis_prec": sc["vis_prec"]}
            rows.append(vrow)
            greedy_vis[(r["image"], r["scale"])] = (row, vrow)

    def reward(x):  # phase-0 reward with owrecall in place of wrecall (visible_only keeps its own vis_prec)
        return (x["owrecall"] + cfg.w_precision * x.get("precision", 0.0) - cfg.w_contra * x.get("contra", 0.0)
                + cfg.w_visible * x.get("vis_prec", 0.0))

    for x in rows:
        x["reward_o"] = reward(x)
    keys_m = ("owrecall", "orecall", "inrecall", "n_out", "n_in", "reward_o")
    summ: dict[str, Any] = {}
    summ["other_crop"] = {m: mean(x[m] for x in rows if x["kind"] == "other_crop") for m in keys_m}
    for s in sorted({x["scale"] for x in rows if x["kind"] == "sampled"}):
        part = {k: [x for x in rows if x["kind"] == k and x["scale"] == s] for k in ("greedy", "sampled", "visible_only")}
        groups: dict[str, list[float]] = {}
        for x in part["sampled"]:
            groups.setdefault(x["image"], []).append(x["reward_o"])
        stds = [statistics.pstdev(v) for v in groups.values() if len(v) > 1]
        paired = [g["owrecall"] - v["owrecall"] for (img, sc_), (g, v) in greedy_vis.items() if sc_ == s]
        summ[f"scale_{s:g}"] = {k: {m: mean(x[m] for x in v) for m in keys_m} for k, v in part.items()} | {
            "groups_with_signal": mean(float(x > 0.05) for x in stds), "group_std": mean(stds),
            "paired_greedy_minus_visible_owrecall": mean(paired),
            "share_greedy_above_visible": mean(float(d > 0) for d in paired),
        }
    (out / "summary_outcrop.json").write_text(json.dumps(summ, indent=1))
    with open(out / "scores_outcrop.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(json.dumps(summ, indent=1), flush=True)


# ---------------------------------------------------------------------------------------------- phase 0d

MULTI_PROMPT = (
    "These are {k} small crops cut out of the same larger photo, at different places. First describe what is "
    "visible in the crops. Then guess what the whole photo shows: the main subjects, the setting, and what is "
    "happening. Use exactly this format:\nVisible: <at most 40 words>\nScene: <at most 50 words>"
)
QUOTE_HIT_PROMPT = (
    "A guess about a photo:\n\"{guess}\"\n\nFact: \"{fact}\"\n\nDoes the guess state this fact, or something that "
    "clearly implies it? Synonyms and more specific wording count (\"a golden retriever\" implies \"a dog\"). The "
    "guess must commit to it: alternatives such as \"a dog or a cat\" and hedges such as \"possibly\" or \"maybe\" "
    "do not count. If yes, reply Yes, then copy the exact words of the guess that state it, in double quotes. "
    "If no, reply No."
)
QUOTE_CHECK_PROMPT = (
    "Text: \"{quote}\"\n\nFact: \"{fact}\"\n\nDoes the text on its own state this fact, or something that clearly "
    "implies it? Answer Yes or No."
)
PHOTO_PROPERTY = re.compile(r"\b(lighting|lit|angle|eye level|composition|photo(graph)?|shot|focus|framed|"
                            r"perspective|blurr?(ed|y)?|depth of field)\b", re.I)


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def quote_in(quote: str, text: str) -> bool:
    """The quoted words occur, in order and contiguous, in the text (case and punctuation ignored)."""
    q, t = norm_words(quote), norm_words(text)
    if not q:
        return False
    return any(t[i : i + len(q)] == q for i in range(len(t) - len(q) + 1))


def parse_quote_hit(text: str) -> tuple[bool, str]:
    t = strip_think(text).strip()
    yes = t[:3].lower() == "yes"
    m = re.findall(r"[\"“](.+?)[\"”]", t, flags=re.S)
    return yes, (max(m, key=len).strip() if m else "")


def text_generate_scored(model, tok, prompts: list[str], max_new_tokens: int, device: str,
                         batch: int) -> list[tuple[str, float]]:
    """Greedy answer plus p(Yes)/(p(Yes)+p(No)) at its first token, in one generate call."""
    import torch

    y, n = yes_no_ids(tok)
    out = []
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        enc = tok([judge_text(tok, p) for p in prompts[i : i + batch]], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                               output_scores=True, return_dict_in_generate=True)
        p_yes = torch.softmax(g.scores[0].float()[:, [y, n]], -1)[:, 0].cpu().tolist()
        texts = tok.batch_decode(g.sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [(t, p if math.isfinite(p) else 0.5) for t, p in zip(texts, p_yes)]
        if (i // batch) % 50 == 0:
            print(f"  text_generate_scored {i + batch}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def overlap(a: Box, b: Box) -> float:
    """Intersection area as a share of the smaller box."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def pick_crops(rng: random.Random, img: Image.Image, scale: float, k: int, cfg: L2GConfig, index=None,
               name: str = "", max_overlap: float = 0.1) -> list[Box]:
    """k crops of area `scale` that barely overlap: each is the most salient of its candidates (thing preferred)
    among those overlapping earlier picks by at most max_overlap; relaxes the limit if none qualifies."""
    from serverless.refgame import edge_energy

    width, height = img.size
    integral = np.pad(edge_energy(img).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    picks: list[Box] = []
    for _ in range(k):
        best, best_key = None, None
        for _ in range(cfg.crop_tries * 4):
            w, h = box_size(rng, width, height, scale, cfg.min_side)
            x0, y0, x1, y1 = b = random_box(rng, width, height, w, h)
            ov = max((overlap(b, p) for p in picks), default=0.0)
            energy = float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
            share = thing_share(index, name, b)
            key = (ov <= max_overlap, share is None or share >= cfg.min_thing_share, -ov, energy)
            if best_key is None or key > best_key:
                best, best_key = b, key
        picks.append(best)
    return picks


def bootstrap_ci(xs: list[float], seed: int = 0, n: int = 10000) -> list[float]:
    if not xs:
        return [float("nan")] * 2
    a = np.asarray(xs, dtype=float)
    r = np.random.default_rng(seed)
    means = a[r.integers(0, len(a), size=(n, len(a)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def multicrop(args: argparse.Namespace) -> None:
    """Phase 0d: k barely-overlapping crops of one image guess the whole photo together.

    Reuses references and verified, weighted facts from a finished phase0 dir (--phase0-dir). Scores only on
    facts that are [subject]/[scene], not photo properties, and visible in none of the k crops ("unseen" facts).
    Two hit scores per (guess, fact): soft = judge p(Yes) (as in phase 0); strict = judge says Yes AND quotes words
    that occur in the guess AND the judge, shown only that quote, agrees it states the fact.

    Rows per image and scale: multi greedy + G sampled; single (crop 1 alone) greedy + G sampled; visible_concat
    (the single-crop greedy Visible lines of all k crops, joined: the copy baseline); multi_visible (the multi
    guess's own Visible line); other_multi (G guesses from k crops of another image: the blind baseline)."""
    p0, out = Path(args.phase0_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = L2GConfig()
    scales = tuple(float(s) for s in args.scales.split(","))
    K, G = args.k, args.group
    (out / "config.json").write_text(json.dumps({"args": vars(args)}, indent=1, default=str))
    load = lambda path: [json.loads(l) for l in path.read_text().splitlines() if l.strip()]  # noqa: E731
    refs = {r["image"]: r["reference"] for r in load(p0 / "references.jsonl")}
    facts = {r["image"]: r["facts"] for r in load(p0 / "facts_weighted.jsonl")}
    names = sorted(refs)[: args.n]
    source = ImageSource(Path(args.images))
    images = {n: source.load(source.index_of(n)) for n in names}
    index = None
    if args.index:
        from serverless.hardneg import LabelIndex

        index = LabelIndex.load(args.index)
    other = {n: names[(i + 1 + len(names) // 2) % len(names)] for i, n in enumerate(names)}
    boxes = {(n, s): pick_crops(random.Random(f"{args.seed}:l2gmulti:{n}:{s}"), images[n], s, K, cfg, index, n)
             for n in names for s in scales}

    def crop_img(name: str, box) -> Image.Image:
        return resize_long(images[name].crop(tuple(box)), cfg.crop_side)

    def multi_item(name: str, bxs) -> tuple[list[dict], list[Image.Image]]:
        content = [{"type": "image"} for _ in bxs] + [{"type": "text", "text": MULTI_PROMPT.format(k=len(bxs))}]
        return content, [crop_img(name, b) for b in bxs]

    def single_item(name: str, box) -> tuple[list[dict], list[Image.Image]]:
        return [{"type": "image"}, {"type": "text", "text": POLICY_PROMPT}], [crop_img(name, box)]

    state: dict[str, Any] = {}

    def vlm():
        if "vlm" not in state:
            state["vlm"] = load_vlm(args.model, args.vlm_device, args.loader)
        return state["vlm"]

    gkw = dict(device=args.vlm_device, batch=args.gen_batch)

    # 1. guesses
    def build_samples():
        model, processor = vlm()
        keys = [(n, s) for n in names for s in scales]
        rows = []
        multi = [multi_item(n, boxes[(n, s)]) for n, s in keys]
        mg = generate(model, processor, multi, 1, args.max_new_tokens, 0.0, **gkw)
        ms = generate(model, processor, multi, G, args.max_new_tokens, 1.0, **gkw)
        singles = [single_item(n, b) for n, s in keys for b in boxes[(n, s)]]
        sg = generate(model, processor, singles, 1, args.max_new_tokens, 0.0, **gkw)
        ss = generate(model, processor, singles[::K], G, args.max_new_tokens, 1.0, **gkw)
        om = generate(model, processor, [multi_item(other[n], boxes[(other[n], s)]) for n, s in keys], G,
                      args.max_new_tokens, 1.0, **gkw)
        for j, (n, s) in enumerate(keys):
            base = {"image": n, "scale": s, "boxes": boxes[(n, s)]}
            g = parse_guess(mg[j][0])
            rows.append({**base, "kind": "multi_greedy", **g})
            rows.append({**base, "kind": "multi_visible", "scene": g["visible"], "visible": g["visible"],
                         "format_ok": True, "raw": ""})
            rows += [{**base, "kind": "multi_sampled", "slot": t, **parse_guess(x)} for t, x in enumerate(ms[j])]
            sgs = [parse_guess(sg[j * K + c][0]) for c in range(K)]
            rows.append({**base, "kind": "single_greedy", **sgs[0]})
            rows += [{**base, "kind": "single_sampled", "slot": t, **parse_guess(x)} for t, x in enumerate(ss[j])]
            vc = " ".join(x["visible"] for x in sgs if x["visible"])
            rows.append({**base, "kind": "visible_concat", "scene": vc, "visible": vc, "format_ok": True, "raw": ""})
            rows += [{**base, "kind": "other_multi", "crop_of": other[n], "slot": t, **parse_guess(x)}
                     for t, x in enumerate(om[j])]
        return rows

    samples = jsonl_cache(out / "samples.jsonl", build_samples)

    # 2. which facts each crop shows (frozen VLM on the crop); unseen = shown by none of the k crops
    def build_crop_facts():
        model, processor = vlm()
        items, keys = [], []
        for n in names:
            for s in scales:
                for c, b in enumerate(boxes[(n, s)]):
                    im = crop_img(n, b)
                    for j, f in enumerate(facts[n]):
                        items.append((VERIFY_PROMPT.format(fact=f["fact"]), im)); keys.append((n, s, c, j))
        probs = vlm_yes_probs(model, processor, items, args.vlm_device, args.score_batch)
        per: dict[tuple, list[list[float]]] = {}
        for (n, s, c, j), p in zip(keys, probs):
            per.setdefault((n, s), [[0.0] * len(facts[n]) for _ in range(K)])[c][j] = p
        return [{"image": n, "scale": s, "in_crop_p": v} for (n, s), v in per.items()]

    in_crop = {(r["image"], r["scale"]): r["in_crop_p"] for r in jsonl_cache(out / "crop_facts.jsonl",
                                                                               build_crop_facts)}
    if "vlm" in state:
        del state["vlm"]
        _free()

    def eligible(f: dict) -> bool:
        return f["type"] in ("subject", "scene") and not PHOTO_PROPERTY.search(f["fact"])

    def unseen(n: str, s: float, upto: int = K) -> list[int]:
        ps = in_crop[(n, s)][:upto]
        return [j for j, f in enumerate(facts[n]) if eligible(f) and all(p[j] < cfg.verify_threshold for p in ps)]

    # 3. judge: soft p(Yes) and strict quoted hit, on the facts unseen by all k crops
    def build_hits():
        judge, jtok = load_text_model(args.judge, args.judge_device)
        pairs = []
        for i, r in enumerate(samples):
            for j in unseen(r["image"], r["scale"]):
                pairs.append((i, j))
        prompts = [QUOTE_HIT_PROMPT.format(guess=samples[i]["scene"] or "(empty)", fact=facts[samples[i]["image"]][j]
                                           ["fact"]) for i, j in pairs]
        res = text_generate_scored(judge, jtok, prompts, args.max_quote_tokens, args.judge_device,
                                   args.judge_gen_batch)
        parsed = [parse_quote_hit(t) for t, _ in res]
        need = [k for k, ((i, _), (yes, q)) in enumerate(zip(pairs, parsed))
                if yes and q and quote_in(q, samples[i]["scene"])]
        check = text_yes_probs(judge, jtok, [QUOTE_CHECK_PROMPT.format(quote=parsed[k][1], fact=facts[
            samples[pairs[k][0]]["image"]][pairs[k][1]]["fact"]) for k in need], args.judge_device, args.judge_batch)
        check_of = dict(zip(need, check))
        rows = []
        for k, ((i, j), (text, p)) in enumerate(zip(pairs, res)):
            yes, q = parsed[k]
            rows.append({"row": i, "fact": j, "soft": p, "yes": yes, "quote": q,
                         "quote_ok": k in check_of, "quote_check": check_of.get(k),
                         "strict": float(k in check_of and check_of[k] >= 0.5), "raw": text})
        del judge
        _free()
        return rows

    hit_rows = jsonl_cache(out / "hits.jsonl", build_hits)
    hit = {(h["row"], h["fact"]): h for h in hit_rows}

    # 4. scores
    def wrec(vals: list[float], ws: list[float]) -> float | None:
        return float(np.dot(vals, ws) / sum(ws)) if ws and sum(ws) > 0 else None

    scored = []
    for i, r in enumerate(samples):
        n = r["image"]
        u = unseen(n, r["scale"])
        ws = [facts[n][j]["weight"] for j in u]
        row = {k: r[k] for k in ("image", "scale", "kind") if k in r} | {"slot": r.get("slot"), "n_unseen": len(u)}
        for m in ("soft", "strict"):
            row[f"uw_{m}"] = wrec([hit[(i, j)][m] for j in u], ws)
            row[f"u_{m}"] = mean([hit[(i, j)][m] for j in u]) if u else None
        row["words"] = n_words(r["scene"])
        row["format_ok"] = r["format_ok"]
        scored.append(row)
    with open(out / "scores.jsonl", "w") as f:
        for x in scored:
            f.write(json.dumps(x) + "\n")

    kinds = ("multi_greedy", "multi_sampled", "single_greedy", "single_sampled", "visible_concat", "multi_visible",
             "other_multi")
    summ: dict[str, Any] = {}
    for s in scales:
        part = [x for x in scored if x["scale"] == s and x["n_unseen"] > 0]
        by = {k: [x for x in part if x["kind"] == k] for k in kinds}
        per_img = lambda k, m: {img: mean(x[m] for x in by[k] if x["image"] == img)  # noqa: E731
                                for img in {x["image"] for x in by[k]}}
        d: dict[str, Any] = {"images": len({x["image"] for x in part}),
                             "unseen_facts_per_image": mean(x["n_unseen"] for x in by["multi_greedy"]),
                             "single_crop_unseen_per_image": mean(len(unseen(n, s, 1)) for n in names)}
        for k in kinds:
            d[k] = {m: mean(x[m] for x in by[k]) for m in ("uw_soft", "uw_strict", "u_soft", "u_strict", "words",
                                                          "format_ok")} | {"n": len(by[k])}
        for m in ("uw_soft", "uw_strict"):
            for a, b in (("multi_greedy", "visible_concat"), ("multi_sampled", "visible_concat"),
                         ("multi_sampled", "single_sampled"), ("multi_greedy", "single_greedy"),
                         ("multi_sampled", "other_multi")):
                pa, pb = per_img(a, m), per_img(b, m)
                diffs = [pa[i] - pb[i] for i in pa if i in pb and not math.isnan(pa[i]) and not math.isnan(pb[i])]
                d[f"gap_{a}-{b}_{m}"] = {"mean": mean(diffs), "ci95": bootstrap_ci(diffs, args.seed),
                                         "share_above": mean(float(x > 0) for x in diffs), "n": len(diffs)}
        groups: dict[str, list[float]] = {}
        for x in by["multi_sampled"]:
            groups.setdefault(x["image"], []).append(x["uw_strict"])
        stds = [statistics.pstdev(v) for v in groups.values() if len(v) > 1 and None not in v]
        d["multi_groups_with_signal_strict"] = mean(float(v > 0.05) for v in stds)
        summ[f"scale_{s:g}"] = d
    yes = [h for h in hit_rows if h["yes"]]
    summ["judge"] = {"pairs": len(hit_rows), "yes": len(yes),
                     "yes_with_valid_quote": mean(float(h["quote_ok"]) for h in yes),
                     "strict_of_yes": mean(h["strict"] for h in yes),
                     "soft_ge_half_but_not_strict": mean(float(h["soft"] >= 0.5 and not h["strict"]) for h in hit_rows)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)

    # 5. label sheet: strict hits of multi guesses that visible_concat did not get (what training would reward)
    vc_row = {(r["image"], r["scale"]): i for i, r in enumerate(samples) if r["kind"] == "visible_concat"}
    cand = []
    for i, r in enumerate(samples):
        if r["kind"] not in ("multi_greedy", "multi_sampled"):
            continue
        v = vc_row[(r["image"], r["scale"])]
        for j in unseen(r["image"], r["scale"]):
            h = hit[(i, j)]
            if h["strict"] and not hit[(v, j)]["strict"]:
                cand.append((r, j, h))
    pick = random.Random(args.seed + 3).sample(cand, min(args.label_n, len(cand)))
    with open(out / "label_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "scale", "kind", "fact_type", "fact", "scene", "quote", "soft", "quote_check",
                    "human_correct_yes_no", "notes"])
        for k, (r, j, h) in enumerate(pick):
            fct = facts[r["image"]][j]
            w.writerow([k, r["image"], r["scale"], r["kind"], fct["type"], fct["fact"], r["scene"], h["quote"],
                        f"{h['soft']:.3f}", f"{h['quote_check']:.3f}", "", ""])
    print("wrote", out / "summary.json", "and", out / "label_sheet.csv", f"({len(pick)} of {len(cand)} rows)")


def _free() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("phase0", help="base-model checks before training")
    q.add_argument("--images", required=True, help="val2017 directory or ZIP")
    q.add_argument("--index", default="", help="COCO label index npz (hardneg build-index); crops + look-alikes")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    q.add_argument("--loader", default="hf", choices=["hf", "unsloth"])
    q.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507")
    q.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    q.add_argument("--retriever", default="google/siglip-base-patch16-224")
    q.add_argument("--no-retrieval", action="store_true")
    q.add_argument("--vlm-device", default="cuda:0")
    q.add_argument("--judge-device", default="cuda:1")
    q.add_argument("--n", type=int, default=200)
    q.add_argument("--scales", default="0.1,0.2,0.3")
    q.add_argument("--group", type=int, default=8)
    q.add_argument("--max-new-tokens", type=int, default=160)
    q.add_argument("--max-ref-tokens", type=int, default=768, help="compute cap only; the prompt sets no length")
    q.add_argument("--max-fact-tokens", type=int, default=600)
    q.add_argument("--max-claim-tokens", type=int, default=200)
    q.add_argument("--gen-batch", type=int, default=16)
    q.add_argument("--score-batch", type=int, default=8)
    q.add_argument("--judge-batch", type=int, default=32)
    q.add_argument("--judge-gen-batch", type=int, default=32)
    q.add_argument("--pool", type=int, default=2000, help="val images searched for retrieval look-alikes")
    q.add_argument("--label-n", type=int, default=50)
    q.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("outcrop", help="phase 0b: re-score phase-0 caches on facts the crop does not show")
    r.add_argument("--images", required=True)
    r.add_argument("--out-dir", required=True, help="a finished phase0 out dir")
    r.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    r.add_argument("--loader", default="hf", choices=["hf", "unsloth"])
    r.add_argument("--vlm-device", default="cuda:0")
    r.add_argument("--score-batch", type=int, default=16)
    m = sub.add_parser("multicrop", help="phase 0d: k crops guess the whole photo together")
    m.add_argument("--images", required=True)
    m.add_argument("--index", default="")
    m.add_argument("--phase0-dir", required=True, help="a finished phase0 out dir (references, weighted facts)")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    m.add_argument("--loader", default="hf", choices=["hf", "unsloth"])
    m.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507")
    m.add_argument("--vlm-device", default="cuda:0")
    m.add_argument("--judge-device", default="cuda:0")
    m.add_argument("--n", type=int, default=200)
    m.add_argument("--k", type=int, default=3, help="crops per guess")
    m.add_argument("--scales", default="0.1,0.2", help="area of EACH crop")
    m.add_argument("--group", type=int, default=8)
    m.add_argument("--max-new-tokens", type=int, default=200)
    m.add_argument("--max-quote-tokens", type=int, default=48)
    m.add_argument("--gen-batch", type=int, default=16)
    m.add_argument("--score-batch", type=int, default=16)
    m.add_argument("--judge-batch", type=int, default=16)
    m.add_argument("--judge-gen-batch", type=int, default=16)
    m.add_argument("--label-n", type=int, default=150)
    m.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "phase0":
        phase0(args)
    elif args.cmd == "outcrop":
        rescore_outcrop(args)
    elif args.cmd == "multicrop":
        multicrop(args)


if __name__ == "__main__":
    main()
