"""Tier-1 evaluation of a Kaggle-trained LoRA adapter against its base model (eval protocol, dims A-F).

One 4-bit base model is loaded with the adapter; every item is answered twice, adapter on ("tuned") and off
("base"), greedy, thinking off. Differences are paired, with 95% bootstrap CIs.

  A  in-domain   caption game on COCO val2017 (forger vs frozen base judge, both caption orders) with a
                 length control, plus SSL-R1 puzzles on val2017
  B  general     MMStar
  C  vision      CV-Bench
  D  hallucin.   POPE
  E  text-only   MMLU-Pro, GSM8K
  F  encoder     vision-tower features: linear CKA base vs tuned, k-NN accuracy on COCO main category

    python kaggle/eval_adapter.py --adapter .../out/capgame-v7/latest --repo .../ServerlessRL-capgame \
        --coco-val .../val2017 --coco-ann .../annotations --out .../eval_v4 [--n 300] [--only A,B]

`--repo` is a checkout of the masked-caption-game branch (caption game + SSL-R1 tasks). Datasets B-E are
downloaded with `datasets` (set HF_HOME / HF_DATASETS_OFFLINE on nodes without internet after a first pull).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

LETTERS = "ABCDEFGHIJ"


# ---------------------------------------------------------------------------------------------- setup

def import_repo(repo: Path, coco_val: Path):
    """Make `serverless.*` importable from a checkout of the repo (it expects a parent `sslvlm.config`)."""
    root = Path("/tmp/sslvlm_eval_src")
    (root / "sslvlm").mkdir(parents=True, exist_ok=True)
    link = root / "serverless"
    if not link.exists():
        link.symlink_to(repo.resolve())
    (root / "sslvlm" / "__init__.py").write_text("")
    (root / "sslvlm" / "config.py").write_text(
        f"from pathlib import Path\nCOCO_DIR = Path({str(coco_val)!r})\nOUTPUT_ROOT = Path('/tmp')\n")
    sys.path.insert(0, str(root))
    stub = types.ModuleType("serverless.train")  # caption_game imports it only for message_reasoning
    stub.message_reasoning = lambda m: None
    sys.modules["serverless.train"] = stub


class Runner:
    """Batched greedy generation with the adapter on or off."""

    def __init__(self, base: str, adapter: Path, batch: int):
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.processor = AutoProcessor.from_pretrained(base)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(base, device_map={"": 0}, dtype=dtype)
        self.model = PeftModel.from_pretrained(model, str(adapter)).eval()
        self.batch = batch
        stops = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")]
        self.stops = {s for s in stops if s is not None}

    def _text(self, content):
        return self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, items, tuned: bool, max_new_tokens: int) -> list[str]:
        """items: list of (content, images); content uses {"type": "image"} placeholders in order."""
        out = []
        ctx = contextlib.nullcontext() if tuned else self.model.disable_adapter()
        with ctx:
            for i in range(0, len(items), self.batch):
                chunk = items[i : i + self.batch]
                texts = [self._text(c) for c, _ in chunk]
                images = [im for _, ims in chunk for im in ims] or None
                inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")
                gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
                for row in gen[:, inputs["input_ids"].shape[1]:]:
                    ids = row.tolist()
                    cut = next((k for k, t in enumerate(ids) if t in self.stops), len(ids))
                    out.append(self.tok.decode(ids[:cut], skip_special_tokens=True).strip())
        return out

    @torch.no_grad()
    def vision_features(self, images, tuned: bool) -> np.ndarray:
        """Mean-pooled vision-tower output per image (after the merger), one row per image."""
        feats = []
        ctx = contextlib.nullcontext() if tuned else self.model.disable_adapter()
        visual = self.model.base_model.model.model.visual
        with ctx:
            for img in images:
                proc = self.processor.image_processor(images=[img], return_tensors="pt")
                pv = proc["pixel_values"].to("cuda", dtype=visual.dtype if hasattr(visual, "dtype") else None)
                out = visual(pv, grid_thw=proc["image_grid_thw"].to("cuda"))
                h = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    h = out.pooler_output
                feats.append(h.float().mean(0).cpu().numpy())
        return np.stack(feats)


# ---------------------------------------------------------------------------------------------- stats

def paired(a: list[float], b: list[float], iters: int = 2000, seed: int = 0) -> dict:
    """Mean of tuned (a), base (b), and the paired difference with a 95% bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (iters, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(d), "tuned": float(a.mean()), "base": float(b.mean()), "diff": float(d.mean()), "ci": [float(lo), float(hi)]}


def letter(text: str, n: int) -> str | None:
    m = re.search(r"\\boxed\{\s*\(?([A-J])\)?\s*\}", text) or re.match(r"\s*\(?([A-J])\b", text) \
        or re.search(r"(?:answer|option)\s*(?:is|:)?\s*\(?([A-J])\b", text, re.I)
    return m.group(1) if m and LETTERS.index(m.group(1)) < n else None


MC_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------------------------- A: caption game

def eval_caption_game(R: Runner, args, out: Path) -> dict:
    import serverless.caption_game as CG
    from serverless.images import ImageSource

    caps = json.load(open(args.coco_ann / "captions_val2017.json"))
    refs: dict[str, list[str]] = {}
    id2name = {im["id"]: im["file_name"] for im in caps["images"]}
    for a in caps["annotations"]:
        refs.setdefault(id2name[a["image_id"]], []).append(a["caption"])
    src = ImageSource(args.coco_val)
    # same images, masks and seeds as training val (seed 0), extended to --n-cap images
    games = CG.step_images(src, 0, "val", 0, args.n_cap)
    masked, fulls, names = [], [], []
    for name, img, rng in games:
        box = CG.salient_box(rng, img, args.mask_ratio, 8, 28)
        masked.append(CG.mask_image(img, box)); fulls.append(img); names.append(name.rsplit("/", 1)[-1])
    lo, hi = 50, 90
    cap_items = [([{"type": "image"}, {"type": "text", "text": CG.CAPTION_PROMPT.format(lo=lo, hi=hi)}], [im])
                 for im in fulls]
    forge_items = [([{"type": "image"}, {"type": "text", "text": CG.FORGE_PROMPT.format(lo=lo, hi=hi)}], [im])
                   for im in masked]
    real = [CG.clean_caption(t) for t in R.generate(cap_items, False, 320)]  # frozen captioner = base
    fake = {c: [CG.clean_caption(t) for t in R.generate(forge_items, c == "tuned", 320)] for c in ("tuned", "base")}

    from serverless.config import FORMAT_PROMPT

    def judge(cond):
        items, keys = [], []
        for i, (r, f) in enumerate(zip(real, fake[cond])):
            for real_first in (True, False):  # both orders cancel position bias
                a, b = (r, f) if real_first else (f, r)
                items.append(([{"type": "image"}, {"type": "text", "text": CG.JUDGE_PROMPT.format(a=a, b=b)
                                                   + "\n" + FORMAT_PROMPT}], [fulls[i]]))
                keys.append((i, "A" if real_first else "B"))
        texts = R.generate(items, False, 512)  # frozen judge = base
        fooled = np.zeros(len(real)); parsed = np.zeros(len(real))
        for (i, ans), t in zip(keys, texts):
            p = CG.parse_judgement(t)
            parsed[i] += (p is not None) / 2
            fooled[i] += (p is not None and p != ans) / 2
        return fooled, parsed

    rows, metrics = [], {}
    per = {}
    for cond in ("tuned", "base"):
        fooled, parsed = judge(cond)
        recall = [np.mean([CG.content_recall(f, r) for r in refs.get(n, [""])]) for f, n in zip(fake[cond], names)]
        prec = []
        for f, n in zip(fake[cond], names):
            cw = CG.content_words(f)
            ref_w = set().union(*[CG.content_words(r) for r in refs.get(n, [""])], CG.content_words(real[names.index(n)]))
            prec.append(len(cw & ref_w) / len(cw) if cw else 0.0)
        gate = [CG.ForgeGate().check(f, r)[1] for f, r in zip(fake[cond], real)]
        per[cond] = dict(fool=fooled, parsed=parsed, recall=np.array(recall), precision=np.array(prec),
                         words=np.array([len(CG.words(f)) for f in fake[cond]], float),
                         leak=np.array([g == "leak" for g in gate], float))
    # length control: truncate both forgeries of an image to the shorter one's word count, then recall
    trunc = {c: [] for c in ("tuned", "base")}
    for i, n in enumerate(names):
        k = int(min(per["tuned"]["words"][i], per["base"]["words"][i]))
        for c in trunc:
            cut = " ".join(CG.words(fake[c][i])[:k])
            trunc[c].append(np.mean([CG.content_recall(cut, r) for r in refs.get(n, [""])]))
    for k in ("fool", "recall", "precision", "words", "leak", "parsed"):
        metrics[f"A/capgame/{k}"] = paired(per["tuned"][k], per["base"][k])
    metrics["A/capgame/recall_len_matched"] = paired(trunc["tuned"], trunc["base"])
    for i, n in enumerate(names):
        rows.append({"image": n, "real": real[i], "tuned": fake["tuned"][i], "base": fake["base"][i],
                     "fool_tuned": per["tuned"]["fool"][i], "fool_base": per["base"]["fool"][i]})
    with open(out / "capgame_items.jsonl", "w") as f:
        f.writelines(json.dumps(r, default=float) + "\n" for r in rows)
    return metrics


def eval_puzzles(R: Runner, args, out: Path) -> dict:
    from serverless.config import FORMAT_PROMPT
    from serverless.images import ImageSource
    from serverless.objective import Objective

    obj = Objective.from_spec("ssl-r1", seed=0)
    src = ImageSource(args.coco_val)
    puzzles = obj.batch(src, "val", 0, args.n_puzzle)
    items = []
    for p in puzzles:
        content, ims = [], []
        for part in p.parts:
            if isinstance(part, Image.Image):
                content.append({"type": "image"}); ims.append(part)
            else:
                content.append({"type": "text", "text": part})
        content.append({"type": "text", "text": "\n" + FORMAT_PROMPT})
        items.append((content, ims))
    acc = {}
    for c in ("tuned", "base"):
        texts = R.generate(items, c == "tuned", 1024)
        acc[c] = [obj.score(p, t).accuracy for p, t in zip(puzzles, texts)]
    m = {"A/ssl-r1/all": paired(acc["tuned"], acc["base"])}
    for task in sorted({p.task for p in puzzles}):
        idx = [i for i, p in enumerate(puzzles) if p.task == task]
        m[f"A/ssl-r1/{task}"] = paired([acc["tuned"][i] for i in idx], [acc["base"][i] for i in idx])
    return m


# ---------------------------------------------------------------------------------------------- B-E: benchmarks

def subset(ds, n, seed=0):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return ds.select(sorted(idx[:n]))


def mc_eval(R, name, rows, max_new_tokens=16):
    """rows: (content, images, answer_letter, n_options)."""
    items = [(c, ims) for c, ims, _, _ in rows]
    res = {}
    for cond in ("tuned", "base"):
        texts = R.generate(items, cond == "tuned", max_new_tokens)
        res[cond] = [float(letter(t, n) == a) for t, (_, _, a, n) in zip(texts, rows)]
    return {name: paired(res["tuned"], res["base"])}


def eval_mmstar(R, args):
    from datasets import load_dataset
    ds = subset(load_dataset("Lin-Chen/MMStar", split="val"), args.n)
    rows = [([{"type": "image"}, {"type": "text", "text": ex["question"] + MC_SUFFIX}], [ex["image"].convert("RGB")],
             ex["answer"].strip()[0], 4) for ex in ds]
    return mc_eval(R, "B/mmstar", rows)


def eval_cvbench(R, args):
    from datasets import load_dataset
    ds = subset(load_dataset("nyu-visionx/CV-Bench", split="test"), args.n)
    rows = []
    for ex in ds:
        opts = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(ex["choices"]))
        q = f"{ex['question']}\n{opts}{MC_SUFFIX}"
        rows.append(([{"type": "image"}, {"type": "text", "text": q}], [ex["image"].convert("RGB")],
                     ex["answer"].strip("() "), len(ex["choices"])))
    return mc_eval(R, "C/cv-bench", rows)


def eval_pope(R, args):
    from datasets import load_dataset
    ds = subset(load_dataset("lmms-lab/POPE", split="test"), args.n)
    items = [([{"type": "image"}, {"type": "text", "text": ex["question"] + "\nAnswer yes or no."}],
              [ex["image"].convert("RGB")]) for ex in ds]
    res = {}
    for cond in ("tuned", "base"):
        texts = R.generate(items, cond == "tuned", 8)
        res[cond] = [float(t.strip().lower().startswith(ex["answer"].strip().lower())) for t, ex in zip(texts, ds)]
        res[cond + "_yes"] = [float(t.strip().lower().startswith("yes")) for t in texts]
    return {"D/pope/acc": paired(res["tuned"], res["base"]),
            "D/pope/yes_rate": paired(res["tuned_yes"], res["base_yes"])}


def eval_mmlu_pro(R, args):
    from datasets import load_dataset
    ds = subset(load_dataset("TIGER-Lab/MMLU-Pro", split="test"), args.n)
    rows = []
    for ex in ds:
        opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(ex["options"]))
        rows.append(([{"type": "text", "text": f"{ex['question']}\n{opts}{MC_SUFFIX}"}], [], ex["answer"],
                     len(ex["options"])))
    return mc_eval(R, "E/mmlu-pro", rows)


def eval_gsm8k(R, args):
    from datasets import load_dataset
    ds = subset(load_dataset("openai/gsm8k", "main", split="test"), args.n)
    items = [([{"type": "text", "text": ex["question"] + "\nReason step by step, then give the final number "
               "after '####'."}], []) for ex in ds]
    gold = [ex["answer"].split("####")[-1].strip().replace(",", "") for ex in ds]

    def num(t):
        t = t.replace(",", "")
        m = re.search(r"####\s*(-?[\d.]+)", t) or None
        nums = re.findall(r"-?\d+(?:\.\d+)?", t)
        v = m.group(1) if m else (nums[-1] if nums else None)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    res = {}
    for cond in ("tuned", "base"):
        texts = R.generate(items, cond == "tuned", 512)
        res[cond] = [float(num(t) is not None and abs(num(t) - float(g)) < 1e-6) for t, g in zip(texts, gold)]
    return {"E/gsm8k": paired(res["tuned"], res["base"])}


# ---------------------------------------------------------------------------------------------- F: encoder

def linear_cka(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return float(np.linalg.norm(x.T @ y) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def knn_acc(x, labels, k=20):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, 1)[:, :k]
    pred = [Counter(labels[nn[i]]).most_common(1)[0][0] for i in range(len(x))]
    return (np.array(pred) == labels).astype(float)


def eval_encoder(R, args):
    inst = json.load(open(args.coco_ann / "instances_val2017.json"))
    area: dict[int, Counter] = {}
    for a in inst["annotations"]:
        area.setdefault(a["image_id"], Counter())[a["category_id"]] += a["area"]
    ims = [im for im in inst["images"] if im["id"] in area]
    random.Random(0).shuffle(ims)
    ims = ims[: args.n_enc]
    images = [Image.open(args.coco_val / im["file_name"]).convert("RGB") for im in ims]
    labels = np.array([area[im["id"]].most_common(1)[0][0] for im in ims])
    f_t, f_b = R.vision_features(images, True), R.vision_features(images, False)
    return {"F/cka_base_vs_tuned": {"value": linear_cka(f_t, f_b), "n": len(ims)},
            "F/knn20_main_category": paired(knn_acc(f_t, labels), knn_acc(f_b, labels))}


# ---------------------------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--base", default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    p.add_argument("--repo", type=Path, required=True, help="checkout of the masked-caption-game branch")
    p.add_argument("--coco-val", type=Path, required=True)
    p.add_argument("--coco-ann", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=300, help="items per benchmark (B-E)")
    p.add_argument("--n-cap", type=int, default=200, help="caption-game images")
    p.add_argument("--n-puzzle", type=int, default=200, help="SSL-R1 puzzles (all 5 tasks)")
    p.add_argument("--n-enc", type=int, default=1000, help="images for the encoder probes")
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--only", default="A,B,C,D,E,F", help="comma list of dimensions")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import_repo(args.repo, args.coco_val)
    R = Runner(args.base, args.adapter, args.batch)
    todo = {
        "A": [lambda: eval_caption_game(R, args, args.out), lambda: eval_puzzles(R, args, args.out)],
        "B": [lambda: eval_mmstar(R, args)],
        "C": [lambda: eval_cvbench(R, args)],
        "D": [lambda: eval_pope(R, args)],
        "E": [lambda: eval_mmlu_pro(R, args), lambda: eval_gsm8k(R, args)],
        "F": [lambda: eval_encoder(R, args)],
    }
    results_path = args.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for dim in args.only.split(","):
        for fn in todo[dim.strip()]:
            t = time.time()
            try:
                res = fn()
            except Exception as e:  # keep going; one broken loader should not lose the rest
                import traceback
                traceback.print_exc()
                res = {f"{dim}/error/{fn.__code__.co_names[0] if fn.__code__.co_names else 'eval'}": repr(e)[:500]}
            results.update(res)
            results_path.write_text(json.dumps(results, indent=1, default=float))
            for k, v in res.items():
                print(k, json.dumps(v, default=lambda x: round(float(x), 4)), f"({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
