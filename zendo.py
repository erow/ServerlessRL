"""Zendo / Bongard-style rule induction: infer a hidden visual rule from examples, then apply it to new images.

One puzzle: a hidden rule r(x) (a predicate computed from COCO instance annotations), `shots` images that follow it
(group A), `shots` that do not (group B), and `queries` new images, half of each. Two tasks:

  Z1 classify  (policy): A + B + queries -> "Rule: <hypothesis>" and "Answers: yes no ..."; reward = share of
               queries answered right (format error = 0).
  Z2 describe  (policy): A + B -> "Rule: <at most max_words words>". A frozen listener (base model) sees the rule
               and ONE query image at a time and answers yes/no; reward = mean over queries of 1 - |p(yes) - r(q)|.

Rules (families): category ("contains a dog"), count ("at least 3 people"), supercategory ("contains an animal"),
size ("the largest object covers more than a third of the image"), clutter ("at least 10 annotated objects"),
spatial ("the largest object is in the left half"), and AND-NOT combinations of category/supercategory rules.
The `spatial` family is held out as the test family.

Negatives ("negatives" option):
  random  - any image that breaks the rule.
  near    - near misses: for each positive, an image that breaks the rule but has the most similar set of COCO
            categories (cosine of per-category counts, ignoring the rule's own categories), picked among the top
            `near_pool`. The two groups then differ mainly in the rule, not in the scene type.
  near-disc - near negatives, and queries chosen to contradict as many other library rules as possible that also
            fit the examples (so a rule that merely fits the examples is not enough).

Phase 0 (no training) checks, per negative scheme: trivial solvers (colour-histogram and category-vector
nearest-centroid, SigLIP nearest-centroid if available), base-model Z1 (greedy + sampled group: accuracy, reward
spread), base-model Z2 (greedy rule scored by the frozen listener) and the listener given the TRUE rule text
(the listener's ceiling), and how many library rules fit each puzzle's examples (ambiguity).

  python -m serverless.zendo phase0 --images /path/val2017 --instances /path/instances_val2017.json \\
      --out-dir out/zendo_phase0 --n 300 --scheme random='{"negatives":"random"}' --scheme near='{"negatives":"near"}'
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

# ------------------------------------------------------------------------------------------------- data


@dataclass
class Rec:
    """One annotated image: file name, size, and its (category name, bbox xywh, area) objects."""
    name: str
    w: int
    h: int
    objs: list[tuple[str, tuple[float, float, float, float], float]]
    counts: Counter = field(default_factory=Counter)
    supers: set[str] = field(default_factory=set)


def load_records(instances: Path, min_objs: int = 1) -> tuple[list[Rec], dict[str, str]]:
    data = json.loads(Path(instances).read_text())
    cats = {c["id"]: c for c in data["categories"]}
    sup = {c["name"]: c["supercategory"] for c in data["categories"]}
    by_img: dict[int, list] = defaultdict(list)
    for a in data["annotations"]:
        if a.get("iscrowd", 0):
            continue
        by_img[a["image_id"]].append((cats[a["category_id"]]["name"], tuple(a["bbox"]), float(a["area"])))
    recs = []
    for im in sorted(data["images"], key=lambda x: x["file_name"]):
        objs = by_img.get(im["id"], [])
        if len(objs) < min_objs:
            continue
        r = Rec(im["file_name"], im["width"], im["height"], objs)
        r.counts = Counter(o[0] for o in objs)
        r.supers = {sup[o[0]] for o in objs}
        recs.append(r)
    return recs, sup


# ------------------------------------------------------------------------------------------------ rules

PLURAL = {"person": "people", "bus": "buses", "sheep": "sheep", "skis": "pairs of skis", "knife": "knives",
          "wine glass": "wine glasses", "couch": "couches", "bench": "benches", "sandwich": "sandwiches",
          "mouse": "mice", "toothbrush": "toothbrushes", "glass": "glasses"}
SUPER_TEXT = {"animal": "an animal", "vehicle": "a vehicle", "food": "some food", "furniture": "a piece of furniture",
              "kitchen": "a kitchen utensil or tableware", "sports": "sports equipment",
              "electronic": "an electronic device", "outdoor": "an outdoor fixture such as a bench, sign or hydrant",
              "accessory": "a bag, umbrella or tie", "appliance": "a kitchen appliance", "indoor": "a book, clock, vase or similar household item",
              "person": "a person"}


def plural(c: str) -> str:
    return PLURAL.get(c, c + "s")


def article(c: str) -> str:
    return ("an " if c[0] in "aeiou" else "a ") + c


@dataclass
class Rule:
    key: str
    family: str
    text: str  # what a perfect player could say; also fed to the listener as its ceiling
    fn: Callable[[Rec], bool] = field(repr=False)
    cats: frozenset[str] = frozenset()  # categories the rule mentions (ignored by near-miss similarity)


def largest(r: Rec):
    return max(r.objs, key=lambda o: o[2])


def build_rules(recs: list[Rec], sup: dict[str, str], min_prev: float, max_prev: float, combos: int,
                seed: int = 0) -> list[Rule]:
    names = sorted({c for r in recs for c in r.counts})
    supers = sorted(set(sup.values()))
    rules: list[Rule] = []
    for c in names:
        rules.append(Rule(f"has:{c}", "category", f"The image contains {article(c)}.",
                          lambda r, c=c: r.counts[c] > 0, frozenset([c])))
        for k in (2, 3, 5):
            rules.append(Rule(f"count:{c}>={k}", "count", f"The image contains at least {k} {plural(c)}.",
                              lambda r, c=c, k=k: r.counts[c] >= k, frozenset([c])))
    for s in supers:
        if s == "person":
            continue
        cs = frozenset(c for c in names if sup[c] == s)
        rules.append(Rule(f"super:{s}", "supercategory", f"The image contains {SUPER_TEXT.get(s, s)}.",
                          lambda r, s=s: s in r.supers, cs))
    rules += [
        Rule("size:big", "size", "The largest object covers more than a third of the image.",
             lambda r: largest(r)[2] > r.w * r.h / 3),
        Rule("size:small", "size", "Every object is small: none covers more than 5% of the image.",
             lambda r: largest(r)[2] < 0.05 * r.w * r.h),
        Rule("clutter:10", "clutter", "The image shows at least 10 separate objects.", lambda r: len(r.objs) >= 10),
        Rule("clutter:single", "clutter", "The image shows exactly one object.", lambda r: len(r.objs) == 1),
        Rule("spatial:left", "spatial", "The largest object is in the left third of the image.",
             lambda r: (largest(r)[1][0] + largest(r)[1][2] / 2) < r.w / 3),
        Rule("spatial:right", "spatial", "The largest object is in the right third of the image.",
             lambda r: (largest(r)[1][0] + largest(r)[1][2] / 2) > 2 * r.w / 3),
        Rule("spatial:top", "spatial", "The largest object is in the top third of the image.",
             lambda r: (largest(r)[1][1] + largest(r)[1][3] / 2) < r.h / 3),
        Rule("spatial:bottom", "spatial", "The largest object is in the bottom third of the image.",
             lambda r: (largest(r)[1][1] + largest(r)[1][3] / 2) > 2 * r.h / 3),
    ]
    n = len(recs)

    def prev(rule: Rule) -> float:
        return sum(rule.fn(r) for r in recs) / n

    keep = [r for r in rules if min_prev <= prev(r) <= max_prev]
    # AND-NOT combinations of frequent category / supercategory rules, e.g. "a person but no car".
    atoms = [r for r in rules if r.family in ("category", "supercategory") and prev(r) >= 2 * min_prev]
    rng = random.Random(f"{seed}:combos")
    seen = set()
    for _ in range(combos * 20):
        if sum(r.family == "and-not" for r in keep) >= combos:
            break
        a, b = rng.sample(atoms, 2)
        if a.cats & b.cats or (a.key, b.key) in seen:
            continue
        seen.add((a.key, b.key))
        rule = Rule(f"{a.key}&!{b.key}", "and-not",
                    a.text[:-1] + " but " + b.text.replace("The image contains", "no").replace(" a ", " ", 1)
                    .replace(" an ", " ", 1).replace(" some ", " ", 1),
                    lambda r, a=a, b=b: a.fn(r) and not b.fn(r), a.cats | b.cats)
        if min_prev <= prev(rule) <= max_prev:
            keep.append(rule)
    return keep


# ---------------------------------------------------------------------------------------------- puzzles


@dataclass
class ZendoConfig:
    shots: int = 6  # images per group (A follows the rule, B does not)
    queries: int = 4  # half follow the rule
    negatives: str = "near"  # random | near | near-disc
    near_pool: int = 10  # near misses are drawn from this many most similar rule-breaking images
    disc_candidates: int = 12  # near-disc: candidate queries per side
    image_side: int = 336  # longer side of every image shown to the model
    families: str = "category,count,supercategory,size,clutter,and-not"  # train families; spatial is held out
    min_prev: float = 0.03
    max_prev: float = 0.5
    combos: int = 40
    max_words: int = 15
    length_penalty: float = 0.05


@dataclass
class Puzzle:
    rule: str
    family: str
    text: str
    pos: list[int]
    neg: list[int]
    queries: list[int]
    labels: list[bool]
    meta: dict[str, Any] = field(default_factory=dict)


class Bank:
    """Records, rule library, truth table and category-count vectors (for near misses and the oracle solver)."""

    def __init__(self, recs: list[Rec], rules: list[Rule]):
        self.recs, self.rules = recs, rules
        self.cats = sorted({c for r in recs for c in r.counts})
        ci = {c: i for i, c in enumerate(self.cats)}
        self.vec = np.zeros((len(recs), len(self.cats)), np.float32)
        for i, r in enumerate(recs):
            for c, k in r.counts.items():
                self.vec[i, ci[c]] = np.log1p(k)
        self.cat_index = ci
        self.truth = np.array([[rule.fn(r) for r in recs] for rule in rules], dtype=bool)  # rules x images
        self.by_key = {r.key: i for i, r in enumerate(rules)}

    def mask(self, ignore: frozenset[str]) -> np.ndarray:
        mask = np.ones(len(self.cats), np.float32)
        for c in ignore:
            if c in self.cat_index:
                mask[self.cat_index[c]] = 0
        return mask

    def similar(self, i: int, cands: np.ndarray, ignore: frozenset[str]) -> np.ndarray:
        """Candidates sorted by cosine similarity of category vectors to image i, rule categories masked out."""
        mask = self.mask(ignore)
        cands = np.asarray(cands, dtype=np.int64)
        a = self.vec[i] * mask
        b = self.vec[cands] * mask
        sim = b @ a / (np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-6)
        return cands[np.argsort(-sim, kind="stable")]


def make_puzzle(cfg: ZendoConfig, bank: Bank, rng: random.Random, families: list[str]) -> Puzzle:
    fam_rules = [i for i, r in enumerate(bank.rules) if r.family in families]
    ri = rng.choice(fam_rules)
    rule = bank.rules[ri]
    t = bank.truth[ri]
    pos_all, neg_all = np.flatnonzero(t), np.flatnonzero(~t)
    q = cfg.queries // 2
    pos = [int(x) for x in rng.sample(list(pos_all), cfg.shots + (q if cfg.negatives != "near-disc" else 0))]
    used = set(pos)

    def near_neg(anchor: int) -> int:
        pool = [int(x) for x in bank.similar(anchor, rng.sample(list(neg_all), min(len(neg_all), 600))
                                             if len(neg_all) > 600 else neg_all, rule.cats)
                if int(x) not in used][: cfg.near_pool]
        j = rng.choice(pool)
        used.add(j)
        return j

    if cfg.negatives == "random":
        neg = [int(x) for x in rng.sample([x for x in neg_all if x not in used], cfg.shots + q)]
        used |= set(neg)
    else:
        neg = [near_neg(p) for p in pos]
    if cfg.negatives == "near-disc":
        # Queries that contradict the most other library rules consistent with the examples.
        ex_pos, ex_neg = pos[: cfg.shots], neg[: cfg.shots]
        fit = np.flatnonzero(bank.truth[:, ex_pos].all(1) & ~bank.truth[:, ex_neg].any(1))
        alts = [k for k in fit if k != ri]
        cp = [int(x) for x in rng.sample([x for x in pos_all if x not in used], min(cfg.disc_candidates, len(pos_all) - len(used)))]
        cn = [near_neg(p) for p in cp]
        def contradicts(img: int, label: bool) -> np.ndarray:
            return bank.truth[alts, img] != label if alts else np.zeros(0, bool)
        chosen_p, chosen_n = [], []
        covered = np.zeros(len(alts), bool)
        for _ in range(q):
            best = max((c for c in cp if c not in chosen_p), key=lambda c: (contradicts(c, True) & ~covered).sum())
            chosen_p.append(best); covered |= contradicts(best, True)
            best = max((c for c in cn if c not in chosen_n), key=lambda c: (contradicts(c, False) & ~covered).sum())
            chosen_n.append(best); covered |= contradicts(best, False)
        pos, neg = pos[: cfg.shots] + chosen_p, neg[: cfg.shots] + chosen_n
    queries = pos[cfg.shots:] + neg[cfg.shots:]
    labels = [True] * q + [False] * q
    order = list(range(len(queries)))
    rng.shuffle(order)
    queries, labels = [queries[k] for k in order], [labels[k] for k in order]
    ex_pos, ex_neg = pos[: cfg.shots], neg[: cfg.shots]
    rng.shuffle(ex_pos); rng.shuffle(ex_neg)
    fit = bank.truth[:, ex_pos].all(1) & ~bank.truth[:, ex_neg].any(1)
    fit_q = fit & (bank.truth[:, queries] == np.array(labels)).all(1)
    return Puzzle(rule.key, rule.family, rule.text, ex_pos, ex_neg, queries, labels,
                  meta={"fit_rules": int(fit.sum()), "fit_rules_with_queries": int(fit_q.sum()),
                        "prevalence": float(bank.truth[ri].mean())})


def make_puzzles(cfg: ZendoConfig, bank: Bank, seed: int, split: str, n: int) -> list[Puzzle]:
    fams = [f for f in cfg.families.split(",") if f] if split == "train" else ["spatial"]
    out = []
    for slot in range(n):
        rng = random.Random(f"{seed}:zendo:{split}:{slot}")
        out.append(make_puzzle(cfg, bank, rng, fams))
    return out


# ---------------------------------------------------------------------------------------------- prompts

Content = list[dict[str, str]]

Z1_PROMPT = (
    "There is a hidden rule about what an image shows. Every image in group A follows the rule; no image in group B "
    "follows it. Work out the rule, then decide for each query image whether it follows the rule.\n"
    "Reply in exactly two lines:\nRule: <the rule in one sentence>\nAnswers: <{q} words, yes or no, for Q1 to Q{q} in order>"
)
Z2_PROMPT = (
    "There is a hidden rule about what an image shows. Every image in group A follows the rule; no image in group B "
    "follows it. State the rule in at most {w} words so that someone who sees one new image at a time can tell whether "
    "it follows the rule. Do not refer to the groups or to image numbers.\nReply in one line:\nRule: <the rule>"
)
LISTENER_PROMPT = "Rule: \"{rule}\"\nDoes this image follow the rule? Answer yes or no."


class Images:
    def __init__(self, root: Path, recs: list[Rec], side: int):
        self.root, self.recs, self.side, self.cache = Path(root), recs, side, {}

    def __getitem__(self, i: int) -> Image.Image:
        if i not in self.cache:
            img = Image.open(self.root / self.recs[i].name).convert("RGB")
            img.thumbnail((self.side, self.side), Image.Resampling.BICUBIC)
            self.cache[i] = img
        return self.cache[i]


def examples_content(pz: Puzzle, ims: Images) -> tuple[Content, list[Image.Image]]:
    content: Content = [{"type": "text", "text": "Group A (follow the rule):\n"}]
    images = []
    for k, i in enumerate(pz.pos):
        content += [{"type": "text", "text": f"A{k + 1} "}, {"type": "image"}, {"type": "text", "text": "\n"}]
        images.append(ims[i])
    content.append({"type": "text", "text": "Group B (do not follow the rule):\n"})
    for k, i in enumerate(pz.neg):
        content += [{"type": "text", "text": f"B{k + 1} "}, {"type": "image"}, {"type": "text", "text": "\n"}]
        images.append(ims[i])
    return content, images


def z1_item(pz: Puzzle, ims: Images) -> tuple[Content, list[Image.Image]]:
    content, images = examples_content(pz, ims)
    content.append({"type": "text", "text": "Query images:\n"})
    for k, i in enumerate(pz.queries):
        content += [{"type": "text", "text": f"Q{k + 1} "}, {"type": "image"}, {"type": "text", "text": "\n"}]
        images.append(ims[i])
    content.append({"type": "text", "text": Z1_PROMPT.format(q=len(pz.queries))})
    return content, images


def z1_query_only_item(pz: Puzzle, ims: Images) -> tuple[Content, list[Image.Image]]:
    """Shortcut floor: the queries without any examples (a guess from priors about 'typical' rules)."""
    content: Content = [{"type": "text", "text": "(The example groups are hidden; guess.)\nQuery images:\n"}]
    images = []
    for k, i in enumerate(pz.queries):
        content += [{"type": "text", "text": f"Q{k + 1} "}, {"type": "image"}, {"type": "text", "text": "\n"}]
        images.append(ims[i])
    content.append({"type": "text", "text": Z1_PROMPT.format(q=len(pz.queries))})
    return content, images


def z2_item(cfg: ZendoConfig, pz: Puzzle, ims: Images) -> tuple[Content, list[Image.Image]]:
    content, images = examples_content(pz, ims)
    content.append({"type": "text", "text": Z2_PROMPT.format(w=cfg.max_words)})
    return content, images


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_z1(text: str, q: int) -> tuple[str, list[bool] | None]:
    t = strip_think(text)
    m = re.search(r"rule\s*:\s*(.+)", t, flags=re.I)
    rule = m.group(1).strip() if m else ""
    m = re.search(r"answers?\s*:\s*(.+)", t, flags=re.I)
    if not m:
        return rule, None
    words = re.findall(r"\b(yes|no)\b", m.group(1).lower())
    if len(words) != q:
        return rule, None
    return rule, [w == "yes" for w in words]


def parse_rule(text: str) -> str:
    t = strip_think(text)
    m = re.search(r"rule\s*:\s*(.+)", t, flags=re.I)
    t = m.group(1) if m else (t.splitlines()[0] if t else "")
    return t.strip().strip('"').strip()


def n_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


ILLEGAL = re.compile(r"\b(group|groups|[AB]\d+|Q\d+|first|last|above|examples?)\b", re.I)


def score_z1(pz: Puzzle, text: str) -> dict[str, float]:
    _, ans = parse_z1(text, len(pz.queries))
    if ans is None:
        return {"reward": 0.0, "accuracy": 0.0, "parsed": 0.0, "yes_rate": float("nan")}
    acc = statistics.mean(float(a == l) for a, l in zip(ans, pz.labels))
    return {"reward": acc, "accuracy": acc, "parsed": 1.0, "yes_rate": statistics.mean(map(float, ans))}


def score_z2(cfg: ZendoConfig, pz: Puzzle, rule: str, p_yes: list[float]) -> dict[str, float]:
    soft = statistics.mean(1 - abs(p - float(l)) for p, l in zip(p_yes, pz.labels))
    acc = statistics.mean(float((p > 0.5) == l) for p, l in zip(p_yes, pz.labels))
    w = n_words(rule)
    illegal = bool(ILLEGAL.search(rule))
    reward = 0.0 if (w == 0 or illegal) else soft - cfg.length_penalty * max(0, w - cfg.max_words)
    return {"reward": reward, "soft": soft, "accuracy": acc, "words": float(w), "illegal": float(illegal)}


# ------------------------------------------------------------------------------------------ trivial solvers

def colour_hist(img: Image.Image, bins: int = 4) -> np.ndarray:
    px = (np.asarray(img.resize((64, 64))).reshape(-1, 3) // (256 // bins)).astype(int)
    h = np.bincount(px[:, 0] * bins * bins + px[:, 1] * bins + px[:, 2], minlength=bins ** 3).astype(np.float32)
    return h / h.sum()


def centroid_solver(feat: Callable[[int], np.ndarray], pz: Puzzle) -> float:
    """Nearest-centroid on a feature: a query 'follows the rule' if it is closer (cosine) to group A's mean."""
    def unit(v):
        return v / (np.linalg.norm(v) + 1e-6)
    a = unit(np.mean([unit(feat(i)) for i in pz.pos], 0))
    b = unit(np.mean([unit(feat(i)) for i in pz.neg], 0))
    pred = [float(unit(feat(q)) @ a > unit(feat(q)) @ b) for q in pz.queries]
    return statistics.mean(float(p == l) for p, l in zip(pred, pz.labels))


def siglip_features(name: str, ims: Images, idx: list[int], device: str) -> dict[int, np.ndarray] | None:
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(name).to(device).eval()
        proc = AutoProcessor.from_pretrained(name)
    except Exception as e:  # optional
        print("siglip unavailable:", repr(e)[:200], flush=True)
        return None
    out = {}
    for k in range(0, len(idx), 64):
        chunk = idx[k : k + 64]
        inp = proc(images=[ims[i] for i in chunk], return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_image_features(**inp)
        f = getattr(f, "pooler_output", f)
        for i, v in zip(chunk, f.float().cpu().numpy()):
            out[i] = v
    del model
    return out


# --------------------------------------------------------------------------------------------- listener

def yes_probs(model, processor, items: list[tuple[Content, list[Image.Image]]], device: str, batch: int) -> list[float]:
    """p(yes) / (p(yes) + p(no)) at the first answer token, summing case variants."""
    import torch

    from serverless.refgame import chat_text

    tok = processor.tokenizer
    tok.padding_side = "left"
    def ids(words):
        out = []
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            if len(e) == 1:
                out.append(e[0])
        return sorted(set(out))
    yes, no = ids(["yes", "Yes", "YES"]), ids(["no", "No", "NO"])
    res = []
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        inputs = processor(text=[chat_text(processor, c) for c, _ in chunk],
                           images=[im for _, ims in chunk for im in ims], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**inputs).logits[:, -1, :].float(), -1)
        py, pn = p[:, yes].sum(-1), p[:, no].sum(-1)
        res += (py / (py + pn + 1e-9)).cpu().tolist()
    return res


def listen(model, processor, cfg: ZendoConfig, pzs: list[Puzzle], rules: list[str], ims: Images, device: str,
           batch: int) -> list[dict[str, float]]:
    items, owner = [], []
    for j, (pz, rule) in enumerate(zip(pzs, rules)):
        for q in pz.queries:
            items.append(([{"type": "image"}, {"type": "text", "text": LISTENER_PROMPT.format(rule=rule or "(none)")}],
                          [ims[q]]))
            owner.append(j)
    p = yes_probs(model, processor, items, device, batch)
    per = [[] for _ in pzs]
    for j, v in zip(owner, p):
        per[j].append(v)
    return [score_z2(cfg, pz, r, pv) for pz, r, pv in zip(pzs, rules, per)]


# ---------------------------------------------------------------------------------------------- phase 0

def contact_sheet(pzs: list[Puzzle], ims: Images, notes: list[str], thumb: int = 96) -> Image.Image:
    rows = []
    for pz, note in zip(pzs, notes):
        tiles = [(ims[i], (0, 160, 0)) for i in pz.pos] + [(ims[i], (200, 0, 0)) for i in pz.neg] + \
                [(ims[i], (0, 160, 0) if l else (200, 0, 0)) for i, l in zip(pz.queries, pz.labels)]
        row = Image.new("RGB", ((thumb + 6) * len(tiles) + 20, thumb + 46), "white")
        x = 0
        for k, (im, col) in enumerate(tiles):
            if k == len(pz.pos) + len(pz.neg):
                x += 20
            t = im.copy(); t.thumbnail((thumb, thumb))
            row.paste(t, (x + 3, 3))
            ImageDraw.Draw(row).rectangle((x, 0, x + thumb + 5, thumb + 5), outline=col, width=3)
            x += thumb + 6
        ImageDraw.Draw(row).text((4, thumb + 10), f"[{pz.rule}] {pz.text}", fill="black")
        ImageDraw.Draw(row).text((4, thumb + 26), note[:200], fill="blue")
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    return sheet


def mean_of(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r[key] == r[key]]
    return statistics.mean(vals) if vals else float("nan")


def by_family(pzs: list[Puzzle], vals: list[float]) -> dict[str, float]:
    d = defaultdict(list)
    for pz, v in zip(pzs, vals):
        d[pz.family].append(v)
    return {k: round(statistics.mean(v), 3) for k, v in sorted(d.items())}


def phase0(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, sup = load_records(Path(args.instances))
    base = ZendoConfig()
    rules = build_rules(recs, sup, base.min_prev, base.max_prev, base.combos, args.seed)
    bank = Bank(recs, rules)
    fam_count = Counter(r.family for r in rules)
    print(f"{len(recs)} images, {len(rules)} rules: {dict(fam_count)}", flush=True)
    (out_dir / "rules.json").write_text(json.dumps(
        [{"key": r.key, "family": r.family, "text": r.text, "prevalence": round(float(bank.truth[k].mean()), 4)}
         for k, r in enumerate(rules)], indent=1))
    ims = Images(Path(args.images), recs, base.image_side)
    model = processor = None
    if not args.no_model:
        from serverless.refgame import load_model
        model, processor = load_model(args.model, args.device)
    schemes = dict(s.split("=", 1) for s in args.scheme) or {"near": "{}"}
    summary: dict[str, Any] = {"rules": dict(fam_count), "images": len(recs)}
    for name, spec in schemes.items():
        cfg = replace(base, **json.loads(spec))
        t0 = time.time()
        pzs = make_puzzles(cfg, bank, args.seed, args.split, args.n)
        row: dict[str, Any] = {"config": asdict(cfg)}
        row["ambiguity"] = {"fit_rules": statistics.mean(p.meta["fit_rules"] for p in pzs),
                            "unique_fit": statistics.mean(float(p.meta["fit_rules"] == 1) for p in pzs),
                            "fit_rules_with_queries": statistics.mean(p.meta["fit_rules_with_queries"] for p in pzs)}
        sol = {"colour": [centroid_solver(lambda i: colour_hist(ims[i]), p) for p in pzs],
               "category_vec": [centroid_solver(lambda i: bank.vec[i], p) for p in pzs],
               # same, with the rule's own categories masked: does correlated scene context give the rule away?
               "context_vec": [centroid_solver(lambda i, m=bank.mask(bank.rules[bank.by_key[p.rule]].cats):
                                               bank.vec[i] * m + 1e-3, p) for p in pzs]}
        if args.siglip:
            idx = sorted({i for p in pzs for i in p.pos + p.neg + p.queries})
            feats = siglip_features(args.siglip, ims, idx, args.device)
            if feats is not None:
                sol["siglip"] = [centroid_solver(lambda i: feats[i], p) for p in pzs]
        row["trivial"] = {k: round(statistics.mean(v), 3) for k, v in sol.items()}
        row["trivial_by_family"] = {k: by_family(pzs, v) for k, v in sol.items()}
        recs_out: list[dict[str, Any]] = [{"rule": p.rule, "family": p.family, "text": p.text, "pos": p.pos,
                                           "neg": p.neg, "queries": p.queries, "labels": p.labels, **p.meta,
                                           **{f"solver_{k}": v[j] for k, v in sol.items()}}
                                          for j, p in enumerate(pzs)]
        notes = [""] * len(pzs)
        if model is not None:
            from serverless.refgame import generate
            kw = dict(device=args.device, batch=args.batch)
            z1 = [z1_item(p, ims) for p in pzs]
            g = [x[0] for x in generate(model, processor, z1, 1, args.max_new_tokens, 0.0, **kw)]
            s_g = [score_z1(p, t) for p, t in zip(pzs, g)]
            samp = generate(model, processor, z1, args.group, args.max_new_tokens, 1.0, **kw)
            s_s = [[score_z1(p, t) for t in grp] for p, grp in zip(pzs, samp)]
            stds = [statistics.pstdev([s["reward"] for s in grp]) for grp in s_s]
            qo = [x[0] for x in generate(model, processor, [z1_query_only_item(p, ims) for p in pzs], 1,
                                         args.max_new_tokens, 0.0, **kw)]
            s_q = [score_z1(p, t) for p, t in zip(pzs, qo)]
            flat = [s for grp in s_s for s in grp]
            row["z1"] = {"greedy_acc": mean_of(s_g, "accuracy"), "greedy_parsed": mean_of(s_g, "parsed"),
                         "greedy_yes_rate": mean_of(s_g, "yes_rate"),
                         "sampled_acc": mean_of(flat, "accuracy"), "sampled_parsed": mean_of(flat, "parsed"),
                         "group_std": statistics.mean(stds),
                         "groups_with_signal": statistics.mean(float(s > 0.05) for s in stds),
                         "pass_at_group": statistics.mean(float(max(s["accuracy"] for s in grp) == 1.0) for grp in s_s),
                         "query_only_acc": mean_of(s_q, "accuracy")}
            row["z1_greedy_by_family"] = by_family(pzs, [s["accuracy"] for s in s_g])
            z2 = [z2_item(cfg, p, ims) for p in pzs]
            r2 = [parse_rule(x[0]) for x in generate(model, processor, z2, 1, args.max_new_tokens, 0.0, **kw)]
            s2 = listen(model, processor, cfg, pzs, r2, ims, args.device, args.batch)
            s_true = listen(model, processor, cfg, pzs, [p.text for p in pzs], ims, args.device, args.batch)
            s_none = listen(model, processor, cfg, pzs, ["(no rule given; guess)"] * len(pzs), ims, args.device,
                            args.batch)
            row["z2"] = {"greedy_soft": mean_of(s2, "soft"), "greedy_acc": mean_of(s2, "accuracy"),
                         "greedy_words": mean_of(s2, "words"), "greedy_illegal": mean_of(s2, "illegal"),
                         "true_rule_soft": mean_of(s_true, "soft"), "true_rule_acc": mean_of(s_true, "accuracy"),
                         "no_rule_acc": mean_of(s_none, "accuracy")}
            row["z2_greedy_by_family"] = by_family(pzs, [s["accuracy"] for s in s2])
            row["true_rule_by_family"] = by_family(pzs, [s["accuracy"] for s in s_true])
            for j, rec in enumerate(recs_out):
                rec.update(z1_greedy=g[j], z1_greedy_score=s_g[j], z1_samples=samp[j],
                           z1_sample_rewards=[s["reward"] for s in s_s[j]], z1_query_only=s_q[j],
                           z2_rule=r2[j], z2_score=s2[j], true_rule_score=s_true[j])
            notes = [f"Z1: {g[j][:90]!r} acc={s_g[j]['accuracy']:.2f} | Z2: {r2[j][:70]!r} acc={s2[j]['accuracy']:.2f}"
                     for j in range(len(pzs))]
        row["seconds"] = round(time.time() - t0, 1)
        summary[name] = row
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for rec in recs_out:
                f.write(json.dumps(rec, default=str) + "\n")
        contact_sheet(pzs[: args.sheet], ims, notes[: args.sheet]).save(out_dir / f"{name}_sheet.jpg", quality=85)
        print(name, json.dumps({k: v for k, v in row.items() if k != "config"}, indent=1), flush=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out_dir / "summary.json")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("phase0", help="base-model and shortcut checks before training")
    q.add_argument("--images", required=True, help="val2017 directory")
    q.add_argument("--instances", required=True, help="instances_val2017.json")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    q.add_argument("--device", default="cuda")
    q.add_argument("--n", type=int, default=300)
    q.add_argument("--split", default="train", help="train = train families, test = held-out spatial family")
    q.add_argument("--group", type=int, default=8)
    q.add_argument("--max-new-tokens", type=int, default=64)
    q.add_argument("--batch", type=int, default=4)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--sheet", type=int, default=10)
    q.add_argument("--siglip", default="google/siglip-base-patch16-224", help="'' to skip the SigLIP solver")
    q.add_argument("--no-model", action="store_true", help="puzzles + trivial solvers only (CPU)")
    q.add_argument("--scheme", action="append", default=[], help="name=JSON overrides of ZendoConfig")
    args = p.parse_args(argv)
    if args.cmd == "phase0":
        phase0(args)


if __name__ == "__main__":
    main()
