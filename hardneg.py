"""Hard negatives for region inpainting: an oracle label index over COCO, distractor filters, and shortcut probes.

The oracle picks distractors from *other* images using COCO annotations: each image is reduced to a coarse label map
(one label per 16x16 cell: 0 background, otherwise the COCO category id painted on top there; stuff first, then
things from largest to smallest), and a window's label histogram stands for "what is in this region".

  # build the index once per image set (instances, plus COCO-Stuff if available)
  python -m serverless.hardneg build-index --instances annotations/instances_val2017.json \\
      --stuff annotations/stuff_val2017.json --out val2017_labels.npz

  # phase-0 shortcut probes: trivial solvers on several distractor schemes, plus preview sheets
  python -m serverless.hardneg probe --images val2017.zip --n 500 --out-dir probe/ \\
      --scheme spatial='{"mask": "square", "distractors": "spatial"}' \\
      --scheme random='{"mask": "square", "distractors": "random"}'
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

CELL = 16  # pixels per label cell
NUM_LABELS = 256  # label ids are COCO category ids (things 1-90, stuff 92-182), 0 = unlabelled

Box = tuple[int, int, int, int]


# ---------------------------------------------------------------------------------------------- label index


def _rasterize(ann: dict, shape: tuple[int, int], scale: float) -> np.ndarray | None:
    """Annotation mask at `scale` of the image size; polygons via PIL, RLE via pycocotools when installed."""
    h, w = shape
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in seg:
            if len(poly) >= 6:
                draw.polygon([(x * scale, y * scale) for x, y in zip(poly[::2], poly[1::2])], fill=1)
        return np.asarray(canvas, dtype=bool)
    if isinstance(seg, dict):
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return None
        rle = seg if isinstance(seg["counts"], str) else mask_utils.frPyObjects(seg, *seg["size"])
        full = mask_utils.decode(rle).astype(bool)
        img = Image.fromarray(full.astype(np.uint8) * 255).resize((w, h), Image.Resampling.BOX)
        return np.asarray(img) >= 128
    return None


def build_index(instance_files: list[Path], stuff_files: list[Path], out: Path, max_side: int = 640) -> None:
    """Coarse label maps for every annotated image, keyed by file name, saved as one compressed npz."""
    images: dict[int, dict] = {}
    anns: dict[int, list[tuple[int, dict]]] = {}
    categories: dict[int, str] = {}
    for layer, files in ((0, stuff_files), (1, instance_files)):
        for path in files:
            data = json.loads(Path(path).read_text())
            images.update({im["id"]: im for im in data["images"]})
            categories.update({c["id"]: c["name"] for c in data["categories"]})
            for a in data["annotations"]:
                if a["category_id"] == 183:  # COCO-Stuff "other": unlabelled
                    continue
                anns.setdefault(a["image_id"], []).append((layer, a))
    names, offsets, shapes, flat = [], [0], [], []
    skipped_rle = 0
    for image_id, im in sorted(images.items(), key=lambda kv: kv[1]["file_name"]):
        w, h = im["width"], im["height"]
        resize = min(1.0, max_side / max(w, h))  # matches ImageSource's thumbnail
        scale = resize / CELL
        gh, gw = max(1, math.ceil(h * resize / CELL)), max(1, math.ceil(w * resize / CELL))
        grid = np.zeros((gh, gw), dtype=np.uint8)
        # stuff under things; larger things first so small objects stay visible
        for layer, a in sorted(anns.get(image_id, []), key=lambda la: (la[0], -la[1].get("area", 0))):
            mask = _rasterize(a, (gh, gw), scale)
            if mask is None:
                skipped_rle += 1
                continue
            grid[mask] = a["category_id"]
        names.append(im["file_name"])
        shapes.append((gh, gw))
        flat.append(grid.ravel())
        offsets.append(offsets[-1] + grid.size)
    np.savez_compressed(out, names=np.array(names), offsets=np.array(offsets, dtype=np.int64),
                        shapes=np.array(shapes, dtype=np.int32), labels=np.concatenate(flat),
                        category_ids=np.array(sorted(categories)), category_names=np.array(
                            [categories[k] for k in sorted(categories)]))
    print(f"{len(names)} images -> {out} ({skipped_rle} RLE masks skipped: install pycocotools to keep them)")


class LabelIndex:
    _cache: dict[str, LabelIndex] = {}

    def __init__(self, path: Path):
        data = np.load(path)
        self.names = list(data["names"])
        self.by_name = {n: i for i, n in enumerate(self.names)}
        self.offsets, self.shapes, self.labels = data["offsets"], data["shapes"], data["labels"]
        self.category_names = dict(zip(data["category_ids"].tolist(), data["category_names"].tolist()))

    @classmethod
    def load(cls, path: str | Path) -> LabelIndex:
        key = str(path)
        if key not in cls._cache:
            cls._cache[key] = cls(Path(path))
        return cls._cache[key]

    def grid(self, name: str) -> np.ndarray | None:
        i = self.by_name.get(name)
        if i is None:
            return None
        return self.labels[self.offsets[i]:self.offsets[i + 1]].reshape(self.shapes[i])


def cell_box(box: Box) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return x0 // CELL, y0 // CELL, max(x0 // CELL + 1, -(-x1 // CELL)), max(y0 // CELL + 1, -(-y1 // CELL))


def histogram(grid: np.ndarray, cbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = cbox
    window = grid[y0:y1, x0:x1]
    hist = np.bincount(window.ravel(), minlength=NUM_LABELS).astype(np.float64)
    return hist / max(1.0, hist.sum())


def window_histograms(grid: np.ndarray, cw: int, ch: int, stride: int = 2) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Label histograms of every cw x ch cell window (top-left on a `stride` lattice) via per-label integral images."""
    gh, gw = grid.shape
    if cw > gw or ch > gh:
        return np.zeros((0, NUM_LABELS)), []
    present = np.unique(grid)
    ys = list(range(0, gh - ch + 1, stride)) or [0]
    xs = list(range(0, gw - cw + 1, stride)) or [0]
    if ys[-1] != gh - ch:
        ys.append(gh - ch)
    if xs[-1] != gw - cw:
        xs.append(gw - cw)
    hists = np.zeros((len(ys) * len(xs), NUM_LABELS))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy, xx = yy.ravel(), xx.ravel()
    for label in present:
        integral = np.pad((grid == label).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        hists[:, label] = (integral[yy + ch, xx + cw] - integral[yy, xx + cw] - integral[yy + ch, xx]
                           + integral[yy, xx])
    hists /= cw * ch
    return hists, list(zip(xx.tolist(), yy.tolist()))


def main_label(hist: np.ndarray) -> tuple[int, float]:
    """Largest labelled class in the window (background only when nothing is labelled)."""
    labelled = hist.copy()
    labelled[0] = 0
    if labelled.sum() < 1e-9:
        return 0, float(hist[0])
    k = int(labelled.argmax())
    return k, float(hist[k])


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence between one histogram p and rows of q (base 2, in [0, 1])."""
    m = 0.5 * (p[None] + q)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1)), 0.0)
        return t.sum(-1)

    return 0.5 * kl(np.broadcast_to(p, q.shape), m) + 0.5 * kl(q, m)


@dataclass
class OracleMatch:
    name: str
    box: Box
    score: float  # lower is more similar
    label: int


def oracle_candidates(index: LabelIndex, gt_name: str, gt_box: Box, mode: str, pool_names: list[str],
                      max_out: int = 20, cat_tolerance: float = 0.2) -> list[OracleMatch]:
    """Windows of the GT size from other images, most similar first.

    mode 'cat': same main labelled class whose share differs by < `cat_tolerance` (ranked by share difference).
    mode 'dist': smallest Jensen-Shannon divergence of the full label histogram (scene composition, stuff included).
    """
    grid = index.grid(gt_name)
    if grid is None:
        return []
    cb = cell_box(gt_box)
    gt_hist = histogram(grid, cb)
    gt_label, gt_share = main_label(gt_hist)
    cw, ch = cb[2] - cb[0], cb[3] - cb[1]
    w, h = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
    out: list[OracleMatch] = []
    for name in pool_names:
        if name == gt_name:
            continue
        g = index.grid(name)
        if g is None:
            continue
        hists, positions = window_histograms(g, cw, ch)
        if not positions:
            continue
        if mode == "cat":
            if gt_label == 0:
                continue
            diff = np.abs(hists[:, gt_label] - gt_share)
            labelled = hists.copy()
            labelled[:, 0] = 0
            ok = (labelled.argmax(1) == gt_label) & (diff < cat_tolerance)
            if not ok.any():
                continue
            j = int(np.where(ok, diff, np.inf).argmin())
            score = float(diff[j])
        elif mode == "dist":
            d = js_divergence(gt_hist, hists)
            j = int(d.argmin())
            score = float(d[j])
        else:
            raise ValueError(f"unknown oracle mode {mode!r}")
        cx, cy = positions[j]
        gh, gw = g.shape
        # pixel box of the GT size anchored at the window, clipped to the (approximate) image extent
        x0 = min(cx * CELL, max(0, gw * CELL - w))
        y0 = min(cy * CELL, max(0, gh * CELL - h))
        out.append(OracleMatch(name, (x0, y0, x0 + w, y0 + h), score, gt_label))
    out.sort(key=lambda m: m.score)
    return out[:max_out]


# --------------------------------------------------------------------------------------- pixel-level helpers


def ring_box(box: Box, size: tuple[int, int], width: int) -> Box:
    x0, y0, x1, y1 = box
    return max(0, x0 - width), max(0, y0 - width), min(size[0], x1 + width), min(size[1], y1 + width)


def ring_pixels(img: Image.Image, box: Box, width: int) -> np.ndarray:
    """Pixels in a band of `width` around `box` (the box itself excluded), as an (N, 3) array."""
    outer = ring_box(box, img.size, width)
    arr = np.asarray(img.crop(outer).convert("RGB"), dtype=np.float32)
    keep = np.ones(arr.shape[:2], dtype=bool)
    x0, y0 = box[0] - outer[0], box[1] - outer[1]
    keep[y0:y0 + box[3] - box[1], x0:x0 + box[2] - box[0]] = False
    return arr[keep]


def boundary_error(context: Image.Image, box: Box, patch: Image.Image, band: int = 2) -> float:
    """Mean absolute difference between the patch's outer `band` pixels and the context pixels just outside the hole,
    over the sides of the hole that lie inside the image. Lower = the patch continues the surroundings better."""
    x0, y0, x1, y1 = box
    W, H = context.size
    ctx = np.asarray(context.convert("RGB"), dtype=np.float32)
    p = np.asarray(patch.convert("RGB").resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR), dtype=np.float32)
    diffs = []
    if y0 - band >= 0:
        diffs.append(np.abs(ctx[y0 - band:y0, x0:x1][::-1] - p[:band]).mean())
    if y1 + band <= H:
        diffs.append(np.abs(ctx[y1:y1 + band, x0:x1] - p[-band:][::-1]).mean())
    if x0 - band >= 0:
        diffs.append(np.abs(ctx[y0:y1, x0 - band:x0][:, ::-1] - p[:, :band]).mean())
    if x1 + band <= W:
        diffs.append(np.abs(ctx[y0:y1, x1:x1 + band] - p[:, -band:][:, ::-1]).mean())
    return float(np.mean(diffs)) if diffs else 0.0


def ahash(img: Image.Image, size: int = 8) -> int:
    a = np.asarray(img.convert("L").resize((size, size), Image.Resampling.BOX), dtype=np.float32)
    bits = (a > a.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate(a: Image.Image, b: Image.Image, max_bits: int = 4) -> bool:
    return bin(ahash(a) ^ ahash(b)).count("1") <= max_bits


def match_colors(patch: Image.Image, reference: np.ndarray) -> Image.Image:
    """Per-channel histogram matching of `patch` to the reference pixels (N, 3)."""
    src = np.asarray(patch.convert("RGB"), dtype=np.float32)
    out = np.empty_like(src)
    for c in range(3):
        s = src[..., c].ravel()
        ref = np.sort(reference[:, c])
        ranks = np.argsort(np.argsort(s, kind="stable"), kind="stable") / max(1, s.size - 1)
        out[..., c] = np.interp(ranks, np.linspace(0, 1, ref.size), ref).reshape(src.shape[:2])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def laplacian_var(arr: np.ndarray) -> float:
    g = arr.mean(-1) if arr.ndim == 3 else arr
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var()) if lap.size else 0.0


# ------------------------------------------------------------------------------------------ shortcut probes


def _ring_width(box: Box) -> int:
    return max(4, min(box[2] - box[0], box[3] - box[1]) // 4)


def solve_boundary(context: Image.Image, box: Box, cands: list[Image.Image]) -> int:
    return int(np.argmin([boundary_error(context, box, c) for c in cands]))


def solve_color(context: Image.Image, box: Box, cands: list[Image.Image]) -> int:
    ring = ring_pixels(context, box, _ring_width(box))
    bins = 16

    def hist(px: np.ndarray) -> np.ndarray:
        h = np.concatenate([np.histogram(px[:, c], bins=bins, range=(0, 256))[0] for c in range(3)]).astype(float)
        return h / max(1.0, h.sum())

    ref = hist(ring)
    dists = []
    for c in cands:
        h = hist(np.asarray(c.convert("RGB"), dtype=np.float32).reshape(-1, 3))
        dists.append(0.5 * np.sum((h - ref) ** 2 / (h + ref + 1e-9)))
    return int(np.argmin(dists))


def solve_sharpness(context: Image.Image, box: Box, cands: list[Image.Image]) -> int:
    """Candidate whose high-frequency energy best matches the surrounding band (log Laplacian variance)."""
    outer = ring_box(box, context.size, _ring_width(box))
    ctx = np.asarray(context.crop(outer).convert("RGB"), dtype=np.float32)
    # zeroed hole would add fake edges; measure the band only through its four strips
    x0, y0, x1, y1 = box[0] - outer[0], box[1] - outer[1], box[2] - outer[0], box[3] - outer[1]
    strips = [s for s in (ctx[:y0], ctx[y1:], ctx[y0:y1, :x0], ctx[y0:y1, x1:]) if min(s.shape[:2]) >= 3]
    ref = statistics.fmean(laplacian_var(s) for s in strips) if strips else 1.0
    return int(np.argmin([abs(math.log1p(laplacian_var(np.asarray(c.convert("RGB"), dtype=np.float32)))
                              - math.log1p(ref)) for c in cands]))


def solve_odd_sharpness(context: Image.Image, box: Box, cands: list[Image.Image]) -> int:
    """Context-free: the candidate with the median-ranked sharpness (GT sits between the zoomed-in and zoomed-out
    SpatialSSRL distractors). Ties to whichever is closest to the median."""
    v = [math.log1p(laplacian_var(np.asarray(c.convert("RGB"), dtype=np.float32))) for c in cands]
    med = statistics.median(v)
    return int(np.argmin([abs(x - med) for x in v]))


CONTEXT_SOLVERS = ("boundary", "color", "sharpness")
SOLVERS = {
    "boundary": solve_boundary,
    "color": solve_color,
    "sharpness": solve_sharpness,
    "odd_sharpness": solve_odd_sharpness,
}


def probe(args: argparse.Namespace) -> None:
    from serverless.images import ImageSource
    from serverless.objective import Objective
    from serverless.tasks.base import LETTERS

    source = ImageSource(args.images)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for spec in args.scheme:
        label, _, params = spec.partition("=")
        params_dict = json.loads(params or "{}")
        objective = Objective.from_spec("inpainting", "paper", {"inpainting": params_dict}, args.seed)
        correct = {k: 0 for k in SOLVERS}
        n_ok, meta_counts = 0, {}
        previews = []
        for i in range(args.n):
            try:
                puzzle = objective.puzzle(source, "val", i, 0, 1)
            except RuntimeError as exc:
                print(f"{label} val-{i}-0: {exc}")
                continue
            n_ok += 1
            box = tuple(puzzle.meta["mask_box"])
            blind = params_dict.get("blind", False)
            context = None if blind else puzzle.images[0]
            cands = puzzle.images if blind else puzzle.images[1:]
            gold = LETTERS.index(puzzle.answer)
            for k, fn in SOLVERS.items():
                if context is None and k in CONTEXT_SOLVERS:
                    continue
                correct[k] += fn(context, box, cands) == gold
            for kind in puzzle.meta.get("distractor_kinds", []):
                meta_counts[kind] = meta_counts.get(kind, 0) + 1
            if len(previews) < args.preview:
                previews.append(puzzle)
        acc = {k: v / max(1, n_ok) for k, v in correct.items()
               if not (params_dict.get("blind") and k in CONTEXT_SOLVERS)}
        se = {k: math.sqrt(a * (1 - a) / max(1, n_ok)) for k, a in acc.items()}
        results[label] = {"params": params_dict, "n": n_ok, "accuracy": acc, "se": se,
                          "distractor_kinds": meta_counts}
        print(f"{label:18s} n={n_ok:4d} " + "  ".join(f"{k}={acc[k]:.3f}±{se[k]:.3f}" for k in acc)
              + f"  kinds={meta_counts}", flush=True)
        if previews:
            contact_sheet(previews, 0 if params_dict.get("blind") else 1).save(args.out_dir / f"preview_{label}.jpg", quality=90)
    (args.out_dir / "probe.json").write_text(json.dumps(results, indent=2) + "\n")


def contact_sheet(puzzles: list, first_candidate: int = 1, thumb: int = 200) -> Image.Image:
    """One row per puzzle: masked image, then the candidates; the correct one framed green."""
    from serverless.tasks.base import LETTERS

    cols = max(len(p.images) for p in puzzles)
    sheet = Image.new("RGB", (cols * (thumb + 8), len(puzzles) * (thumb + 8)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, p in enumerate(puzzles):
        for c, img in enumerate(p.images):
            t = img.copy()
            t.thumbnail((thumb, thumb))
            x, y = c * (thumb + 8) + 4, r * (thumb + 8) + 4
            sheet.paste(t, (x, y))
            if c >= first_candidate and LETTERS[c - first_candidate] == p.answer:
                draw.rectangle([x - 3, y - 3, x + t.width + 2, y + t.height + 2], outline=(0, 200, 0), width=3)
    return sheet


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-index", help="coarse COCO label maps for the oracle")
    b.add_argument("--instances", type=Path, nargs="+", required=True)
    b.add_argument("--stuff", type=Path, nargs="*", default=[])
    b.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("probe", help="trivial-solver accuracy per distractor scheme")
    q.add_argument("--images", type=Path, required=True)
    q.add_argument("--scheme", action="append", required=True, help="label=JSON inpainting params (repeatable)")
    q.add_argument("--n", type=int, default=500)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--preview", type=int, default=12, help="puzzles per preview sheet")
    q.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "build-index":
        build_index(args.instances, args.stuff, args.out)
    else:
        probe(args)


if __name__ == "__main__":
    main()
