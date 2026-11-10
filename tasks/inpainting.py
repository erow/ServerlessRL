"""Region Inpainting: choose the region that fills a black mask.

Default (SSL-R1): a rectangle of area ratio*H*W, distractors are other regions of the same image.

Options for the hard-negative study (all deterministic given the puzzle's rng):

- `mask="square"`: SpatialSSRL's square hole of side s = min(H, W) // 2 at a uniform position.
- `distractors`: one kind for every slot, or a comma list with one kind per distractor slot. Kinds:
    same-image  same-size region elsewhere in the same image (SSL-R1)
    rot90       the true patch rotated by 90 or 270 degrees (SpatialSSRL)
    interior    the central half of the true patch, resized up (SpatialSSRL)
    exterior    the patch grown by theta in {0.25, 0.5} on every side, clipped, resized down (SpatialSSRL)
    spatial     shorthand for rot90,interior,exterior
    shift       the same-size region shifted by s/8-s/4 in a random direction (same image)
    random      same-size region of a random other image, at native resolution
    oracle-cat  other-image region whose main COCO class matches (needs `oracle_index`)
    oracle-dist other-image region with the closest COCO label composition, stuff included (needs `oracle_index`)
- `color_match`: match every candidate's colours (the true patch too) to the band around the hole, when any
  distractor comes from another image: true/"hist" = RGB histogram matching, "meanstd" = mean/std in YCbCr.
- `color_select`: for cross-image kinds, gather this many acceptable candidates and keep the one whose colours are
  closest to the band around the hole (mines colour-similar negatives without editing pixels).
- `gap`: the hole is larger than the patch by gap * min(w, h) on every side, so candidate edges never touch the
  surroundings and exact pixel continuation cannot decide the answer (a jigsaw-style gap).
- `uniform_resample`: downsample every candidate by 2 and back, then JPEG, so sharpness cannot tell them apart.
- `blind`: drop the masked image from the prompt (context-blind shortcut probe).

Cross-image distractors are filtered against false negatives: a candidate that near-duplicates the true patch is
skipped, and so is one whose border continues the hole's surroundings about as well as the true patch (boundary error
<= `boundary_filter` x the true patch's; None turns this off). Note the filter makes a pixel-level boundary matcher
solve every cross-image slot by construction; the phase-0 probe reports schemes with and without it.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image

from serverless.config import INPAINT_MIN_SIDE, INPAINT_RATIO, NUM_CHOICES
from serverless.images import ImageSource
from serverless.tasks.base import Puzzle, choice_parts, shuffle_choices

Box = tuple[int, int, int, int]

KINDS = ("same-image", "rot90", "interior", "exterior", "shift", "random", "oracle-cat", "oracle-dist")
CROSS_IMAGE = ("random", "oracle-cat", "oracle-dist")


def sample_mask_size(rng: random.Random, width: int, height: int, ratio: float, min_side: int) -> tuple[int, int]:
    """h x w ~= ratio * H * W with both sides >= min_side (Sec. 3.2.1)."""
    area = ratio * width * height
    lo = max(min_side, int(-(-area // width)))
    hi = min(height, int(area // min_side))
    if lo > hi:
        raise ValueError(f"cannot fit ratio {ratio} in {width}x{height}")
    h = rng.randint(lo, hi)
    w = min(width, max(min_side, round(area / h)))
    return w, h


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 1.0


def random_box(rng: random.Random, width: int, height: int, w: int, h: int) -> Box:
    x, y = rng.randint(0, width - w), rng.randint(0, height - h)
    return x, y, x + w, y + h


def distractor_boxes(rng: random.Random, width: int, height: int, gt: Box, count: int) -> list[Box]:
    """Same-size boxes that differ in position; the overlap limit relaxes only when the image leaves no room."""
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    for limit in (0.5, 0.7, 0.9):
        boxes: list[Box] = []
        for _ in range(200):
            box = random_box(rng, width, height, w, h)
            if all(iou(box, other) < limit for other in [gt, *boxes]):
                boxes.append(box)
                if len(boxes) == count:
                    return boxes
    raise ValueError("could not place distinct distractor regions")


def parse_kinds(spec: str, count: int) -> list[str]:
    kinds = [k.strip() for k in spec.split(",") if k.strip()]
    kinds = [x for k in kinds for x in (("rot90", "interior", "exterior") if k == "spatial" else (k,))]
    if len(kinds) == 1:
        kinds = kinds * count
    if len(kinds) != count:
        raise ValueError(f"distractors {spec!r}: need 1 or {count} kinds, got {len(kinds)}")
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        raise ValueError(f"unknown distractor kinds {unknown}; choose from {KINDS} or 'spatial'")
    return kinds


def uniform_resample(img: Image.Image, factor: float = 0.5, quality: int = 75) -> Image.Image:
    small = img.resize((max(1, round(img.width * factor)), max(1, round(img.height * factor))),
                       Image.Resampling.BICUBIC)
    back = small.resize(img.size, Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    back.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


@dataclass
class InpaintingTask:
    ratio: float = INPAINT_RATIO
    min_side: int = INPAINT_MIN_SIDE
    choices: int = NUM_CHOICES
    mask: str = "ratio"
    distractors: str = "same-image"
    oracle_index: str | None = None
    oracle_pool: int = 300
    boundary_filter: float | None = 1.0
    color_match: bool | str = False
    color_select: int = 1
    gap: float = 0.0
    uniform_resample: bool = False
    blind: bool = False
    name: str = "inpainting"
    kind: str = "choice"

    def __post_init__(self) -> None:
        if self.mask not in ("ratio", "square"):
            raise ValueError(f"mask must be 'ratio' or 'square', got {self.mask!r}")
        self.kinds = parse_kinds(self.distractors, self.choices - 1)
        if any(k.startswith("oracle") for k in self.kinds) and not self.oracle_index:
            raise ValueError("oracle distractors need oracle_index (python -m serverless.hardneg build-index)")

    def _mask_box(self, rng: random.Random, width: int, height: int) -> Box:
        if self.mask == "square":
            s = min(width, height) // 2
            return random_box(rng, width, height, s, s)
        w, h = sample_mask_size(rng, width, height, self.ratio, self.min_side)
        return random_box(rng, width, height, w, h)

    def _hole(self, gt: Box, width: int, height: int) -> Box:
        g = round(self.gap * min(gt[2] - gt[0], gt[3] - gt[1]))
        return max(0, gt[0] - g), max(0, gt[1] - g), min(width, gt[2] + g), min(height, gt[3] + g)

    def generate(self, rng: random.Random, source: ImageSource) -> Puzzle:
        for _ in range(10):
            idx, img = source.sample(rng)
            width, height = img.size
            try:
                gt = self._mask_box(rng, width, height)
                hole = self._hole(gt, width, height)
                patches, info = self._distractors(rng, source, idx, img, gt, hole)
            except ValueError:
                continue
            break
        else:
            raise RuntimeError("inpainting: no image admitted distinct distractors")
        masked = img.copy()
        masked.paste((0, 0, 0), hole)
        truth = info.pop("truth", None) or img.crop(gt)
        if self.uniform_resample:
            truth, patches = uniform_resample(truth), [uniform_resample(p) for p in patches]
        candidates, answer = shuffle_choices(rng, truth, patches)
        if self.blind:
            parts = choice_parts(["One of the following candidate images was cut out of a photograph that is not "
                                  "shown; the others are distractors.\n"], candidates,
                                 "Determine the candidate image that is the missing part of the photograph.")
        else:
            parts = choice_parts(["You are given an image with a missing part.\n", masked], candidates,
                                 "Determine the candidate image that best fits the missing part of the original image.")
        meta = {"image": source.names[idx], "mask_box": list(hole), "distractor_kinds": list(self.kinds), **info}
        if hole != gt:
            meta["patch_box"] = list(gt)
        return Puzzle(self.name, parts, answer, "choice", meta)

    # ------------------------------------------------------------------------------------------- distractors

    def _distractors(self, rng: random.Random, source: ImageSource, idx: int, img: Image.Image,
                     gt: Box, hole: Box) -> tuple[list[Image.Image], dict]:
        from serverless import hardneg

        width, height = img.size
        x0, y0, x1, y1 = gt
        w, h = x1 - x0, y1 - y0
        truth = img.crop(gt)
        masked = img.copy()
        masked.paste((0, 0, 0), hole)
        cross = any(k in CROSS_IMAGE for k in self.kinds)
        gt_err = hardneg.boundary_error(masked, hole, truth, inner=gt) if cross else None
        ring = hardneg.ring_pixels(masked, hole, max(4, min(w, h) // 4)) if cross else None
        ring_hist = hardneg.color_hist(ring) if cross else None
        patches: list[Image.Image] = []
        sources: list[dict] = []
        used_images = {idx}

        same = [k for k in self.kinds if k == "same-image"]
        same_boxes = iter(distractor_boxes(rng, width, height, gt, len(same))) if same else iter(())

        def accept(patch: Image.Image) -> bool:
            if hardneg.near_duplicate(patch, truth):
                return False
            if self.boundary_filter is None:
                return True
            return hardneg.boundary_error(masked, hole, patch, inner=gt) > self.boundary_filter * gt_err

        def closest_color(found: list[tuple]) -> tuple:
            return min(found, key=lambda f: hardneg.color_distance(f[0], ring_hist))

        def base(i: int) -> str:
            return source.names[i].rsplit("/", 1)[-1]

        for kind in self.kinds:
            if kind == "same-image":
                box = next(same_boxes)
                patches.append(img.crop(box))
                sources.append({"kind": kind, "box": list(box)})
            elif kind == "rot90":
                angle = rng.choice((90, 270))
                patches.append(truth.rotate(angle, expand=True).resize((w, h), Image.Resampling.BICUBIC))
                sources.append({"kind": kind, "angle": angle})
            elif kind == "interior":
                box = (x0 + w // 4, y0 + h // 4, x0 + 3 * w // 4, y0 + 3 * h // 4)
                patches.append(img.crop(box).resize((w, h), Image.Resampling.BICUBIC))
                sources.append({"kind": kind, "box": list(box)})
            elif kind == "exterior":
                theta = rng.choice((0.25, 0.5))
                box = (max(0, round(x0 - theta * w)), max(0, round(y0 - theta * h)),
                       min(width, round(x1 + theta * w)), min(height, round(y1 + theta * h)))
                patches.append(img.crop(box).resize((w, h), Image.Resampling.BICUBIC))
                sources.append({"kind": kind, "box": list(box), "theta": theta})
            elif kind == "shift":
                for _ in range(50):
                    d = rng.uniform(0.125, 0.25)
                    dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)))
                    nx, ny = x0 + round(dx * d * w), y0 + round(dy * d * h)
                    if 0 <= nx <= width - w and 0 <= ny <= height - h and (nx, ny) != (x0, y0):
                        break
                else:
                    raise ValueError("no room to shift the region")
                box = (nx, ny, nx + w, ny + h)
                patches.append(img.crop(box))
                sources.append({"kind": kind, "box": list(box)})
            elif kind == "random":
                found = []
                for _ in range(30 * self.color_select):
                    j, other = source.sample(rng, exclude=used_images)
                    if other.width < w or other.height < h:
                        continue
                    box = random_box(rng, other.width, other.height, w, h)
                    patch = other.crop(box)
                    if accept(patch):
                        found.append((patch, j, box))
                        if len(found) == self.color_select:
                            break
                if not found:
                    raise ValueError("no random cross-image distractor passed the filters")
                patch, j, box = closest_color(found)
                used_images.add(j)
                patches.append(patch)
                sources.append({"kind": kind, "image": source.names[j], "box": list(box)})
            else:  # oracle-cat / oracle-dist
                index = hardneg.LabelIndex.load(self.oracle_index)
                pool = [rng.randrange(len(source)) for _ in range(self.oracle_pool)]
                matches = hardneg.oracle_candidates(index, base(idx), gt, kind.split("-")[1],
                                                    [base(i) for i in pool if i not in used_images])
                found = []
                for m in matches:
                    j = source.index_of(m.name)
                    if j is None or j in used_images:
                        continue
                    other = source.load(j)
                    bx0, by0 = min(m.box[0], other.width - w), min(m.box[1], other.height - h)
                    if bx0 < 0 or by0 < 0:
                        continue
                    box = (bx0, by0, bx0 + w, by0 + h)
                    patch = other.crop(box)
                    if accept(patch):
                        found.append((patch, j, box, m))
                        if len(found) == self.color_select:
                            break
                if not found:
                    raise ValueError(f"no {kind} distractor passed the filters")
                patch, j, box, m = closest_color(found)
                used_images.add(j)
                patches.append(patch)
                sources.append({"kind": kind, "image": m.name, "box": list(box), "score": m.score,
                                "label": index.category_names.get(m.label, str(m.label))})
        info: dict = {"distractors": sources}
        if gt_err is not None:
            info["gt_boundary_error"] = gt_err
        if self.color_match and cross:
            # every candidate, the true patch included, so "the only unmatched one" is not a tell
            method = "hist" if self.color_match is True else self.color_match
            info["truth"] = hardneg.match_colors(truth, ring, method)
            patches = [hardneg.match_colors(p, ring, method) for p in patches]
        return patches, info
