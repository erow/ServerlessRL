"""Region Inpainting: choose the region that fills a black mask (distractors are other regions of the same image)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image

from serverless.config import INPAINT_MIN_SIDE, INPAINT_RATIO, NUM_CHOICES
from serverless.images import ImageSource
from serverless.tasks.base import Puzzle, choice_parts, shuffle_choices

Box = tuple[int, int, int, int]


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


@dataclass
class InpaintingTask:
    ratio: float = INPAINT_RATIO
    min_side: int = INPAINT_MIN_SIDE
    choices: int = NUM_CHOICES
    name: str = "inpainting"
    kind: str = "choice"

    def generate(self, rng: random.Random, source: ImageSource) -> Puzzle:
        for _ in range(10):
            idx, img = source.sample(rng)
            width, height = img.size
            try:
                w, h = sample_mask_size(rng, width, height, self.ratio, self.min_side)
                gt = random_box(rng, width, height, w, h)
                others = distractor_boxes(rng, width, height, gt, self.choices - 1)
            except ValueError:
                continue
            break
        else:
            raise RuntimeError("inpainting: no image admitted distinct regions")
        masked = img.copy()
        masked.paste((0, 0, 0), gt)
        candidates, answer = shuffle_choices(rng, img.crop(gt), [img.crop(b) for b in others])
        parts = choice_parts(
            ["You are given an image with a missing part.\n", masked],
            candidates,
            "Determine the candidate image that best fits the missing part of the original image.",
        )
        meta = {"image": source.names[idx], "mask_box": list(gt), "distractor_boxes": [list(b) for b in others]}
        return Puzzle(self.name, parts, answer, "choice", meta)
