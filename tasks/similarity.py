"""Visual Similarity: pick the augmented view of the same image among four candidates (SimCLR augmentations)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from serverless.config import NUM_CHOICES, SIM_CROP
from serverless.images import ImageSource
from serverless.tasks.base import Puzzle, choice_parts, shuffle_choices


def random_resized_crop(
    rng: random.Random, img: Image.Image, size: int, scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> Image.Image:
    """torchvision RandomResizedCrop sampling, implemented with PIL."""
    width, height = img.size
    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target = area * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(*log_ratio))
        w = round(math.sqrt(target * aspect))
        h = round(math.sqrt(target / aspect))
        if 0 < w <= width and 0 < h <= height:
            x, y = rng.randint(0, width - w), rng.randint(0, height - h)
            break
    else:  # central crop fallback
        in_ratio = width / height
        if in_ratio < ratio[0]:
            w, h = width, round(width / ratio[0])
        elif in_ratio > ratio[1]:
            h, w = height, round(height * ratio[1])
        else:
            w, h = width, height
        x, y = (width - w) // 2, (height - h) // 2
    return img.resize((size, size), Image.Resampling.BICUBIC, box=(x, y, x + w, y + h))


def _shift_hue(img: Image.Image, shift: float) -> Image.Image:
    hsv = np.array(img.convert("HSV"), dtype=np.uint8)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + round(shift * 255)) % 256
    return Image.fromarray(hsv, "HSV").convert("RGB")


def color_jitter(rng: random.Random, img: Image.Image, strength: float = 1.0) -> Image.Image:
    """ColorJitter(0.8s, 0.8s, 0.8s, 0.2s) with ops in random order."""
    b, c, s, h = 0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength
    ops = [
        lambda im: ImageEnhance.Brightness(im).enhance(rng.uniform(max(0.0, 1 - b), 1 + b)),
        lambda im: ImageEnhance.Contrast(im).enhance(rng.uniform(max(0.0, 1 - c), 1 + c)),
        lambda im: ImageEnhance.Color(im).enhance(rng.uniform(max(0.0, 1 - s), 1 + s)),
        lambda im: _shift_hue(im, rng.uniform(-h, h)),
    ]
    rng.shuffle(ops)
    for op in ops:
        img = op(img)
    return img


def simclr_view(rng: random.Random, img: Image.Image, size: int = SIM_CROP, weak: bool = False) -> Image.Image:
    """SimCLR view; `weak` keeps only crop and flip (the paper's easier setting)."""
    view = random_resized_crop(rng, img, size, scale=(0.2, 1.0) if weak else (0.08, 1.0))
    if rng.random() < 0.5:
        view = ImageOps.mirror(view)
    if weak:
        return view
    if rng.random() < 0.8:
        view = color_jitter(rng, view)
    if rng.random() < 0.2:
        view = ImageOps.grayscale(view).convert("RGB")
    if rng.random() < 0.5:
        view = view.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 2.0)))
    return view


@dataclass
class SimilarityTask:
    weak: bool = False
    size: int = SIM_CROP
    choices: int = NUM_CHOICES
    name: str = "similarity"
    kind: str = "choice"

    def generate(self, rng: random.Random, source: ImageSource) -> Puzzle:
        idx, img = source.sample(rng)
        reference = simclr_view(rng, img, self.size, self.weak)
        positive = simclr_view(rng, img, self.size, self.weak)
        used, negatives = {idx}, []
        for _ in range(self.choices - 1):
            neg_idx, neg = source.sample(rng, exclude=used)
            used.add(neg_idx)
            negatives.append(simclr_view(rng, neg, self.size, self.weak))
        candidates, answer = shuffle_choices(rng, positive, negatives)
        intro = [
            f"You are given one reference image and {_number(self.choices)} candidate images. Each image may include "
            "augmentations (e.g., random cropping, flipping, color adjustments, grayscale conversion, or blurring).\n"
            "Here is the reference image:\n",
            reference,
        ]
        parts = choice_parts(
            intro,
            candidates,
            "Compare the reference image with each candidate image and determine which candidate image is most "
            "visually similar to the reference image.",
        )
        meta = {"image": source.names[idx], "negatives": [source.names[i] for i in sorted(used - {idx})]}
        return Puzzle(self.name, parts, answer, "choice", meta)


def _number(n: int) -> str:
    return {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(n, str(n))
