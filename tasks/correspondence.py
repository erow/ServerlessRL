"""Geometric Correspondence: match K labelled points across a random affine transform."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from serverless.config import CORR_MIN_DIST, CORR_POINTS
from serverless.images import ImageSource
from serverless.tasks.base import Puzzle
from serverless.tasks.ordering import example_answer

_NUMBERS = {3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
_PAPER_EXAMPLE = "3, 1, 2, 6, 5, 4"


def random_affine(rng: random.Random, width: int, height: int) -> np.ndarray:
    """Forward 3x3 map from original to output pixel coordinates (rotation, shear, scale, shift about centre)."""
    cx, cy = width / 2, height / 2
    theta = math.radians(rng.uniform(-45, 45))
    shear = math.radians(rng.uniform(-15, 15))
    scale = rng.uniform(0.7, 1.0)
    tx, ty = rng.uniform(-0.1, 0.1) * width, rng.uniform(-0.1, 0.1) * height
    rot = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    shr = np.array([[1, math.tan(shear), 0], [0, 1, 0], [0, 0, 1]])
    scl = np.diag([scale, scale, 1.0])
    to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    back = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]])
    return back @ rot @ shr @ scl @ to_origin


def apply_affine(img: Image.Image, forward: np.ndarray) -> Image.Image:
    inv = np.linalg.inv(forward)  # PIL expects output -> input coefficients
    coeffs = tuple(inv[0].tolist() + inv[1].tolist())
    return img.transform(img.size, Image.Transform.AFFINE, coeffs, Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))


def map_points(forward: np.ndarray, points: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homog @ forward.T)[:, :2]


def _min_dist(points: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(points - candidate, axis=1))) if len(points) else math.inf


def sample_points(
    rng: random.Random, width: int, height: int, forward: np.ndarray, k: int, min_dist: float, margin: float,
) -> np.ndarray | None:
    """Points >= min_dist apart in the original whose images stay inside the output with a margin."""
    src = np.zeros((0, 2))
    dst = np.zeros((0, 2))
    for _ in range(2000):
        p = np.array([[rng.uniform(margin, width - margin), rng.uniform(margin, height - margin)]])
        q = map_points(forward, p)
        if not (margin <= q[0, 0] <= width - margin and margin <= q[0, 1] <= height - margin):
            continue
        if _min_dist(src, p[0]) < min_dist or _min_dist(dst, q[0]) < 2 * margin:
            continue
        src, dst = np.vstack([src, p]), np.vstack([dst, q])
        if len(src) == k:
            return src
    return None


def draw_labels(img: Image.Image, points: np.ndarray, labels: list[int], radius: int) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=round(radius * 1.3))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for (x, y), label in zip(points, labels):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(230, 25, 25), outline=(255, 255, 255))
        draw.text((x, y), str(label), fill=(255, 255, 255), font=font, anchor="mm")
    return out


@dataclass
class CorrespondenceTask:
    points: int = CORR_POINTS
    min_dist: int = CORR_MIN_DIST
    name: str = "correspondence"
    kind: str = "sequence"

    def generate(self, rng: random.Random, source: ImageSource) -> Puzzle:
        for _ in range(20):
            idx, img = source.sample(rng)
            width, height = img.size
            radius = max(9, round(0.025 * min(width, height)))
            forward = random_affine(rng, width, height)
            src = sample_points(rng, width, height, forward, self.points, self.min_dist, radius + 2)
            if src is not None:
                break
        else:
            raise RuntimeError("correspondence: could not place points")
        dst = map_points(forward, src)
        shown_from = list(range(self.points))  # label t+1 on the transformed image marks original point shown_from[t]
        rng.shuffle(shown_from)
        answer = [shown_from.index(j) + 1 for j in range(self.points)]
        first = draw_labels(img, src, list(range(1, self.points + 1)), radius)
        second = draw_labels(apply_affine(img, forward), dst[shown_from], list(range(1, self.points + 1)), radius)
        example = _PAPER_EXAMPLE if self.points == 6 else example_answer(self.points)
        parts = [
            "You are given two images: the first one is the original image and the second one is its transformed "
            f"version. {_NUMBERS.get(self.points, str(self.points))} points are marked on each of the image with red "
            "circular labels (1, 2, 3, ...).\n",
            first,
            "\n",
            second,
            "\nTask:\nFor each point (1, 2, 3, ...) in the first image, find the corresponding point in the second "
            "image.\nAnswer with the ordered sequence of point numbers, separated by commas.\n"
            f"Answer format example:\n{example}",
        ]
        meta = {
            "image": source.names[idx],
            "affine": forward[:2].round(6).tolist(),
            "points": src.round(2).tolist(),
            "shown_from": shown_from,
        }
        return Puzzle(self.name, parts, ", ".join(map(str, answer)), "sequence", meta)
