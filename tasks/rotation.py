"""Rotation Prediction: predict the clockwise angle between an image and its rotated copy."""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image

from serverless.config import ROTATION_STEPS
from serverless.images import ImageSource
from serverless.tasks.base import Puzzle


def rotate_clockwise(img: Image.Image, angle: int) -> Image.Image:
    # PIL rotates counter-clockwise for positive angles; expand keeps the whole image (black corners off-axis).
    return img.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0))


@dataclass
class RotationTask:
    steps: int = ROTATION_STEPS
    name: str = "rotation"
    kind: str = "integer"

    def __post_init__(self) -> None:
        if 360 % self.steps:
            raise ValueError(f"rotation steps must divide 360, got {self.steps}")

    def generate(self, rng: random.Random, source: ImageSource) -> Puzzle:
        idx, img = source.sample(rng)
        unit = 360 // self.steps
        angle = unit * rng.randrange(self.steps)
        parts = [
            "You are given an image and its rotated version. The image has been rotated clockwise by an angle "
            f"that is a multiple of {unit} degrees.\n",
            img,
            "\n",
            rotate_clockwise(img, angle),
            "\nTask:\nDetermine the rotation angle of the image (clockwise direction).\nAnswer format example:\n180",
        ]
        return Puzzle(self.name, parts, str(angle), "integer", {"image": source.names[idx], "angle": angle})
