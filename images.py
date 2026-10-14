"""Streamed COCO image access (flat directory or ZIP) and inline image encoding."""

from __future__ import annotations

import base64
import io
import os
import random
import threading
import zipfile
from pathlib import Path

from PIL import Image

from serverless.config import JPEG_QUALITY, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE


class ImageSource:
    """Sorted list of JPEG names from a directory or ZIP; images are decoded on demand."""

    def __init__(self, root: Path, max_side: int = MAX_IMAGE_SIDE, min_side: int = MIN_IMAGE_SIDE):
        self.root = Path(root)
        self.max_side = max_side
        self.min_side = min_side
        self._local = threading.local()
        if self.root.suffix == ".zip":
            with zipfile.ZipFile(self.root) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
        else:
            with os.scandir(self.root) as it:
                names = [e.name for e in it if e.name.lower().endswith(".jpg")]
        if not names:
            raise FileNotFoundError(f"no JPEG images under {self.root}")
        self.names = sorted(names)

    def __len__(self) -> int:
        return len(self.names)

    def _read(self, name: str) -> bytes:
        if self.root.suffix != ".zip":
            return (self.root / name).read_bytes()
        zf = getattr(self._local, "zf", None)
        if zf is None:
            zf = self._local.zf = zipfile.ZipFile(self.root)
        return zf.read(name)

    def load(self, index: int) -> Image.Image:
        img = Image.open(io.BytesIO(self._read(self.names[index])))
        img = img.convert("RGB")
        if max(img.size) > self.max_side:
            img.thumbnail((self.max_side, self.max_side), Image.Resampling.BICUBIC)
        return img

    def sample(
        self, rng: random.Random, exclude: set[int] = frozenset(), attempts: int = 50,
    ) -> tuple[int, Image.Image]:
        """Draw a usable image; skips unreadable or too-small images deterministically."""
        for _ in range(attempts):
            idx = rng.randrange(len(self))
            if idx in exclude:
                continue
            try:
                img = self.load(idx)
            except (OSError, zipfile.BadZipFile):
                continue
            if min(img.size) >= self.min_side:
                return idx, img
        raise RuntimeError(f"no usable image after {attempts} draws from {self.root}")


def to_data_uri(img: Image.Image, quality: int = JPEG_QUALITY) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
