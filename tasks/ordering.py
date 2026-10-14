"""Patch Ordering: reassemble shuffled M x N tiles; answer lists shown-tile indices in raster order."""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image

from serverless.config import ORDER_GRID
from serverless.images import ImageSource
from serverless.tasks.base import Puzzle

_NUMBERS = {4: "four", 6: "six", 9: "nine", 16: "sixteen"}
_PAPER_EXAMPLE = "5, 1, 3, 7, 9, 2, 4, 8, 6"


def split_tiles(img: Image.Image, rows: int, cols: int) -> list[Image.Image]:
    """Raster-order tiles of equal size; the remainder pixels on the right/bottom are dropped."""
    tw, th = img.width // cols, img.height // rows
    return [img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)) for r in range(rows) for c in range(cols)]


def example_answer(n: int) -> str:
    if n == 9:
        return _PAPER_EXAMPLE
    perm = list(range(1, n + 1))
    random.Random(n).shuffle(perm)
    return ", ".join(map(str, perm))


@dataclass
class OrderingTask:
    grid: int = ORDER_GRID
    name: str = "ordering"
    kind: str = "sequence"

    def generate(self, rng: random.Random, source: ImageSource) -> Puzzle:
        idx, img = source.sample(rng)
        tiles = split_tiles(img, self.grid, self.grid)
        n = len(tiles)
        shown_from = list(range(n))  # shown tile t holds original patch shown_from[t]
        rng.shuffle(shown_from)
        answer = [shown_from.index(j) + 1 for j in range(n)]
        layout = f"{self.grid}*{self.grid}"
        parts: list[str | Image.Image] = [
            f"You are given {_NUMBERS.get(n, str(n))} shuffled image tiles that were created by slicing one image "
            f"into a {layout} grid.\nHere are the tiles, each tagged with an index reflecting the current (shuffled) "
            "order in which they are shown:"
        ]
        for t in range(n):
            parts += [f"\nTile {t + 1}: ", tiles[shown_from[t]]]
        parts.append(
            f"\nTask:\nMentally reassemble the original image, arranging the tiles into the correct {layout} layout "
            "and provide the tile indices in raster-scan order (left-to-right, top-to-bottom), separated by commas.\n"
            f"Answer format example:\n{example_answer(n)}"
        )
        meta = {"image": source.names[idx], "shown_from": shown_from}
        return Puzzle(self.name, parts, ", ".join(map(str, answer)), "sequence", meta)
