"""Puzzle container and task interface shared by the SSL-R1 tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from PIL import Image

from serverless.images import ImageSource

AnswerKind = Literal["choice", "integer", "sequence"]
LETTERS = "ABCDEFGH"


@dataclass
class Puzzle:
    """One verifiable question. `parts` interleaves prompt text and images in display order."""

    task: str
    parts: list[str | Image.Image]
    answer: str
    kind: AnswerKind
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def images(self) -> list[Image.Image]:
        return [p for p in self.parts if isinstance(p, Image.Image)]

    @property
    def text(self) -> str:
        return "".join(p if isinstance(p, str) else "<image>" for p in self.parts)


class Task(Protocol):
    name: str
    kind: AnswerKind

    def generate(self, rng: random.Random, source: ImageSource) -> Puzzle: ...


def choice_parts(intro: list[str | Image.Image], candidates: list[Image.Image], task_line: str) -> list[str | Image.Image]:
    """Fig. 5 multiple-choice layout: intro, task, '(A) <image>' lines, single-letter instruction."""
    parts = list(intro) + [f"\nTask:\n{task_line}\nSelect the correct answer from the following {len(candidates)} choices:"]
    for letter, img in zip(LETTERS, candidates):
        parts += [f"\n({letter}) ", img]
    parts.append("\nThe final answer should be a single letter.")
    return parts


def shuffle_choices(rng: random.Random, correct: Image.Image, distractors: list[Image.Image]) -> tuple[list[Image.Image], str]:
    order = list(range(len(distractors) + 1))
    rng.shuffle(order)
    pool = [correct] + distractors
    candidates = [pool[i] for i in order]
    return candidates, LETTERS[order.index(0)]
