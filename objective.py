"""RL objective built by composing SSL tasks: a weighted, stratified mixture with one shared reward definition."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from serverless.config import FORMAT_PROMPT, FORMAT_REWARD
from serverless.images import ImageSource, to_data_uri
from serverless.rewards import REQUIRE_THINK, Score, score
from serverless.tasks import EASY, TASKS, Puzzle, Task, build_task

# Table 2 abbreviations, in the paper's order of single-task average performance.
ABBREV = {"cor": "correspondence", "ord": "ordering", "rot": "rotation", "sim": "similarity", "inp": "inpainting"}
PRESETS = {
    "ssl-r1": "correspondence,ordering,rotation,similarity,inpainting",
    "cor": "correspondence",
    "cor+ord": "correspondence,ordering",
    "cor+ord+rot": "correspondence,ordering,rotation",
    "cor+ord+rot+sim": "correspondence,ordering,rotation,similarity",
}


def parse_mixture(spec: str) -> dict[str, float]:
    """'ssl-r1' | 'rotation,ordering' | 'rot:2,ord:1' -> {task: weight}."""
    spec = PRESETS.get(spec, spec)
    weights: dict[str, float] = {}
    for item in filter(None, (x.strip() for x in spec.split(","))):
        name, _, weight = item.partition(":")
        name = ABBREV.get(name, name)
        if name not in TASKS:
            raise KeyError(f"unknown task {name!r}; choose from {sorted(TASKS)} or presets {sorted(PRESETS)}")
        weights[name] = weights.get(name, 0.0) + float(weight or 1.0)
    if not weights or any(w <= 0 for w in weights.values()):
        raise ValueError(f"invalid task mixture {spec!r}")
    return weights


def stratified_schedule(weights: dict[str, float], resolution: int = 100) -> list[str]:
    """Deterministic interleaved task order whose frequencies match the weights (equal weights = round robin)."""
    total = sum(weights.values())
    counts = {k: max(1, round(resolution * w / total)) for k, w in weights.items()}
    credit = {k: 0.0 for k in counts}
    order = []
    for _ in range(sum(counts.values())):
        for k in credit:
            credit[k] += counts[k]
        best = max(credit, key=lambda k: credit[k])
        credit[best] -= sum(counts.values())
        order.append(best)
    return order


def puzzle_messages(puzzle: Puzzle, format_prompt: str = FORMAT_PROMPT) -> list[dict[str, Any]]:
    """One user turn with text and images interleaved in prompt order; the format prompt is appended (Fig. 5)."""
    content: list[dict[str, Any]] = []
    for part in puzzle.parts:
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
        else:
            content.append({"type": "image_url", "image_url": {"url": to_data_uri(part)}})
    if format_prompt:
        content.append({"type": "text", "text": "\n" + format_prompt})
    return [{"role": "user", "content": content}]


@dataclass
class Objective:
    weights: dict[str, float]
    difficulty: str = "paper"
    task_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    seed: int = 0
    format_reward: float = FORMAT_REWARD

    def __post_init__(self) -> None:
        if self.difficulty not in ("paper", "easy"):
            raise ValueError(f"difficulty must be 'paper' or 'easy', got {self.difficulty!r}")
        self.tasks: dict[str, Task] = {}
        for name in self.weights:
            params = dict(EASY[name]) if self.difficulty == "easy" else {}
            params.update(self.task_params.get(name, {}))
            self.tasks[name] = build_task(name, **params)
        self.schedule = stratified_schedule(self.weights)

    @classmethod
    def from_spec(cls, spec: str, difficulty: str = "paper", task_params: str | dict | None = None, seed: int = 0,
                  format_reward: float = FORMAT_REWARD) -> Objective:
        params = json.loads(task_params) if isinstance(task_params, str) else (task_params or {})
        return cls(parse_mixture(spec), difficulty, params, seed, format_reward)

    def describe(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "difficulty": self.difficulty,
            "tasks": {k: {f: getattr(t, f) for f in t.__dataclass_fields__} for k, t in self.tasks.items()},
            "seed": self.seed,
            "format_reward": self.format_reward,
            "format_prompt": FORMAT_PROMPT,
            "format_requires_think": REQUIRE_THINK,
        }

    def puzzle(self, source: ImageSource, split: str, step: int, slot: int, slots_per_step: int) -> Puzzle:
        """Puzzle for a (split, step, slot) position; identical across restarts for the same seed."""
        position = step * slots_per_step + slot
        name = self.schedule[position % len(self.schedule)]
        rng = random.Random(f"{self.seed}:{split}:{step}:{slot}")
        puzzle = self.tasks[name].generate(rng, source)
        puzzle.meta["puzzle_id"] = f"{split}-{step}-{slot}"
        return puzzle

    def batch(self, source: ImageSource, split: str, step: int, size: int) -> list[Puzzle]:
        return [self.puzzle(source, split, step, i, size) for i in range(size)]

    def score(self, puzzle: Puzzle, response: str, reasoning: str | None = None) -> Score:
        """Score the final answer; `reasoning` is the native thinking channel when thinking mode is on."""
        return score(response, puzzle.answer, puzzle.kind, self.format_reward, reasoning=reasoning)
