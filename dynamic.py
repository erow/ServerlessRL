"""Dynamic difficulty: reweight the task mixture so every task contributes the same expected accuracy mass.

Task t is drawn with probability p_t proportional to w_t / acc_t, where w_t is the static mixture weight and acc_t a
running (EMA) estimate of its training accuracy, so p_t * acc_t / w_t is equal across tasks. With equal static
weights, accuracies 0.9 and 0.3 give p = 1/4 and 3/4. Accuracy is floored at `min_accuracy` so an unsolved task
cannot take the whole batch.

The state (accuracy estimates) is appended to a JSONL file after every update; on resume the row for the restart
step is reloaded, so the per-step task allocation is reproducible given the logged state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DynamicDifficulty:
    base_weights: dict[str, float]
    momentum: float = 0.8  # EMA weight on the previous estimate
    min_accuracy: float = 0.05
    accuracy: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if not 0.0 < self.min_accuracy <= 1.0:
            raise ValueError(f"min_accuracy must be in (0, 1], got {self.min_accuracy}")
        # Unseen tasks start at a common prior, which leaves the static mixture unchanged until data arrives.
        self.accuracy = {t: float(self.accuracy.get(t, 0.5)) for t in self.base_weights}

    @classmethod
    def from_summary(cls, base_weights: dict[str, float], path: Path, **kwargs: Any) -> DynamicDifficulty:
        """Initialise estimates from a `serverless.difficulty` summary.json (its per-task "accuracy")."""
        summary = json.loads(Path(path).read_text())
        return cls(base_weights, accuracy={t: summary[t]["accuracy"] for t in base_weights if t in summary}, **kwargs)

    def weights(self) -> dict[str, float]:
        """Normalised sampling probabilities p_t ∝ w_t / max(acc_t, min_accuracy)."""
        raw = {t: w / max(self.accuracy[t], self.min_accuracy) for t, w in self.base_weights.items()}
        total = sum(raw.values())
        return {t: v / total for t, v in raw.items()}

    def update(self, metrics: dict[str, float]) -> None:
        """EMA update from a step summary (`<task>/accuracy` keys); tasks absent from the step keep their estimate."""
        for t in self.accuracy:
            if (acc := metrics.get(f"{t}/accuracy")) is not None:
                self.accuracy[t] = self.momentum * self.accuracy[t] + (1.0 - self.momentum) * acc

    def state(self) -> dict[str, Any]:
        return {"accuracy": dict(self.accuracy), "weights": self.weights()}

    def save(self, path: Path, step: int) -> None:
        with Path(path).open("a") as f:
            f.write(json.dumps({"step": step, **self.state()}) + "\n")

    def load(self, path: Path, step: int) -> bool:
        """Restore the estimate logged for `step` (the latest row at or before it); False if none exists."""
        if not Path(path).exists():
            return False
        rows = [r for r in map(json.loads, Path(path).read_text().splitlines()) if r["step"] <= step]
        if not rows:
            return False
        self.accuracy.update({t: a for t, a in rows[-1]["accuracy"].items() if t in self.accuracy})
        return True


def systematic_allocation(weights: dict[str, float], size: int, offset: float) -> list[str]:
    """Assign `size` slots to tasks by systematic sampling: unbiased for `weights`, with per-step counts within one
    of size * p_t. `offset` in [0, 1) is the single random draw (seeded per step by the caller)."""
    names = list(weights)
    total = sum(weights.values())
    out, cum, j = [], weights[names[0]] / total, 0
    for i in range(size):
        u = (offset + i) / size
        while u >= cum and j < len(names) - 1:
            j += 1
            cum += weights[names[j]] / total
        out.append(names[j])
    return out
