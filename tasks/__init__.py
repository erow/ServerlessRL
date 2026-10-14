"""SSL-R1 task registry. Each task turns a COCO image into a verifiable puzzle."""

from __future__ import annotations

from typing import Any

from serverless.tasks.base import LETTERS, Puzzle, Task
from serverless.tasks.correspondence import CorrespondenceTask
from serverless.tasks.inpainting import InpaintingTask
from serverless.tasks.ordering import OrderingTask
from serverless.tasks.rotation import RotationTask
from serverless.tasks.similarity import SimilarityTask

TASKS: dict[str, type] = {
    "rotation": RotationTask,
    "similarity": SimilarityTask,
    "inpainting": InpaintingTask,
    "ordering": OrderingTask,
    "correspondence": CorrespondenceTask,
}

# Paper "easier" settings (Sec. 4.3 ablations).
EASY: dict[str, dict[str, Any]] = {
    "rotation": {"steps": 4},
    "similarity": {"weak": True},
    "inpainting": {"ratio": 0.25},
    "ordering": {"grid": 2},
    "correspondence": {"points": 4},
}
HARD_ROTATION = {"steps": 8}


def build_task(name: str, **params: Any) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name](**params)


__all__ = ["EASY", "HARD_ROTATION", "LETTERS", "Puzzle", "TASKS", "Task", "build_task"]
