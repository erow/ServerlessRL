"""Weave tracing for rollouts.

Each rollout is one Weave call ("rollout_train" / "rollout_val") whose inputs carry the puzzle identity, prompt
text (with <image> placeholders), the puzzle images in prompt order, and the ground truth; its output carries the
response and reward breakdown. ART's managed OpenAI client is not auto-patched by Weave, so the images are logged
here rather than through a child chat-completion call.
"""

from __future__ import annotations

from typing import Any, Callable


def init(entity: str | None, project: str) -> None:
    import weave

    weave.init(f"{entity}/{project}" if entity else project)


def sent_prompt(messages: list[dict[str, Any]]) -> str:
    """The user turn exactly as sent (task prompt plus appended format prompt), images shown as <image>."""
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content or []:
            parts.append(part["text"] if part.get("type") == "text" else "<image>")
    return "".join(parts)


def _inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    puzzle = inputs["puzzle"]
    return {
        "split": inputs.get("split"),
        "task": puzzle.task,
        "puzzle_id": puzzle.meta["puzzle_id"],
        "answer": puzzle.answer,
        "prompt": sent_prompt(inputs.get("messages") or []),
        "images": puzzle.images,
        "temperature": inputs.get("temperature"),
        "puzzle_meta": {k: v for k, v in puzzle.meta.items() if k != "puzzle_id"},
    }


def _output(traj: Any) -> dict[str, Any]:
    if traj is None:  # the rollout raised; weave still post-processes the (missing) output
        return {}
    return {"reasoning": traj.metadata.get("reasoning"), "response": traj.metadata.get("response"),
            "reward": traj.reward, "policy_step": traj.metadata.get("policy_step"), **traj.metrics}


def _display_name(call: Any) -> str:
    inputs = call.inputs if isinstance(call.inputs, dict) else {}
    puzzle = inputs.get("puzzle")
    if puzzle is not None and hasattr(puzzle, "task"):
        return f"{puzzle.task} {puzzle.meta['puzzle_id']}"
    if "task" in inputs:  # already post-processed
        return f"{inputs['task']} {inputs.get('puzzle_id', '')}".strip()
    return "rollout"


def traced(fn: Callable, split: str, sample_rate: float = 1.0) -> Callable:
    """Wrap an async rollout(puzzle, messages, temperature, sink, split=...) as a Weave op."""
    import weave

    return weave.op(
        fn,
        name=f"rollout_{split}",
        call_display_name=_display_name,
        postprocess_inputs=_inputs,
        postprocess_output=_output,
        tracing_sample_rate=sample_rate,
    )
