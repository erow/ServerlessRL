"""SSL-R1 reward (Sec. 3.2.2): accuracy in [0, 1] plus a 0.2 format reward.

The format check follows the format prompt actually sent: with the paper's <think> instruction it requires exactly
one <think>...</think> before exactly one \\boxed{}; without it (default for Gemma 4, which reasons without tags) it
requires reasoning text followed by exactly one \\boxed{}. Parsing is strict: only the final \\boxed{} is read, and
answers must have the exact shape the prompt asks for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from serverless.config import FORMAT_PROMPT, FORMAT_REWARD
from serverless.tasks.base import AnswerKind

_INT = re.compile(r"^\s*(\d+)\s*(?:°|\^\\circ|\^\{\\circ\}|degrees?)?\s*$")
_LETTER = re.compile(r"^\s*\(?\s*([A-H])\s*\)?\s*$")
_SEQ_ITEM = re.compile(r"^\s*(\d+)\s*$")


def boxed_contents(text: str) -> list[str]:
    """Contents of every \\boxed{...}, with balanced inner braces."""
    out = []
    start = 0
    while (pos := text.find("\\boxed{", start)) != -1:
        depth, i = 1, pos + len("\\boxed{")
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth:
            break
        out.append(text[pos + len("\\boxed{") : i - 1])
        start = i
    return out


REQUIRE_THINK = "<think>" in FORMAT_PROMPT


def format_ok(text: str, require_think: bool = REQUIRE_THINK, reasoning: str | None = None) -> bool:
    """Exactly one \\boxed{} preceded by reasoning.

    `reasoning` is the separate thinking channel (native thinking mode): it must be non-empty. Otherwise the reasoning
    must precede the box in `text`, and with `require_think` it must be one <think>...</think>.
    """
    if text.count("\\boxed{") != 1 or len(boxed_contents(text)) != 1:
        return False
    if reasoning is not None:
        return bool(reasoning.strip())
    box_at = text.find("\\boxed{")
    if not require_think:
        return bool(text[:box_at].strip())
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False
    open_at, close_at = text.find("<think>"), text.find("</think>")
    return open_at < close_at < box_at


def parse_answer(content: str, kind: AnswerKind) -> str | list[int] | None:
    if kind == "choice":
        m = _LETTER.match(content)
        return m.group(1) if m else None
    if kind == "integer":
        m = _INT.match(content)
        return str(int(m.group(1))) if m else None
    body = content.strip().removeprefix("[").removesuffix("]")
    items = [_SEQ_ITEM.match(x) for x in body.split(",")]
    return [int(m.group(1)) for m in items] if items and all(items) else None


def accuracy(pred: str | list[int] | None, answer: str, kind: AnswerKind) -> float:
    if pred is None:
        return 0.0
    if kind != "sequence":
        return float(pred == answer)
    target = [int(x) for x in answer.split(",")]
    if not isinstance(pred, list) or len(pred) != len(target):
        return 0.0
    return sum(p == t for p, t in zip(pred, target)) / len(target)


@dataclass(frozen=True)
class Score:
    reward: float
    accuracy: float
    format: bool
    parsed: bool


def score(
    response: str, answer: str, kind: AnswerKind, format_reward: float = FORMAT_REWARD,
    require_think: bool = REQUIRE_THINK, reasoning: str | None = None,
) -> Score:
    boxes = boxed_contents(response)
    pred = parse_answer(boxes[-1], kind) if boxes else None
    acc = accuracy(pred, answer, kind)
    fmt = format_ok(response, require_think, reasoning)
    return Score(acc + (format_reward if fmt else 0.0), acc, fmt, pred is not None)
