"""Strict action boundaries and exact numeric answers."""

from dataclasses import dataclass, field
from fractions import Fraction
import re
from typing import Any


def number(value: str) -> Fraction:
    value = str(value).strip()
    if len(value) > 256:
        raise ValueError("Numeric answer is too long")
    atom = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?"
    if not re.fullmatch(rf"{atom}(?:/{atom})?%?", value):
        raise ValueError(f"Invalid numeric answer: {value!r}")
    percent = value.endswith("%")
    parts = value.rstrip("%").replace(",", "").split("/")
    result = Fraction(parts[0])
    if len(parts) == 2:
        result /= Fraction(parts[1])
    return result / 100 if percent else result


def numeric_equal(left: str, right: str) -> bool:
    try:
        return number(left) == number(right)
    except (ValueError, ZeroDivisionError):
        return False


@dataclass(frozen=True)
class Action:
    kind: str
    replacement: str | None = None

    def __post_init__(self):
        if self.kind not in {"KEEP", "REPLACE", "INVALID"}:
            raise ValueError("Unknown action kind")
        if (self.kind == "REPLACE") != (self.replacement is not None):
            raise ValueError("Only REPLACE actions carry a replacement")


def parse_action(text: str) -> Action:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return Action("INVALID")
    directives = [line for line in lines if line == "KEEP" or line.startswith("REPLACE")]
    if len(directives) != 1 or directives[0] != lines[-1]:
        return Action("INVALID")
    if lines[-1] == "KEEP":
        return Action("KEEP")
    match = re.fullmatch(r"REPLACE ####\s+(.+)", lines[-1])
    if match:
        try:
            number(match[1])
            return Action("REPLACE", match[1])
        except (ValueError, ZeroDivisionError):
            pass
    return Action("INVALID")


@dataclass(frozen=True)
class Problem:
    id: str
    text: str
    direct_solution: str
    direct_answer: str
    direct_tokens: int = 0
    checks: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.direct_tokens < 0:
            raise ValueError("Token counts must be nonnegative")
