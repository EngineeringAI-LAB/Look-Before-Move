"""Standalone numeric helpers for Plan-A local engines."""

from __future__ import annotations

from statistics import median
from typing import Iterable


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp one float into the given inclusive range."""

    return max(lower, min(upper, float(value)))


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    """Return one arithmetic mean or the provided default."""

    numbers = [float(value) for value in values]
    if not numbers:
        return float(default)
    return sum(numbers) / len(numbers)


def safe_median(values: Iterable[float], default: float = 0.0) -> float:
    """Return one median or the provided default."""

    numbers = [float(value) for value in values]
    if not numbers:
        return float(default)
    return float(median(numbers))


def trimmed_mean(values: Iterable[float], trim_ratio: float = 0.2, default: float = 0.0) -> float:
    """Return one trimmed mean for a small candidate pool."""

    numbers = sorted(float(value) for value in values)
    if not numbers:
        return float(default)
    if len(numbers) <= 2:
        return safe_mean(numbers, default=default)
    trim_count = int(len(numbers) * trim_ratio)
    if trim_count <= 0:
        return safe_mean(numbers, default=default)
    if trim_count * 2 >= len(numbers):
        return safe_mean(numbers, default=default)
    return safe_mean(numbers[trim_count:-trim_count], default=default)


__all__ = [
    "clamp",
    "safe_mean",
    "safe_median",
    "trimmed_mean",
]


