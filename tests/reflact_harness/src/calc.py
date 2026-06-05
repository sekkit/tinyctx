"""Simple calculator module — intentionally has some rough edges for testing."""

from typing import Optional


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference of two numbers."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


def divide(a: float, b: float) -> Optional[float]:
    """Return a divided by b. Returns None if b is zero."""
    if b == 0:
        return None
    return a / b
