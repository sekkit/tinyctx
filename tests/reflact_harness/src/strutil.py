"""String utilities — intentionally simple with gaps to test agent behavior."""

from typing import Optional


def reverse(text: str) -> str:
    """Return the reversed string."""
    return text[::-1]


def count_words(text: str) -> int:
    """Return the number of words in text."""
    if not text.strip():
        return 0
    return len(text.split())


def to_title(text: str) -> str:
    """Convert to title case."""
    return text.title()
