"""Tests for strutil module."""

from src.strutil import reverse, count_words, to_title


def test_reverse():
    assert reverse("abc") == "cba"


def test_count_words():
    assert count_words("hello world") == 2
    assert count_words("") == 0


def test_to_title():
    assert to_title("hello world") == "Hello World"
