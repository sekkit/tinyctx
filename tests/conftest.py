"""Test-suite conftest: makes `tests/` itself importable so saga tests can
share helpers from `tests/helpers/integration.py` without a brittle
`tests.helpers.X` package import (which would require `tests/__init__.py`
and changes pytest's collection root).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


@pytest.fixture(autouse=True)
def _reset_adaptive_model_state():
    from tinyctx import adaptive_model

    adaptive_model.reset_state()
    yield
    adaptive_model.reset_state()
