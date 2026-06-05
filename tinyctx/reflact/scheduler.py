"""Edit-budget schedulers for ReflACT (learning-rate analogues).

Controls the max number of edits per step over training.
Early steps: larger budget (more to fix).  Later steps: smaller budget (fine-tuning).

Supported: constant, linear, cosine.
"""

from __future__ import annotations

import math


class LRScheduler:
    """Base class for edit-budget schedulers."""

    def __init__(self, max_lr: int, min_lr: int, total_steps: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = max(total_steps, 1)
        self._step = 0

    def step(self) -> int:
        self._step += 1
        return max(self.min_lr, self._compute(self._step))

    def _compute(self, step: int) -> int:
        raise NotImplementedError


class ConstantScheduler(LRScheduler):
    def _compute(self, step: int) -> int:
        return self.max_lr


class LinearScheduler(LRScheduler):
    def _compute(self, step: int) -> int:
        progress = min(1.0, (step - 1) / max(self.total_steps - 1, 1))
        return round(self.max_lr - (self.max_lr - self.min_lr) * progress)


class CosineScheduler(LRScheduler):
    def _compute(self, step: int) -> int:
        progress = min(1.0, (step - 1) / max(self.total_steps - 1, 1))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return round(self.min_lr + (self.max_lr - self.min_lr) * coeff)


def build_scheduler(
    mode: str = "cosine",
    max_lr: int = 10,
    min_lr: int = 2,
    total_steps: int = 20,
) -> LRScheduler:
    mode = mode.strip().lower()
    if mode == "constant":
        return ConstantScheduler(max_lr, min_lr, total_steps)
    if mode == "linear":
        return LinearScheduler(max_lr, min_lr, total_steps)
    if mode == "cosine":
        return CosineScheduler(max_lr, min_lr, total_steps)
    raise ValueError(f"unknown scheduler mode: {mode!r}")
