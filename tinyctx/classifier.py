"""Lightweight learned escalation classifier.

A FrugalGPT-style scorer (Stanford 2305.05176): given features of a Codex
request, predict whether the local-model draft is likely to satisfy the
user (and therefore stay local) vs needs escalation to the frontier model.

Implementation is pure-Python logistic regression trained with SGD. We
deliberately avoid sklearn/numpy as deps — the feature vector is tiny
(<= 12 floats) and SGD on a few thousand examples runs in <1 sec.

Training data shape (one event per line, JSONL):

    {"features": {...float fields...},
     "label": 1 (escalate to frontier) | 0 (local was fine)}

Bootstrap with `tinyctx-stats` logs by treating `decision == "frontier"`
as the supervision label, then refine over time as you observe which
local-routed requests had to be retried at the frontier (error_streak,
user reissues).

CLI:
    python -m tinyctx.classifier train ~/.tinyctx/labeled.jsonl
    python -m tinyctx.classifier predict-feature key=value ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Stable feature order. Adding new features means a model retrain.
FEATURE_ORDER = (
    "est_tokens",
    "log_est_tokens",
    "turn_count",
    "error_streak",
    "is_compaction",
    "tool_call_count",
    "max_message_chars",
    "code_density",
    "has_apply_patch",
    "has_image",
    "instructions_chars",
)


@dataclass
class Features:
    est_tokens: float = 0.0
    log_est_tokens: float = 0.0
    turn_count: float = 0.0
    error_streak: float = 0.0
    is_compaction: float = 0.0
    tool_call_count: float = 0.0
    max_message_chars: float = 0.0
    code_density: float = 0.0
    has_apply_patch: float = 0.0
    has_image: float = 0.0
    instructions_chars: float = 0.0

    def to_vector(self) -> list[float]:
        return [getattr(self, k) for k in FEATURE_ORDER]


def extract_features(body: dict[str, Any], *,
                     est_tokens: int | None = None,
                     turn_count: int | None = None,
                     error_streak: int = 0,
                     is_compaction: bool = False) -> Features:
    f = Features()
    f.est_tokens = float(est_tokens or 0)
    f.log_est_tokens = math.log1p(f.est_tokens)
    f.turn_count = float(turn_count or 0)
    f.error_streak = float(error_streak)
    f.is_compaction = 1.0 if is_compaction else 0.0

    instr = body.get("instructions") or ""
    if isinstance(instr, str):
        f.instructions_chars = float(len(instr))

    tools = body.get("tools") or []
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "apply_patch":
                f.has_apply_patch = 1.0

    items = body.get("input") or body.get("messages") or []
    if isinstance(items, list):
        max_chars = 0
        tool_calls = 0
        code_chars = 0
        total_chars = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("type") in ("function_call", "tool_use"):
                tool_calls += 1
            content = it.get("content")
            if isinstance(content, str):
                max_chars = max(max_chars, len(content))
                total_chars += len(content)
                if "```" in content or "    def " in content or ";" in content:
                    code_chars += len(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("type")
                        if t == "input_image" or t == "image_url":
                            f.has_image = 1.0
                        text = c.get("text", "")
                        if isinstance(text, str):
                            max_chars = max(max_chars, len(text))
                            total_chars += len(text)
                            if "```" in text:
                                code_chars += len(text)
        f.tool_call_count = float(tool_calls)
        f.max_message_chars = float(max_chars)
        f.code_density = float(code_chars) / max(1.0, float(total_chars))
    return f


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class Model:
    weights: list[float] = field(default_factory=lambda: [0.0] * len(FEATURE_ORDER))
    bias: float = 0.0
    feature_means: list[float] = field(default_factory=lambda: [0.0] * len(FEATURE_ORDER))
    feature_stds: list[float] = field(default_factory=lambda: [1.0] * len(FEATURE_ORDER))
    feature_order: tuple[str, ...] = FEATURE_ORDER
    n_train: int = 0

    def standardize(self, x: list[float]) -> list[float]:
        return [(xi - mu) / (sd or 1.0) for xi, mu, sd in
                zip(x, self.feature_means, self.feature_stds)]

    def predict_proba(self, x: list[float]) -> float:
        z = sum(w * xi for w, xi in zip(self.weights, self.standardize(x))) + self.bias
        return _sigmoid(z)

    def predict(self, x: list[float], *, threshold: float = 0.5) -> int:
        return 1 if self.predict_proba(x) >= threshold else 0

    def to_json(self) -> str:
        d = asdict(self)
        d["feature_order"] = list(self.feature_order)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Model":
        d = json.loads(text)
        d["feature_order"] = tuple(d.get("feature_order", FEATURE_ORDER))
        return cls(**d)


def _fit_standardization(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(X)
    if n == 0:
        return [0.0] * len(FEATURE_ORDER), [1.0] * len(FEATURE_ORDER)
    k = len(X[0])
    means = [sum(row[i] for row in X) / n for i in range(k)]
    stds: list[float] = []
    for i in range(k):
        var = sum((row[i] - means[i]) ** 2 for row in X) / n
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def train(X: list[list[float]], y: list[int],
          *, lr: float = 0.05, epochs: int = 80, l2: float = 0.001,
          seed: int = 7) -> Model:
    """Train logistic regression with mini-batch SGD."""
    if not X:
        return Model()
    means, stds = _fit_standardization(X)
    n = len(X)
    k = len(X[0])
    rnd = random.Random(seed)

    Xz = [[(row[i] - means[i]) / (stds[i] or 1.0) for i in range(k)] for row in X]

    weights = [0.0] * k
    bias = 0.0
    indices = list(range(n))
    for _ in range(epochs):
        rnd.shuffle(indices)
        for idx in indices:
            xi = Xz[idx]
            yi = y[idx]
            z = sum(w * x for w, x in zip(weights, xi)) + bias
            p = _sigmoid(z)
            err = p - yi
            for j in range(k):
                grad = err * xi[j] + l2 * weights[j]
                weights[j] -= lr * grad
            bias -= lr * err
    return Model(weights=weights, bias=bias,
                 feature_means=means, feature_stds=stds, n_train=n)


def load_jsonl(path: Path) -> tuple[list[list[float]], list[int]]:
    X: list[list[float]] = []
    y: list[int] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        feats = row.get("features") or {}
        if not isinstance(feats, dict):
            continue
        vec = [float(feats.get(k, 0.0)) for k in FEATURE_ORDER]
        X.append(vec)
        y.append(int(row.get("label", 0)))
    return X, y


def model_path() -> Path:
    return Path.home() / ".tinyctx" / "classifier.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.classifier")
    sub = p.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train", help="train from a labeled JSONL")
    pt.add_argument("input")
    pt.add_argument("--output", default=str(model_path()))
    pt.add_argument("--epochs", type=int, default=80)
    pp = sub.add_parser("predict", help="predict from key=value features")
    pp.add_argument("kvs", nargs="+")
    pp.add_argument("--model", default=str(model_path()))
    args = p.parse_args(argv)

    if args.cmd == "train":
        X, y = load_jsonl(Path(args.input))
        m = train(X, y, epochs=args.epochs)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(m.to_json())
        print(f"trained on {m.n_train} rows -> {out}")
        return 0

    if args.cmd == "predict":
        m = Model.from_json(Path(args.model).read_text())
        feats: dict[str, float] = {}
        for kv in args.kvs:
            k, _, v = kv.partition("=")
            try:
                feats[k] = float(v)
            except ValueError:
                feats[k] = 0.0
        vec = [feats.get(k, 0.0) for k in FEATURE_ORDER]
        prob = m.predict_proba(vec)
        print(f"p(escalate)={prob:.4f}  pred={'frontier' if prob >= 0.5 else 'local'}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
