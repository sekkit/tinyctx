"""Tests for the lightweight learned escalation classifier.

We verify:
  - Feature extraction picks up the right fields from a Responses-style body.
  - Training a tiny linearly-separable dataset yields a model that
    separates the classes.
  - JSON round-trip preserves predictions.
  - Default model from `model_path()` integrates with router.decide
    without errors when present.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from tinyctx.classifier import (
    FEATURE_ORDER, Features, Model, extract_features, train, load_jsonl,
)


def test_extract_features_basic_shape():
    body = {
        "instructions": "you are an assistant",
        "tools": [{"type": "apply_patch"}, {"type": "function"}],
        "input": [
            {"role": "user", "content": "hello world"},
            {"type": "function_call", "name": "do_thing"},
            {"role": "assistant", "content": [
                {"type": "output_text", "text": "```python\ndef f():\n    return 1\n```"},
                {"type": "input_image", "image_url": "..."},
            ]},
        ],
    }
    f = extract_features(body, est_tokens=12345, turn_count=4,
                         error_streak=1, is_compaction=False)
    assert f.has_apply_patch == 1.0
    assert f.has_image == 1.0
    assert f.tool_call_count == 1.0
    assert f.is_compaction == 0.0
    assert f.error_streak == 1.0
    assert f.turn_count == 4.0
    assert f.est_tokens == 12345.0
    assert math.isclose(f.log_est_tokens, math.log1p(12345.0), rel_tol=1e-6)
    # code_density > 0 because the assistant message has ```
    assert f.code_density > 0
    # FEATURE_ORDER vector is the right length
    assert len(f.to_vector()) == len(FEATURE_ORDER)


def test_train_separates_synthetic_data():
    # Construct a linearly-separable problem on est_tokens alone.
    X: list[list[float]] = []
    y: list[int] = []
    for i in range(50):
        f = Features(est_tokens=float(i), log_est_tokens=math.log1p(float(i)))
        X.append(f.to_vector())
        y.append(0)
    for i in range(60_000, 60_050):
        f = Features(est_tokens=float(i), log_est_tokens=math.log1p(float(i)))
        X.append(f.to_vector())
        y.append(1)
    m = train(X, y, epochs=200)
    assert m.n_train == 100

    small = Features(est_tokens=10, log_est_tokens=math.log1p(10)).to_vector()
    big = Features(est_tokens=70_000, log_est_tokens=math.log1p(70_000)).to_vector()
    p_small = m.predict_proba(small)
    p_big = m.predict_proba(big)
    assert p_small < 0.4, f"small should not escalate (p={p_small})"
    assert p_big > 0.6, f"big should escalate (p={p_big})"


def test_model_json_roundtrip():
    m = Model(weights=[0.1] * len(FEATURE_ORDER), bias=-1.0,
              feature_means=[0.0] * len(FEATURE_ORDER),
              feature_stds=[1.0] * len(FEATURE_ORDER), n_train=42)
    text = m.to_json()
    m2 = Model.from_json(text)
    x = [1.0] * len(FEATURE_ORDER)
    assert math.isclose(m.predict_proba(x), m2.predict_proba(x), abs_tol=1e-9)
    assert m2.n_train == 42


def test_load_jsonl_roundtrip():
    with TemporaryDirectory() as td:
        p = Path(td) / "labels.jsonl"
        rows = [
            {"features": {"est_tokens": 100}, "label": 0},
            {"features": {"est_tokens": 80000}, "label": 1},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        X, y = load_jsonl(p)
        assert len(X) == 2
        assert y == [0, 1]
        # Other features default to 0
        assert X[0][FEATURE_ORDER.index("est_tokens")] == 100.0


def test_router_uses_classifier_when_present():
    """When a model exists at the standard path, router.decide queries it.
    We don't assert a specific outcome — only that the path is exercised
    without raising and the decision has a sensible reason."""
    from tinyctx import router as router_mod
    from tinyctx.classifier import model_path
    # build and save a model that always votes 'escalate'
    m = Model(weights=[10.0] * len(FEATURE_ORDER), bias=10.0,
              feature_means=[0.0] * len(FEATURE_ORDER),
              feature_stds=[1.0] * len(FEATURE_ORDER), n_train=1)
    mp = model_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    backup = mp.read_text() if mp.exists() else None
    try:
        mp.write_text(m.to_json())
        # Reset the cached classifier so router re-loads.
        router_mod._CLASSIFIER = None
        router_mod._CLASSIFIER_LOADED = False
        from tinyctx.config import Config
        cfg = Config()
        body = {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "tiny request"}],
        }
        d = router_mod.decide(body, cfg)
        # heuristic alone would say local; classifier overrides to frontier
        assert d.route == "frontier"
        assert "classifier" in d.reason
    finally:
        if backup is None:
            mp.unlink()
        else:
            mp.write_text(backup)
        router_mod._CLASSIFIER = None
        router_mod._CLASSIFIER_LOADED = False


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
