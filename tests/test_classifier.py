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
    _sigmoid, _fit_standardization, model_path, main,
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


# ---------------------------------------------------------------------------
# Additional coverage: feature-extraction edge cases, score boundaries,
# fallback paths, CLI behaviour, and Config integration.
# ---------------------------------------------------------------------------


def test_extract_features_empty_body_defaults():
    """An empty dict should produce a Features instance whose every field is
    zero (or 0.0 in the case of code_density, which is computed)."""
    f = extract_features({})
    assert f.est_tokens == 0.0
    assert f.log_est_tokens == 0.0
    assert f.turn_count == 0.0
    assert f.error_streak == 0.0
    assert f.is_compaction == 0.0
    assert f.tool_call_count == 0.0
    assert f.max_message_chars == 0.0
    assert f.code_density == 0.0
    assert f.has_apply_patch == 0.0
    assert f.has_image == 0.0
    assert f.instructions_chars == 0.0
    assert f.to_vector() == [0.0] * len(FEATURE_ORDER)


def test_extract_features_handles_non_string_instructions_and_non_list_tools():
    """Non-string instructions and non-list tools should be silently ignored
    rather than raising — extract_features must be defensive."""
    body = {"instructions": {"unexpected": "dict"}, "tools": "not a list"}
    f = extract_features(body)
    assert f.instructions_chars == 0.0
    assert f.has_apply_patch == 0.0


def test_extract_features_skips_non_dict_items_and_finds_image_url():
    """Items that aren't dicts must be skipped; image_url-typed parts must
    still set has_image."""
    body = {
        "input": [
            "stray-string-not-a-dict",
            42,
            {"role": "user", "content": [
                {"type": "image_url", "image_url": "http://x"},
                {"type": "output_text", "text": "plain text"},
            ]},
        ],
    }
    f = extract_features(body)
    assert f.has_image == 1.0
    assert f.max_message_chars == float(len("plain text"))


def test_extract_features_code_density_heuristic_picks_up_def_and_semicolon():
    """code_density should fire on `    def ` or `;` even without triple
    backticks — and equal 1.0 when every char is in a code-tagged message."""
    body = {
        "input": [
            {"role": "user", "content": "    def helper():\n        return 1"},
        ],
    }
    f = extract_features(body)
    assert f.code_density == 1.0


def test_extract_features_is_compaction_flag_propagates():
    f = extract_features({}, is_compaction=True)
    assert f.is_compaction == 1.0


def test_sigmoid_boundary_values():
    """_sigmoid should be 0.5 at z=0 and asymptote to 1.0 / 0.0 for large
    magnitudes — both branches of the implementation are exercised."""
    assert math.isclose(_sigmoid(0.0), 0.5, abs_tol=1e-12)
    assert _sigmoid(50.0) > 0.999_999
    assert _sigmoid(-50.0) < 1e-6
    # Branch coverage: positive z and negative z each must produce a value
    # in (0, 1).
    assert 0.0 < _sigmoid(5.0) < 1.0
    assert 0.0 < _sigmoid(-5.0) < 1.0


def test_model_predict_threshold_boundary_exact_and_extremes():
    """Predict must return 1 iff probability >= threshold — boundary
    inclusive on the high side, exclusive on the low side."""
    # Construct a model whose predict_proba(0-vector) is exactly 0.5.
    m = Model(weights=[0.0] * len(FEATURE_ORDER), bias=0.0,
              feature_means=[0.0] * len(FEATURE_ORDER),
              feature_stds=[1.0] * len(FEATURE_ORDER))
    x = [0.0] * len(FEATURE_ORDER)
    assert math.isclose(m.predict_proba(x), 0.5, abs_tol=1e-12)
    # threshold == probability => predict 1 (>=)
    assert m.predict(x, threshold=0.5) == 1
    # threshold just above => predict 0
    assert m.predict(x, threshold=0.5 + 1e-9) == 0
    # threshold == 0.0 always 1, threshold == 1.0 only if prob>=1.0
    assert m.predict(x, threshold=0.0) == 1
    assert m.predict(x, threshold=1.0) == 0


def test_model_standardize_handles_zero_std():
    """A zero feature_std should be replaced by 1.0 to avoid divide-by-zero."""
    m = Model(weights=[1.0] * len(FEATURE_ORDER), bias=0.0,
              feature_means=[0.0] * len(FEATURE_ORDER),
              feature_stds=[0.0] * len(FEATURE_ORDER))
    z = m.standardize([7.0] * len(FEATURE_ORDER))
    # zero std -> divisor becomes 1.0, value passes through unchanged.
    assert z == [7.0] * len(FEATURE_ORDER)


def test_train_empty_inputs_returns_default_model():
    m = train([], [])
    assert isinstance(m, Model)
    assert m.n_train == 0
    assert m.weights == [0.0] * len(FEATURE_ORDER)
    assert m.bias == 0.0


def test_fit_standardization_constant_column_uses_unit_std():
    """When every value in a column is identical, variance is zero — the
    helper must coerce std to 1.0 so callers don't divide by zero."""
    X = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]
    means, stds = _fit_standardization(X)
    assert means[0] == 5.0
    assert stds[0] == 1.0
    assert math.isclose(means[1], 2.0, abs_tol=1e-12)
    assert stds[1] > 0


def test_load_jsonl_skips_malformed_and_missing_features():
    """Malformed JSON, blank lines and rows without a dict `features` field
    must be silently dropped without raising."""
    with TemporaryDirectory() as td:
        p = Path(td) / "labels.jsonl"
        p.write_text(
            "\n"
            "not valid json at all\n"
            + json.dumps({"features": "not-a-dict", "label": 1}) + "\n"
            + json.dumps({"features": {"est_tokens": 5}, "label": 0}) + "\n"
            + json.dumps({"features": {"est_tokens": 9}}) + "\n"  # missing label -> 0
        )
        X, y = load_jsonl(p)
        assert len(X) == 2
        assert y == [0, 0]
        assert X[0][FEATURE_ORDER.index("est_tokens")] == 5.0
        assert X[1][FEATURE_ORDER.index("est_tokens")] == 9.0


def test_model_path_lives_under_home_tinyctx():
    """The default model path must live in ~/.tinyctx and be a Path object."""
    p = model_path()
    assert isinstance(p, Path)
    assert p.name == "classifier.json"
    assert p.parent.name == ".tinyctx"
    assert p.parent.parent == Path.home()


def test_router_falls_back_to_heuristic_when_classifier_file_missing():
    """If no model file exists at the expected path, router.decide must
    silently fall back to the heuristic and never raise."""
    from tinyctx import router as router_mod
    from tinyctx.classifier import model_path
    mp = model_path()
    backup = mp.read_text() if mp.exists() else None
    try:
        if mp.exists():
            mp.unlink()
        router_mod._CLASSIFIER = None
        router_mod._CLASSIFIER_LOADED = False
        from tinyctx.config import Config
        cfg = Config()
        body = {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "hi"}],
        }
        d = router_mod.decide(body, cfg)
        # No model -> heuristic-only path; small request stays local.
        assert d.route == "local"
    finally:
        if backup is not None:
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(backup)
        router_mod._CLASSIFIER = None
        router_mod._CLASSIFIER_LOADED = False


def test_config_self_classify_threshold_default_is_07():
    """The Config default for self_classify_threshold must be 0.7 — proxy.py
    reads this to gate self-classification escalation."""
    from tinyctx.config import Config
    cfg = Config()
    assert isinstance(cfg.self_classify_threshold, float)
    assert math.isclose(cfg.self_classify_threshold, 0.7, abs_tol=1e-9)


def test_main_cli_train_then_predict_roundtrip(capsys):
    """The CLI `train` then `predict` commands should round-trip without
    error and emit a probability in [0, 1]."""
    with TemporaryDirectory() as td:
        labels = Path(td) / "labels.jsonl"
        labels.write_text(
            json.dumps({"features": {"est_tokens": 10}, "label": 0}) + "\n"
            + json.dumps({"features": {"est_tokens": 80_000}, "label": 1}) + "\n"
        )
        out = Path(td) / "model.json"
        rc = main(["train", str(labels), "--output", str(out), "--epochs", "5"])
        assert rc == 0
        assert out.is_file()
        # Verify the persisted JSON is a valid Model.
        m = Model.from_json(out.read_text())
        assert m.n_train == 2

        rc = main(["predict", "est_tokens=50000", "--model", str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "p(escalate)=" in captured.out
        assert "pred=" in captured.out


def test_main_cli_predict_handles_non_numeric_value(capsys):
    """When a predict feature value isn't numeric, the CLI must coerce to
    0.0 rather than raise."""
    with TemporaryDirectory() as td:
        m = Model(weights=[0.0] * len(FEATURE_ORDER), bias=0.0,
                  feature_means=[0.0] * len(FEATURE_ORDER),
                  feature_stds=[1.0] * len(FEATURE_ORDER), n_train=1)
        out = Path(td) / "model.json"
        out.write_text(m.to_json())
        rc = main(["predict", "est_tokens=not-a-number", "--model", str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        # bias=0, weights=0, x=0 => prob=0.5 -> pred=frontier (>=0.5)
        assert "p(escalate)=0.5000" in captured.out
        assert "pred=frontier" in captured.out


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
