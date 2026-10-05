"""CPU-only logging tests; no model prediction or design execution."""
import json
import numpy as np
import pytest
from hallucination_lab import Run, Stage
from hallucination_lab.engine import StepResult
from hallucination_lab.backends.af2 import AF2
from hallucination_lab.checkpoint import load, save


def backend():
    model = object.__new__(AF2)
    model.design_length = 1
    model.positions = np.array([1])
    model.sequence = "NA"
    model._sequence_bias = np.zeros((2, 20))
    model._sequence_bias[1, 1] = 1.5
    model.metadata = {"test": "identity-only"}
    return model


def make_run(output, model=None):
    values = np.zeros((1, 20), np.float32)
    values[0, 0] = 1

    def step(state, unused_backend, controls, key):
        # Effective controls differ from requested ones; first step stays best.
        return StepResult(state.logits.copy(), float(state.step + 1), {},
                          state.logits.copy(), effective_controls={"alpha": state.step + 1.})

    return Run(backend() if model is None else model, values,
               [Stage("test", 2, controls={"alpha": 9.})],
               problem_id="identity-log", step_fn=step, output=output)


def summary(path):
    return json.loads((path / "summary.json").read_text())


def test_bias_fixed_positions_effective_controls_and_resume(tmp_path):
    make_run(tmp_path / "full").run()
    make_run(tmp_path / "resume").run(max_steps=1)
    resumed = make_run(tmp_path / "resume").restore(tmp_path / "resume/checkpoint.npz")
    assert resumed.state.best_controls == {"alpha": 1.}
    resumed.run()
    rows = [json.loads(s) for s in (tmp_path / "resume/trajectory.jsonl").read_text().splitlines()]
    assert [r["evaluated_sequence"] for r in rows] == ["NR", "NA"]
    assert [r["evaluated_raw_design_sequence"] for r in rows] == ["A", "A"]
    assert rows[0]["requested_controls"]["alpha"] == 9.
    assert rows[0]["evaluated_sequence_controls"] == {"alpha": 1.}
    assert rows[0]["evaluated_sequence_semantics"] == "backend_hard_identity"
    result = summary(tmp_path / "resume")
    assert result["best_evaluated_sequence"] == "NR"
    assert result["best_evaluated_sequence_controls"] == {"alpha": 1.}
    assert result["next_sequence"] == "NA"
    assert result["next_sequence_controls"] == {"alpha": 2.}
    assert result == summary(tmp_path / "full")


def test_legacy_checkpoint_does_not_invent_controls(tmp_path):
    make_run(tmp_path).run(max_steps=1)
    path = tmp_path / "checkpoint.npz"
    old = load(path)
    old["state"].pop("last_controls")
    old["state"].pop("best_controls")
    save(path, old)
    resumed = make_run(tmp_path).restore(path)
    resumed.run(max_steps=0)
    result = summary(tmp_path)
    for prefix in ("next", "best_evaluated"):
        assert result[f"{prefix}_sequence"] is None
        assert result[f"{prefix}_raw_design_sequence"] == "A"
        assert result[f"{prefix}_sequence_semantics"] == "unknown_effective_controls"
        assert result[f"{prefix}_sequence_controls"] is None
    resumed.run()
    result = summary(tmp_path)
    assert result["next_sequence"] == "NA"
    assert result["best_evaluated_sequence"] is None


def test_backend_without_decoder_is_explicit_raw_fallback(tmp_path):
    from types import SimpleNamespace
    make_run(tmp_path, SimpleNamespace(design_length=1, metadata={})).run()
    result = summary(tmp_path)
    assert result["next_sequence"] == "A"
    assert result["next_sequence_semantics"] == "raw_design_argmax"


def test_decoder_error_is_not_masked_and_state_not_advanced(tmp_path):
    model = backend()

    def invalid(logits, controls):
        raise ValueError("decoder rejected controls")

    model.sequence_from_logits = invalid
    run = make_run(tmp_path, model)
    with pytest.raises(ValueError, match="decoder rejected controls"):
        run.run()
    assert run.state.step == 0
    assert not (tmp_path / "trajectory.jsonl").exists()


def test_af3_decoder_accepts_controls_and_retains_fixed_positions():
    from hallucination_lab.backends.af3 import AF3
    model = object.__new__(AF3)
    model.design_length = 1
    model.positions = np.array([1])
    model.sequence = "NA"
    values = np.zeros((1, 20), np.float32)
    values[0, 1] = 1
    assert model.sequence_from_logits(values) == "NR"
    assert model.sequence_from_logits(values, {"alpha": 2., "hard": 1.}) == "NR"
    with pytest.raises(ValueError, match="alpha"):
        model.sequence_from_logits(values, {"alpha": -1.})
