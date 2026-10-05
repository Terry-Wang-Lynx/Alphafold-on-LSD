import json
import numpy as np
import pytest
from hallucination_lab import Run, Stage
from hallucination_lab.numerics import NumericalAuditError
from hallucination_lab.types import Evaluation


def test_guard_rejection_persists_recoverable_state_without_committing(quadratic, tmp_path):
    def fail(state, backend, controls, key):
        error = NumericalAuditError({"passed": False, "controls": controls,
                                     "value": np.float32(1.0)})
        error.evaluation = Evaluation(1.0, {}, gradient=np.ones_like(state.logits))
        error.input_logits = state.logits + 2  # A custom step may reject a proposal.
        raise error

    initial = np.zeros((2, 20), np.float32)
    run = Run(quadratic, initial, [Stage("audit", 1)], problem_id="rejection-v1",
              step_fn=fail, output=tmp_path)
    for _ in range(2):
        with pytest.raises(NumericalAuditError) as caught:
            run.run()
        assert caught.value.evidence_directory
    assert run.state.step == 0 and not (tmp_path / "trajectory.jsonl").exists()
    folders = sorted((tmp_path / "failures").iterdir())
    assert len(folders) == 2
    report = json.loads((folders[0] / "report.json").read_text())
    assert report["status"] == "numerical_failure" and report["step"] == 0
    assert not report["audit"]["passed"]
    with np.load(folders[0] / "rejected.npz", allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["state_logits"], initial)
        np.testing.assert_array_equal(raw["rejected_logits"], initial + 2)
        assert (raw["rejected_gradient"] == 1).all()
    run.restore(folders[0] / "state.npz")
    assert run.state.step == 0
