import json
import numpy as np
import pytest
from hallucination_lab.assessment import paired_evaluate
from hallucination_lab.types import Evaluation


def condition(name="held-fixed", **controls):
    return {"name": name, "key": [9, 5], "controls": controls}


def test_fixed_pairs_do_not_reuse_training_scores(quadratic, tmp_path):
    before, after = np.zeros((2, 20)), np.full((2, 20), .7)
    path = tmp_path / "paired.json"
    report = paired_evaluate(quadratic, before, after,
                             conditions=[condition(), condition("second", target=.6)], output=path)
    assert report["all_valid"] and report["improved_pairs"] == 2
    assert report["requested_pairs"] == report["complete_pairs"] == 2
    assert report["endpoint_sha256"]["before"] != report["endpoint_sha256"]["after"]
    assert report == json.loads(path.read_text())
    with pytest.raises(FileExistsError):
        paired_evaluate(quadratic, before, after, conditions=[condition()], output=path)


def test_failures_and_nonfinite_values_remain_in_denominator():
    class Backend:
        design_length, metadata = 1, {}

        def evaluate(self, values, key, controls, *, gradient):
            if controls["fail"] == 1:
                raise RuntimeError("kernel failed")
            return Evaluation(float("nan") if controls["fail"] == 2 else 1, {})

    report = paired_evaluate(Backend(), np.zeros((1, 20)), np.ones((1, 20)),
                             conditions=[condition(str(i), fail=i) for i in range(3)])
    assert not report["all_valid"]
    assert report["requested_pairs"] == 3 and report["complete_pairs"] == 1
    assert report["improved_pairs"] == 0
    assert report["pairs"][1]["before"]["error_type"] == "RuntimeError"
    assert report["pairs"][2]["after"]["error_type"] == "FloatingPointError"


def test_backend_cannot_mutate_other_endpoint_or_protocol():
    class Backend:
        design_length, metadata = 1, {}

        def evaluate(self, values, key, controls, *, gradient):
            assert controls["nested"] == [1] and key.tolist() == [9, 5]
            assert not gradient
            loss = values.sum()
            values[:] = 500
            controls["nested"].append(2)
            key[:] = 0
            return Evaluation(float(loss), {})

    before, after = np.zeros((1, 20)), np.ones((1, 20))
    protocol = [condition(nested=[1])]
    result = paired_evaluate(Backend(), before, after, conditions=protocol)
    assert result["pairs"][0]["delta"]["loss"] == 20
    assert not before.any() and (after == 1).all()
    assert protocol == [condition(nested=[1])]


@pytest.mark.parametrize("conditions", [[], [condition(), condition()],
    [{"name": "x", "key": [-1, 0], "controls": {}}],
    [{"name": "x", "key": [True, 0], "controls": {}}],
    [{"name": "x", "key": [2**32, 0], "controls": {}}]])
def test_protocol_validation(quadratic, conditions):
    with pytest.raises(ValueError):
        paired_evaluate(quadratic, np.zeros((2, 20)), np.ones((2, 20)), conditions=conditions)


def test_metric_contract_mismatch_is_not_hidden():
    class Backend:
        design_length, metadata = 1, {}

        def evaluate(self, values, *args, **kwargs):
            return Evaluation(0, {str(values.sum()): 1})

    report = paired_evaluate(Backend(), np.zeros((1, 20)), np.ones((1, 20)),
                             conditions=[condition()])
    assert not report["all_valid"]
    assert report["complete_pairs"] == 0


def test_numerical_guard_failures_retain_diagnostics(tmp_path):
    from hallucination_lab.numerics import NumericalAuditError

    class Backend:
        design_length, metadata = 1, {}

        def evaluate(self, *args, **kwargs):
            raise NumericalAuditError({"passed": False, "delta": np.float32(.02),
                                       "key": np.array([9, 5], np.uint32)})

    path = tmp_path / "assessment.json"
    report = paired_evaluate(Backend(), np.zeros((1, 20)), np.ones((1, 20)),
                             conditions=[condition()], output=path)
    assert not report["all_valid"]
    saved = json.loads(path.read_text())
    assert saved["pairs"][0]["after"]["audit"]["key"] == [9, 5]
    assert saved["pairs"][0]["before"]["audit"]["delta"] == pytest.approx(.02)


def test_finite_endpoints_with_overflowing_delta_do_not_count_as_improvement(tmp_path):
    class Backend:
        design_length, metadata = 1, {}

        def evaluate(self, values, *args, **kwargs):
            return Evaluation(-1e308 if values.any() else 1e308, {})

    report = paired_evaluate(Backend(), np.zeros((1, 20)), np.ones((1, 20)),
                             conditions=[condition()], output=tmp_path / "overflow.json")
    assert not report["all_valid"] and report["improved_pairs"] == 0
    assert report["pairs"][0]["error"] == "Nonfinite endpoint difference"
