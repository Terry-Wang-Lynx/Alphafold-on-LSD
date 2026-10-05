import json

import numpy as np
import pytest

from hallucination_lab.numerics import (NumericalAuditError, Tolerances,
                                        audit_backend, compare_evaluations)
from hallucination_lab.types import Evaluation


def result(gradient, loss=1.):
    return Evaluation(loss, {}, np.asarray(gradient, dtype=float))


def test_gradient_scale_failure_cannot_hide_behind_direction_and_update():
    r = compare_evaluations(result([[1., 2.]]), result([[10., 20.]]))
    assert r['checks']['gradient_cosine']
    assert r['checks']['normalized_update_rms']
    assert not r['checks']['gradient_relative_l2']
    assert not r['passed']


def test_zero_missing_nonfinite_and_opposite_gradients():
    assert compare_evaluations(result([[0.]]), result([[0.]]))['passed']
    for x, y in [(result([[0.]]), result([[1.]])),
                 (result([[1.]]), result([[-1.]])),
                 (result([[np.nan]]), result([[1.]])),
                 (Evaluation(1., {}), result([[1.]]))]:
        report = compare_evaluations(x, y)
        assert not report['passed']
        json.dumps(report, allow_nan=False)


class Backend:
    metadata = {'runtime': 'analytic-test'}
    def evaluate(self, x, key, controls, *, gradient=True):
        return Evaluation(float(np.square(x).sum()), {}, 2*x if gradient else None)
    def probe_forward(self, x, key, controls):
        return Evaluation(float(np.square(x).sum()) + .01, {})


def test_audit_detects_independent_forward_and_saves_before_raising(tmp_path):
    path = tmp_path/'audit.json'
    with pytest.raises(NumericalAuditError) as exc:
        audit_backend(Backend(), np.ones((1,20)), np.array([1,2],np.uint32), output=path)
    report = json.loads(path.read_text())
    assert report == exc.value.report
    assert report['checks']['repeat']['passed']
    assert report['checks']['score']['passed']
    assert not report['checks']['independent_forward']['passed']


def test_audit_marks_nonfinite_as_numerical_failure():
    class Nonfinite(Backend):
        def evaluate(self, *args, **kwargs):
            raise FloatingPointError('nonfinite sequence gradient')
    r = audit_backend(Nonfinite(), np.ones((1,20)), np.array([1,2],np.uint32), strict=False)
    assert not r['passed']
    assert 'numerical_exception' in r


@pytest.mark.parametrize('kwargs', [{'loss_rtol': -1}, {'loss_atol': float('nan')},
                                    {'gradient_cosine_min': 2}])
def test_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        Tolerances(**kwargs)


def test_outputs_are_created_without_overwriting_and_controls_are_isolated(tmp_path):
    class Mutating(Backend):
        def evaluate(self, x, key, controls, *, gradient=True):
            assert controls['nested']['values'] == [1]
            controls['nested']['values'].append(2)
            return super().evaluate(x, key, controls, gradient=gradient)
    controls = {'nested': {'values': [1]}}
    path = tmp_path/'new-directory'/'audit.json'
    args = (Mutating(), np.ones((1,20)), np.array([1,2],np.uint32), controls)
    audit_backend(*args, output=path, strict=False)
    assert controls == {'nested': {'values': [1]}}
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        audit_backend(*args, output=path, strict=False)
    assert path.read_bytes() == original
    path.unlink()
    with pytest.raises(FileExistsError):
        audit_backend(*args, output=path, strict=False)


def test_nested_guard_failure_preserves_evidence(tmp_path):
    class Guarded(Backend):
        def evaluate(self, *args, **kwargs):
            raise NumericalAuditError({'passed': False, 'scope': 'scalar'},
                                      evaluation=result([[1.]]))
    path = tmp_path/'guard.json'
    report = audit_backend(Guarded(), np.ones((1,20)), np.array([1,2],np.uint32),
                           output=path, strict=False)
    assert report['backend_guard']['scope'] == 'scalar'
    np.testing.assert_array_equal(np.load(path.with_suffix('.npz'))['rejected_gradient'], [[1.]])
