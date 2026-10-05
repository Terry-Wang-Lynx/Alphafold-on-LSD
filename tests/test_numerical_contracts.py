"""Numerical policy, same-evaluation outputs, and provenance contracts."""
from types import SimpleNamespace
import jax
import numpy as np
import pytest
from hallucination_lab import Run, Stage, initial_logits
from hallucination_lab.backends.common import numerical_policy, with_precision


def test_precision_scope_restores_caller_after_error():
    @with_precision
    def probe(self):
        assert jax.config.jax_default_matmul_precision == 'highest'
        raise RuntimeError('probe')
    with jax.default_matmul_precision('default'):
        with pytest.raises(RuntimeError):
            probe(SimpleNamespace(_matmul_precision='highest'))
        assert jax.config.jax_default_matmul_precision == 'default'


def test_precision_changes_checkpoint_identity(quadratic):
    args = (quadratic, initial_logits(2), [Stage('test', 1)])
    first = Run(*args, problem_id='numerical-policy')
    quadratic.metadata['numerics'] = numerical_policy('default')
    second = Run(*args, problem_id='numerical-policy')
    assert first.fingerprint != second.fingerprint
    with pytest.raises(ValueError, match='matmul_precision'):
        numerical_policy('typo')


@pytest.mark.parametrize('gradient', [False, True])
def test_af3_prediction_and_score_use_one_evaluation(prediction, gradient, key):
    from hallucination_lab.backends.af3 import AF3
    b = AF3.__new__(AF3)
    b.design_length = 2
    b._matmul_precision = 'highest'
    b._carrier = lambda *args: {}
    calls = []
    def same_forward(*args):
        calls.append('forward')
        return 7., ({'marker': 7.}, prediction)
    def same_gradient(*args):
        calls.append('gradient')
        return same_forward(*args), np.ones((2, 20), np.float32)
    def forbidden(*args):
        raise AssertionError('Separate score/prediction path must not run')
    b._forward, b._gradient_full = same_forward, same_gradient
    b._score = b._gradient = b._predict = forbidden
    result = b.evaluate(initial_logits(2), key, gradient=gradient, prediction=True)
    assert result.loss == 7. and result.prediction is not None
    assert calls.count('forward') == 1
    assert (result.gradient is not None) == gradient


@pytest.mark.parametrize('gradient', [False, True])
def test_functional_returned_prediction_reproduces_loss(quadratic, key, gradient):
    result = quadratic.evaluate(initial_logits(2), key, gradient=gradient, prediction=True)
    independently_computed = np.mean((np.asarray(result.prediction.sequence)-.7)**2)
    np.testing.assert_allclose(result.loss, independently_computed, rtol=1e-6)
    assert quadratic.metadata['numerics']['matmul_precision'] == 'highest'


@pytest.mark.parametrize('override,expected', [(None, False), ({'use_bfloat16': True}, True)])
def test_af2_explicit_research_dtype_default(monkeypatch, override, expected):
    import sys
    from hallucination_lab.backends.af2 import AF2
    captured = {}
    class PreparedEnough(Exception):
        pass
    def constructor(**kwargs):
        captured.update(kwargs)
        raise PreparedEnough()
    monkeypatch.setitem(sys.modules, 'colabdesign', SimpleNamespace(mk_afdesign_model=constructor))
    with pytest.raises(PreparedEnough):
        AF2(sequence='A', objective=lambda p, c: 0., data_dir='.', model_options=override)
    assert captured['use_bfloat16'] is expected
