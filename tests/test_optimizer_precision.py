"""High precision scalar references for the host optimizer, without model calls."""
from copy import deepcopy
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hallucination_lab.engine import State, gradient_step
from hallucination_lab.types import Evaluation


def oracle(x, gradients, optimizer, clip):
    """Evaluate the mathematical recurrence using 80-digit scalar arithmetic."""
    with localcontext() as context:
        context.prec = 80
        current = x.copy()
        first = [Decimal(0)] * x.size
        second = [Decimal(0)] * x.size
        for count, gradient in enumerate(gradients, 1):
            values = [Decimal.from_float(float(g)) for g in gradient.flat]
            norm = sum(g * g for g in values).sqrt()
            if clip is not None and norm > Decimal(str(clip)):
                values = [g * Decimal(str(clip)) / norm for g in values]
            updated = []
            for i, (old, g) in enumerate(zip(current.flat, values)):
                if optimizer == 'adam':
                    first[i] = Decimal('.9') * first[i] + Decimal('.1') * g
                    second[i] = Decimal('.999') * second[i] + Decimal('.001') * g * g
                    m = first[i] / (1 - Decimal('.9') ** count)
                    v = second[i] / (1 - Decimal('.999') ** count)
                    direction = m / (v.sqrt() + Decimal('1e-8'))
                else:
                    direction = g
                updated.append(float(Decimal.from_float(float(old)) - Decimal('.1') * direction))
            current = np.asarray(updated, np.float32).reshape(x.shape)
    return current


@pytest.mark.parametrize('optimizer', ['adam', 'sgd'])
@pytest.mark.parametrize('clip', [None, 1.])
def test_extreme_finite_gradients_against_decimal_reference(optimizer, clip):
    x = np.full((2, 20), .3, np.float32)
    pattern = np.array([1e20, -2e20, 1e-25, -.3, 0], np.float32)
    gradients = [np.resize(pattern * sign, x.shape) for sign in (1., -.5, 2.)]
    state = State(x.copy())
    for g in gradients:
        original = g.copy()
        before = deepcopy(state)
        backend = SimpleNamespace(evaluate=lambda *a, **k: Evaluation(1., {}, g))
        result = gradient_step(state, backend, {'learning_rate': .1},
                               np.array([0, 1], np.uint32), optimizer=optimizer, clip_norm=clip)
        np.testing.assert_array_equal(g, original)
        np.testing.assert_array_equal(state.logits, before.logits)
        for key, value in before.optimizer.items():
            np.testing.assert_array_equal(state.optimizer[key], value)
        assert np.isfinite(result.metrics['gradient_norm'])
        if optimizer == 'adam':
            assert result.optimizer['first'].dtype == result.optimizer['second'].dtype == np.float64
            assert np.isfinite(result.optimizer['second']).all()
        state = State(result.logits, optimizer=result.optimizer)
    expected = oracle(x, gradients, optimizer, clip)
    np.testing.assert_array_max_ulp(state.logits, expected, maxulp=1)
    assert state.logits.dtype == np.float32
    assert not np.array_equal(state.logits, x)


def test_promotes_legacy_moments_and_preserves_transform_interface():
    x = np.zeros((1, 20), np.float32)
    g = np.ones_like(x)
    state = State(x, optimizer={'kind': 'adam', 'count': 3,
                               'first': g * .2, 'second': g * .1})
    original = deepcopy(state.optimizer)
    backend = SimpleNamespace(evaluate=lambda *a, **k: Evaluation(1., {}, g))

    def transform(gradient, current, controls):
        assert gradient.dtype == np.float32
        gradient *= 2
        return gradient

    r = gradient_step(state, backend, {'learning_rate': .1}, np.array([0, 1], np.uint32),
                      transform=transform)
    np.testing.assert_array_equal(g, np.ones_like(g))
    for key, value in original.items():
        np.testing.assert_array_equal(state.optimizer[key], value)
    assert r.optimizer['count'] == 4
    assert r.optimizer['first'].dtype == r.optimizer['second'].dtype == np.float64


@pytest.mark.parametrize('override', [
    {'count': -1}, {'count': 1.5}, {'count': True}, {'count': 2},
    {'first': np.zeros((2, 20))}, {'second': np.full((1, 20), -1.)},
    {'first': np.full((1, 20), np.inf)}, {'second': np.full((1, 20), np.nan)},
])
def test_invalid_adam_state_rejected(override):
    state = State(np.zeros((1, 20), np.float32), optimizer={'kind': 'adam', **override})
    backend = SimpleNamespace(evaluate=lambda *a, **k: Evaluation(1., {}, np.ones((1, 20))))
    with pytest.raises(ValueError):
        gradient_step(state, backend, {'learning_rate': .1}, np.array([0, 1], np.uint32))
    np.testing.assert_array_equal(state.logits, 0.)


def test_unrepresentable_sgd_update_rejected_without_advancing_state():
    x = np.zeros((1, 20), np.float32)
    state = State(x.copy())
    backend = SimpleNamespace(evaluate=lambda *a, **k: Evaluation(
        1., {}, np.full_like(x, np.finfo(np.float32).max)))
    with pytest.raises(FloatingPointError):
        gradient_step(state, backend, {'learning_rate': 2.}, np.array([0, 1], np.uint32),
                      optimizer='sgd')
    np.testing.assert_array_equal(state.logits, x)
    assert not state.optimizer


def test_overflowing_bias_correction_of_external_moments_rejected():
    x = np.zeros((1, 20), np.float32)
    state = State(x.copy(), optimizer={'kind': 'adam', 'count': 1,
                                      'first': np.full(x.shape, 1e150),
                                      'second': np.full(x.shape, 1e308)})
    before = deepcopy(state.optimizer)
    backend = SimpleNamespace(evaluate=lambda *a, **k: Evaluation(1., {}, np.ones_like(x)))
    with pytest.raises(FloatingPointError):
        gradient_step(state, backend, {'learning_rate': .1}, np.array([0, 1], np.uint32))
    np.testing.assert_array_equal(state.logits, x)
    for key, value in before.items():
        np.testing.assert_array_equal(state.optimizer[key], value)


def test_valid_sub_ulp_update_is_allowed():
    x = np.ones((1, 20), np.float32)
    backend = SimpleNamespace(evaluate=lambda *a, **k: Evaluation(1., {}, np.full_like(x, 1e-30)))
    r = gradient_step(State(x.copy()), backend, {'learning_rate': .1},
                      np.array([0, 1], np.uint32), optimizer='sgd')
    np.testing.assert_array_equal(r.logits, x)
