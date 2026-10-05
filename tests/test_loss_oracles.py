"""Independent closed-form float64 oracles for loss values and gradients.

These check the objective layer, not an AF2/AF3 trunk derivative. The reference
uses NumPy and analytic derivatives, with padding, directed masks, weight
scales and untied top-k choices evaluated independently of the implementation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hallucination_lab import Prediction, losses


def log_probabilities(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def case(scale):
    rng = np.random.default_rng(371)
    logits = rng.normal(0, scale, (7, 7, 64)).astype(np.float32)
    edges = np.linspace(2, 22, 63, dtype=np.float32)
    mask = rng.uniform(.1, 1, (5, 5)).astype(np.float32)
    mask[0] = 0
    np.fill_diagonal(mask, 0)
    target = rng.dirichlet(np.ones(64), size=(5, 5)).astype(np.float32)
    valid = np.array([1, 1, 1, 1, 0, 0, 0], np.float32)
    prediction = Prediction(jnp.asarray(logits), jnp.asarray(edges), jnp.asarray(valid),
                            jnp.arange(7), jnp.ones(7), jnp.zeros((2, 20)), {})
    return prediction, mask, target, valid, edges


def oracle(prediction, mask, target, valid, family):
    raw = np.asarray(prediction.logits, dtype=np.float64)
    logp = log_probabilities(raw)
    p = np.exp(logp)
    weights = np.pad(mask.astype(np.float64), ((0, 2), (0, 2)))
    weights *= valid[:, None] * valid[None, :]
    if family == 'distribution':
        q = np.pad(target.astype(np.float64), ((0, 2), (0, 2), (0, 0)))
        pair_loss = -(q * logp).sum(-1)
        pair_gradient = p * q.sum(-1, keepdims=True) - q
    else:
        event = np.arange(64) < 19
        if family == 'exclude':
            event = ~event
        conditional = np.zeros_like(p)
        conditional[..., event] = np.exp(log_probabilities(raw[..., event]))
        # log sum exp of the selected logits, independent of the global normalizer.
        chosen = raw[..., event]
        max_chosen = chosen.max(-1)
        log_z_event = max_chosen + np.log(np.exp(chosen - max_chosen[..., None]).sum(-1))
        max_all = raw.max(-1)
        log_z_all = max_all + np.log(np.exp(raw - max_all[..., None]).sum(-1))
        pair_loss = log_z_all - log_z_event
        pair_gradient = p - conditional
        if family == 'topk':
            selected = np.zeros_like(weights)
            active = np.flatnonzero(weights.sum(-1))
            for row in active:
                eligible = np.flatnonzero(weights[row])
                ranked = eligible[np.argsort(pair_loss[row, eligible])]
                assert pair_loss[row, ranked[2]] - pair_loss[row, ranked[1]] > 1e-5
                selected[row, ranked[:2]] = 1 / (2 * len(active))
            weights = selected
    weights /= weights.sum()
    return float((weights * pair_loss).sum()), pair_gradient * weights[..., None]


@pytest.mark.parametrize('family', ['distribution', 'event', 'exclude', 'topk'])
@pytest.mark.parametrize('scale', [1.0, 35.0])
def test_values_and_gradients_against_closed_form(family, scale):
    prediction, mask, target, valid, edges = case(scale)
    if family == 'distribution':
        term = losses.distribution(target, mask, edges=edges)
    elif family == 'topk':
        mask = (mask > 0).astype(np.float32)
        term = losses.contact(mask, edges, cutoff=float(edges[18]), partners=2)
    else:
        term = losses.interval(mask, edges, upper=float(edges[18]), exclude=family == 'exclude')
    expected, expected_gradient = oracle(prediction, mask, target, valid, family)
    # Fixed positive scaling must not hide an inaccurate derivative behind atol.
    for weight in (1e-6, 1., 1e6):
        objective = losses.weighted({'term': (weight, term)})
        value, gradient = jax.value_and_grad(
            lambda z: objective(prediction._replace(logits=z), {})[0])(prediction.logits)
        np.testing.assert_allclose(float(value) / weight, expected, rtol=2e-5, atol=2e-6)
        observed = np.asarray(gradient, dtype=np.float64) / weight
        relative = np.linalg.norm(observed - expected_gradient) / np.linalg.norm(expected_gradient)
        assert relative < 5e-5
        assert np.count_nonzero(observed[4:]) == np.count_nonzero(observed[:, 4:]) == 0


@pytest.mark.parametrize('coefficients', [(0.7, -1.3), (0.0, 2.0)])
def test_weight_override_combination_against_independent_oracles(coefficients):
    """Each mask normalizes its own term; controls change only the combination."""
    prediction, mask, target, valid, edges = case(1.0)
    exclusion_mask = mask.T.copy()
    ce, gce = oracle(prediction, mask, target, valid, 'distribution')
    exclusion, gexclusion = oracle(prediction, exclusion_mask, target, valid, 'exclude')
    objective = losses.weighted({
        'shape': (3.0, losses.distribution(target, mask, edges=edges)),
        'exclusion': (-4.0, losses.interval(exclusion_mask, edges,
                                           upper=float(edges[18]), exclude=True)),
    })
    a, b = coefficients
    controls = {'shape': a, 'exclusion': b}
    (value, metrics), gradient = jax.value_and_grad(
        lambda z: objective(prediction._replace(logits=z), controls), has_aux=True)(
            prediction.logits)
    np.testing.assert_allclose(float(value), a * ce + b * exclusion,
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose([float(metrics['shape']), float(metrics['exclusion'])],
                               [ce, exclusion], rtol=2e-5, atol=2e-6)
    expected_gradient = a * gce + b * gexclusion
    relative = np.linalg.norm(np.asarray(gradient, np.float64) - expected_gradient)
    relative /= np.linalg.norm(expected_gradient)
    assert relative < 5e-5
    assert np.count_nonzero(np.asarray(gradient)[4:]) == 0
    assert np.count_nonzero(np.asarray(gradient)[:, 4:]) == 0
