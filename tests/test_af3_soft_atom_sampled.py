"""Exact finite-state oracle for full-chain native-layout sampling."""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hallucination_lab.backends._af3.soft_atom_sampled import (
    draw_identities,
    mean_gradient,
    require_stable_entities,
)
from hallucination_lab.backends.af3 import AF3
from hallucination_lab.sequence import ALPHABET, validate_controls
from hallucination_lab.types import Prediction


@pytest.mark.parametrize("scale", [1.0, 20.0])
def test_expected_sampled_gradient_matches_exhaustive_ad(scale):
    logits = jnp.array([[.02, -.055, .05], [.01, .04, -.03]])
    states = jnp.array(list(product(range(3), repeat=2)))
    # A nonlinear joint-layout effect and a token term that depends on the draw.
    def loss(z, draw):
        p = jax.nn.softmax(scale * z)
        return (draw[0] == draw[1]) * 2. + jnp.sin(p[0, draw[1]] * p[1, draw[0]])

    rows = jnp.arange(2)
    def distribution(z):
        return jax.nn.softmax(scale * z)[rows[None, :], states].prod(axis=-1)

    exact = jax.grad(lambda z: distribution(z) @ jax.vmap(loss, (None, 0))(z, states))(logits)
    losses, direct = jax.vmap(jax.value_and_grad(loss), (None, 0))(logits, states)
    prob = np.asarray(distribution(logits))
    p = np.asarray(jax.nn.softmax(scale * logits))
    np_states = np.asarray(states)
    estimated = np.zeros_like(logits, dtype=np.float64)
    for a, b in product(range(len(states)), repeat=2):
        ids = [a, b]
        estimated += prob[a] * prob[b] * mean_gradient(
            np.asarray(losses)[ids], np.asarray(direct)[ids], np_states[ids], p, scale,
        )
    np.testing.assert_allclose(estimated, exact, atol=2e-6, rtol=3e-6)


def test_60_positions_draw_all_canonical_identities_reproducibly():
    key = jax.random.PRNGKey(7)
    log_p = jnp.full((60, 20), -jnp.log(20.))
    draws = np.asarray(draw_identities(key, log_p, 4096))
    assert draws.shape == (4096, 60)
    np.testing.assert_array_equal(draws, draw_identities(key, log_p, 4096))
    assert not np.array_equal(draws, draw_identities(jax.random.PRNGKey(8), log_p, 4096))
    assert all(set(column) == set(range(20)) for column in draws.T)
    observed = np.eye(20)[draws].mean(axis=0)
    np.testing.assert_allclose(observed, .05, atol=.02, rtol=0.)


def test_full_chain_keeps_entities_and_constructor_options_explicit():
    require_stable_entities({"A": "GGGG", "B": "AAA"}, "B", [0, 1, 2])
    with pytest.raises(ValueError, match="entities"):
        require_stable_entities({"A": "GAG", "B": "AAA"}, "B", [0, 2])
    require_stable_entities({"A": "GGG", "B": "AAA"}, "B", [0, 2])
    args = dict(input_json="unused", design_chain="A", objective=None, model_dir="unused")
    for mode, samples in (("soft", 2), ("hard", 2), ("soft_sampled", 1), ("soft_sampled", True)):
        with pytest.raises(ValueError, match="soft_atom_samples"):
            AF3(**args, atom_mode=mode, soft_atom_samples=samples)


def test_backend_passes_each_sample_to_native_carrier_and_exports_a_real_draw(monkeypatch):
    from hallucination_lab.backends._af3 import soft_atom_sampled

    af = AF3.__new__(AF3)
    af.positions = np.arange(1, 61)
    af.sequence = "A" * 62
    af._soft_atom_samples = 3
    af._full_parameters = {}
    p = jnp.full((60, 20), .05)
    af._sample_distribution = lambda *_: (p, jnp.log(p))
    draws = np.array([[0] * 60, [1] * 60, [1] * 60])
    monkeypatch.setattr(soft_atom_sampled, "draw_identities", lambda *_: draws)
    visited = []

    def batch(seq):
        visited.append(seq)
        return seq

    pred = Prediction(None, None, None, None, None, p, {})
    af._batch = batch
    af._gradient_full = lambda *_: ((1., ({"contact": 1.}, pred)), np.zeros_like(p))
    af._gradient = lambda *_: ((4., {"contact": 4.}), np.zeros_like(p))
    got = af._evaluate_sampled(np.zeros_like(p), jax.random.PRNGKey(0), validate_controls(None),
                               gradient=True, prediction=True)
    assert visited == ["A" + ALPHABET[aa] * 60 + "A" for aa in (0, 1)]
    assert got.loss == got.metrics["contact"] == 3.
    np.testing.assert_allclose(got.gradient[:, :2], np.tile([-1., 1.], (60, 1)), atol=1e-7)
    np.testing.assert_array_equal(got.prediction.native["sampled_layout_identities"], draws)
    np.testing.assert_array_equal(got.prediction.native["sampled_layout_losses"], [1., 4., 4.])

    visited.clear()
    af._forward = lambda *_: (2., ({"contact": 2.}, pred))
    locked = af._evaluate_sampled(np.zeros_like(p), jax.random.PRNGKey(0),
                                  validate_controls({"hard": 1}), gradient=False,
                                  prediction=True)
    assert visited == ["A" * 62]
    assert locked.loss == 2. and locked.gradient is None
