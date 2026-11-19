import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hallucination_lab import losses


def test_interval_exact_event_and_gradient(prediction):
    mask = np.array([[0, 1], [0, 0]])
    fn = losses.interval(mask, [4, 8], upper=8)
    expected = -np.log(np.asarray(jax.nn.softmax(prediction.logits[0, 1]))[:2].sum())
    np.testing.assert_allclose(fn(prediction, {}), expected, rtol=1e-5)
    def objective(logits):
        return fn(prediction._replace(logits=logits), {})
    grad = jax.grad(objective)(prediction.logits)
    eps = 1e-3
    direction = jnp.ones_like(grad).at[0, 1, 0].set(2)
    finite_difference = (objective(prediction.logits + eps*direction)
                         - objective(prediction.logits - eps*direction))/(2*eps)
    np.testing.assert_allclose(jnp.sum(grad*direction), finite_difference, rtol=2e-3, atol=1e-4)
    assert np.count_nonzero(np.asarray(grad)[1]) == 0


def test_distribution_cross_entropy(prediction):
    target = np.zeros((2, 2, 3), np.float32)
    target[:, :, 1] = 1
    fn = losses.distribution(target, np.ones((2, 2)), edges=[4, 8])
    expected = -jax.nn.log_softmax(prediction.logits, -1)[:, :, 1].mean()
    np.testing.assert_allclose(fn(prediction, {}), expected)


@pytest.mark.parametrize("compiled", [False, True])
def test_distribution_zero_target_mass_at_extreme_finite_logits(prediction, compiled):
    target = np.zeros((2, 2, 3), np.float32)
    target[:, :, 0] = 1
    fn = losses.distribution(target, np.ones((2, 2)), edges=[4, 8])
    logits = jnp.broadcast_to(jnp.array([3e38, -3e38, -3e38], jnp.float32), (2, 2, 3))
    evaluate = jax.value_and_grad(lambda x: fn(prediction._replace(logits=x), {}))
    if compiled:
        evaluate = jax.jit(evaluate)
    value, gradient = evaluate(logits)
    np.testing.assert_array_equal(value, 0)
    np.testing.assert_array_equal(gradient, np.zeros((2, 2, 3)))


def test_exclusion_complement(prediction):
    fn = losses.interval(np.ones((2, 2)), [4, 8], upper=8, exclude=True)
    np.testing.assert_allclose(fn(prediction, {}), -jax.nn.log_softmax(prediction.logits, -1)[:, :, 2].mean())


def test_topk_selects_best_partner(prediction):
    fn = losses.contact(np.ones((2, 2)), [4, 8], cutoff=8, partners=1)
    logmass = jax.scipy.special.logsumexp(jax.nn.log_softmax(prediction.logits, -1)[:, :, :2], -1)
    np.testing.assert_allclose(fn(prediction, {}), -logmass.max(-1).mean())
    grad = jax.grad(lambda x: fn(prediction._replace(logits=x), {}))(prediction.logits)
    assert np.isfinite(grad).all()


def test_topk_inactive_rows_have_finite_gradients(prediction):
    fn = losses.contact([[0, 1], [0, 0]], [4, 8], cutoff=8, partners=1)
    grad = jax.grad(lambda x: fn(prediction._replace(logits=x), {}))(prediction.logits)
    assert np.isfinite(grad).all()


def test_padding_ignored_and_empty_is_error(prediction):
    fn = losses.interval([[1, 0], [0, 0]], [4, 8], upper=8)
    assert np.isnan(fn(prediction._replace(token_mask=jnp.array([0, 1])), {}))
    with pytest.raises(ValueError):
        losses.interval(np.zeros((2, 2)), [4, 8], upper=8)


def test_wrong_bin_definitions_rejected(prediction):
    fn = losses.interval(np.ones((2, 2)), [3, 9], upper=9)
    assert np.isnan(fn(prediction, {}))
    with pytest.raises(ValueError, match="No bins"):
        losses.bin_event([4, 8], lower=4.1, upper=7.9)
    np.testing.assert_array_equal(losses.bin_event([4, 8], lower=4.1, upper=7.9, mode="overlap"), [0, 1, 0])


@pytest.mark.parametrize("mask", [np.ones((2,3)), [[0, -1], [0, 0]], [[0, np.nan], [0, 0]]])
def test_invalid_masks(mask):
    with pytest.raises(ValueError):
        losses.contact(mask, [4, 8])


def test_no_silent_target_normalization():
    with pytest.raises(ValueError, match="sum to one"):
        losses.distribution(np.ones((2, 2, 3)), np.ones((2, 2)), edges=[4, 8])


def test_weight_controls(prediction):
    term = losses.interval(np.ones((2, 2)), [4, 8], upper=8)
    objective = losses.weighted({"event": (2, term)})
    loss, metrics = objective(prediction, {"event": 3})
    np.testing.assert_allclose(loss, 3*metrics["event"])


@pytest.mark.parametrize("upper_type", [float, np.float64])
@pytest.mark.parametrize("mode, expected", [("contained", 3), ("overlap", 3)])
def test_exact_declared_boundary_does_not_depend_on_scalar_type(upper_type, mode, expected):
    edges = np.linspace(2., 22., 63)
    upper = upper_type(edges[2])
    assert losses.bin_event(edges, upper=upper, mode=mode).sum() == expected


@pytest.mark.parametrize("name", [
    "soft", "hard", "temperature", "alpha", "learning_rate", "model_index",
    "recycles", "pssm_hard", "trials", "dropout",
])
def test_builtin_controls_cannot_silently_rescale_objective_terms(name):
    with pytest.raises(ValueError, match="reserved"):
        losses.weighted({name: (3., lambda p, c: 2.)})


@pytest.mark.parametrize("family", ["interval", "contact", "distribution"])
@pytest.mark.parametrize("inactive", ["mask", "padding"])
def test_masked_extreme_logits_do_not_affect_value_or_gradient(prediction, family, inactive):
    mask = [[1, 0], [0, 0]] if inactive == "mask" else np.ones((2, 2))
    valid = [1, 1] if inactive == "mask" else [1, 0]
    pred = prediction._replace(token_mask=jnp.asarray(valid))
    target = np.ones((2, 2, 3), dtype=np.float32) / 3
    if family == "distribution":
        objective = losses.distribution(target, mask, edges=[4., 8.])
    elif family == "interval":
        objective = losses.interval(mask, [4., 8.], upper=8.)
    else:
        objective = losses.contact(mask, [4., 8.], cutoff=8., partners=1)
    score = jax.jit(jax.value_and_grad(lambda x: objective(pred._replace(logits=x), {})))
    reference = jnp.zeros_like(pred.logits)
    extreme = jnp.broadcast_to(jnp.array([-3e38, -3e38, 3e38], jnp.float32), reference.shape)
    contaminated = extreme.at[0, 0].set(reference[0, 0])
    expected_loss, expected_grad = score(reference)
    actual_loss, actual_grad = score(contaminated)
    assert np.isfinite(actual_grad).all() and np.isfinite(actual_loss)
    np.testing.assert_array_equal(actual_loss, expected_loss)
    np.testing.assert_array_equal(actual_grad, expected_grad)


def test_weighted_mean_large_finite_values_and_jvp():
    values = jnp.array([1.e30, 2.e30], jnp.float32)
    weights = jnp.array([1.e30, 3.e30], jnp.float32)
    val, (gv, gw) = jax.jit(jax.value_and_grad(losses._mean, argnums=(0, 1)))(values, weights)
    np.testing.assert_allclose(val, 1.75e30, rtol=2e-6)
    np.testing.assert_allclose(gv, [.25, .75], rtol=2e-6)
    np.testing.assert_allclose(gw, [-.1875, .0625], rtol=2e-6)
    # Even a finite objective can have an unrepresentable weight derivative.
    # The sequence JVP differentiates values only, so it remains well defined.
    tiny = jnp.array([1e-30, 3e-30], jnp.float32)
    val, tangent = jax.jit(lambda x: jax.jvp(lambda v: losses._mean(v, tiny),
                                            (x,), (jnp.ones_like(x),)))(values)
    np.testing.assert_allclose(val, 1.75e30, rtol=2e-6)
    np.testing.assert_allclose(tangent, 1., rtol=2e-6)
