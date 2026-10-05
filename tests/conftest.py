import numpy as np
import pytest
from hallucination_lab import Prediction
from hallucination_lab.backends.functional import Functional


@pytest.fixture
def quadratic():
    import jax.numpy as jnp
    def forward(logits, key, controls):
        return Prediction(jnp.zeros((2, 2, 3)), jnp.array([4., 8.]), jnp.ones(2),
                          jnp.arange(2), jnp.ones(2), logits, {})
    def objective(pred, controls):
        target = controls.get("target", 0.7)
        value = jnp.mean((pred.sequence-target)**2)
        return controls.get("weight", 1.0) * value, {"mse": value}
    return Functional(forward, objective, design_length=2, identifier="quadratic-v1")


@pytest.fixture
def prediction():
    import jax.numpy as jnp
    return Prediction(jnp.array([[[1., 2., 3.], [2., 0., -1.]],
                                  [[-1., 0., 2.], [1., 1., 1.]]]),
                      jnp.array([4., 8.]), jnp.ones(2), jnp.arange(2), jnp.ones(2),
                      jnp.zeros((2, 20)), {})


@pytest.fixture
def key():
    return np.array([0, 7], np.uint32)
