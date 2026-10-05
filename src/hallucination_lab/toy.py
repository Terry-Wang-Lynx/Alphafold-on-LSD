"""Installed CPU-only numerical demo. This is not a protein structure predictor.

Run it from any directory after installing the ``jax`` extra:

    hallulab run hallucination_lab.toy:build --max-steps 3
"""
import jax.numpy as jnp
from .engine import Run, Stage
from .sequence import initial_logits
from .types import Prediction
from .backends.functional import Functional
from .provenance import experiment_id


def build(config):
    def forward(logits, key, controls):
        return Prediction(jnp.zeros((4, 4, 3)), jnp.array([4., 8.]), jnp.ones(4),
                          jnp.arange(4), jnp.ones(4), logits, {})

    def objective(prediction, controls):
        error = jnp.mean((prediction.sequence - 0.5)**2)
        return error, {"squared_error": error}

    backend = Functional(forward, objective, design_length=4, identifier="numerical-demo-v1")
    settings = {key: value for key, value in config.items()
                if key not in {"output", "data_dir", "model_dir", "input_json", "pair_mask"}}
    return Run(backend, initial_logits(4, seed=3), [Stage("demo", 10)],
               problem_id=experiment_id(files=[__file__], settings=settings),
               output=config.get("output", "runs/toy"))
