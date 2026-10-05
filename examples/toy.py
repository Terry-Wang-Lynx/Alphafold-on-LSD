"""CPU-only numerical demo. This is NOT a protein structure predictor.

    PYTHONPATH=examples hallulab run toy:build --max-steps 3
"""
import jax.numpy as jnp
from hallucination_lab import Prediction, Run, Stage, initial_logits
from hallucination_lab.backends.functional import Functional
from hallucination_lab.provenance import experiment_id


def build(config):
    def forward(logits, key, controls):
        return Prediction(jnp.zeros((4, 4, 3)), jnp.array([4., 8.]), jnp.ones(4),
                          jnp.arange(4), jnp.ones(4), logits, {})

    def objective(prediction, controls):
        error = jnp.mean((prediction.sequence - 0.5)**2)
        return error, {"squared_error": error}

    backend = Functional(forward, objective, design_length=4, identifier="numerical-demo-v1")
    return Run(backend, initial_logits(4, seed=3), [Stage("demo", 10)],
               problem_id=experiment_id(files=[__file__], settings={k: v for k, v in config.items()
                                                  if k not in {"output", "data_dir", "model_dir", "input_json", "pair_mask"}}),
               output=config.get("output", "runs/toy"))
