"""User-side loss, optimizer, and gradient transform; no framework edits."""
from functools import partial
import jax.numpy as jnp
from hallucination_lab import Run, Stage, gradient_step
from hallucination_lab.provenance import experiment_id


def native_objective(prediction, controls):
    """AF2-only example. pLDDT here is ColabDesign's normalized [0,1] value."""
    confidence = prediction.native["plddt"]
    loss = 1 - jnp.sum(confidence * prediction.token_mask) / prediction.token_mask.sum()
    return loss, {"confidence_loss": loss}


def center_gradient(gradient, state, controls):
    """Example custom gradient transform; assess its utility in your experiment."""
    return gradient - gradient.mean(-1, keepdims=True)


def make_run(backend, logits, output):
    step = partial(gradient_step, optimizer="sgd", clip_norm=1., transform=center_gradient)
    return Run(backend, logits, [Stage("custom", 10, learning_rate=0.02)],
               step_fn=step, output=output,
               problem_id=experiment_id(files=[__file__], label="custom-interface"))
