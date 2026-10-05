"""Chain-rule composition across conditions, targets, or independently prepared models."""
from ..types import objective_result


class Composite:
    """Shared design variables, independently scored conditions.

    combine(losses, controls) returns scalar or (scalar, metrics). This supports
    positive/negative states, margins, smooth worst-case objectives, etc. Its
    scientific interpretation is caller-defined. Models run sequentially;
    gradients combine exactly by the chain rule. Every backend must use the
    same design-variable order, not merely the same length.
    """
    def __init__(self, backends, combine, *, variable_order):
        if not backends or len({b.design_length for b in backends.values()}) != 1:
            raise ValueError("All named backends must share a nonzero design length")
        self.design_length = next(iter(backends.values())).design_length
        if len(variable_order) != self.design_length or len(set(variable_order)) != len(variable_order):
            raise ValueError("Declare one unique variable identifier for every design row")
        self.backends, self.combine = dict(backends), combine
        self.metadata = {"backend": "composite", "variable_order": list(variable_order),
                         "members": {k: b.metadata for k, b in backends.items()},
                         "key_policy": "same key in each condition"}

    def evaluate(self, logits, key, controls=None, *, gradient=True, prediction=False):
        import jax
        import jax.numpy as jnp
        from .common import inputs, evaluation
        logits, key, controls = inputs(logits, key, controls, self.design_length)
        if prediction:
            raise ValueError("Retrieve condition predictions from the named member backends")
        results = {name: backend.evaluate(logits, key, controls, gradient=gradient)
                   for name, backend in self.backends.items()}
        values = {name: jnp.asarray(result.loss) for name, result in results.items()}
        def aggregate(x):
            return objective_result(self.combine(x, controls))
        if gradient:
            (loss, metrics), coefficients = jax.value_and_grad(aggregate, has_aux=True)(values)
            grad = sum(float(coefficients[name]) * result.gradient for name, result in results.items())
        else:
            loss, metrics = aggregate(values)
            grad = None
        metrics = {**{f"condition/{k}": v.loss for k, v in results.items()}, **metrics}
        return evaluation(loss, metrics, grad)
