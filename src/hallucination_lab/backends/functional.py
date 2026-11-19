"""Adapter for caller-owned differentiable models and small numerical tests."""
from .common import inputs, evaluation, numerical_policy, with_precision
from ..types import objective_result


class Functional:
    """forward(logits, key, controls) -> Prediction; objective(prediction, controls).

    This contains no protein predictor. It is a plugin seam and a way to test
    research algorithms cheaply before invoking a real model.
    """
    def __init__(self, forward, objective, *, design_length, identifier, matmul_precision="highest"):
        import jax
        numerics = numerical_policy(matmul_precision)
        self._matmul_precision = matmul_precision
        if type(design_length) is not int or design_length < 1 or not identifier:
            raise ValueError("Provide positive design_length and stable model identifier")
        self.design_length = design_length
        self.metadata = {"numerics": numerics, "backend": "functional", "identifier": identifier,
                         "design_length": design_length, "jax": jax.__version__}
        def scored_prediction(logits, key, controls):
            predicted = forward(logits, key, controls)
            loss, metrics = objective_result(objective(predicted, controls))
            return loss, (metrics, predicted)

        def score(logits, key, controls):
            loss, (metrics, _) = scored_prediction(logits, key, controls)
            return loss, metrics

        self._forward = jax.jit(scored_prediction)
        self._gradient_full = jax.jit(jax.value_and_grad(scored_prediction, has_aux=True))
        self._score = jax.jit(score)
        self._gradient = jax.jit(jax.value_and_grad(score, has_aux=True))

    @with_precision
    def evaluate(self, logits, key, controls=None, *, gradient=True, prediction=False):
        values, key, controls = inputs(logits, key, controls, self.design_length)
        predicted = None
        if prediction:
            if gradient:
                (loss, (metrics, predicted)), grad = self._gradient_full(values, key, controls)
            else:
                loss, (metrics, predicted) = self._forward(values, key, controls)
                grad = None
        elif gradient:
            (loss, metrics), grad = self._gradient(values, key, controls)
        else:
            loss, metrics = self._score(values, key, controls)
            grad = None
        return evaluation(loss, metrics, grad, predicted)
