"""Chain-rule composition across conditions, targets, or independently prepared models."""
import numpy as np

from ..sequence import decode
from ..types import objective_result


class Composite:
    """Shared design rows across sequential, independently scored conditions.

    variable_order names the common rows. Native backends must use identical
    residue numbering, or callers must supply each member's ordered local
    indices in member_positions to declare the alignment explicitly. Fixed
    backgrounds may differ. The active rows must decode to one shared sequence.
    """
    def __init__(self, backends, combine, *, variable_order, member_positions=None):
        if not backends or len({b.design_length for b in backends.values()}) != 1:
            raise ValueError("All named backends must share a nonzero design length")
        self.design_length = next(iter(backends.values())).design_length
        if self.design_length < 1 or len(variable_order) != self.design_length or len(
                set(variable_order)) != self.design_length:
            raise ValueError("Declare one unique variable identifier for every design row")
        if member_positions is not None and set(member_positions) != set(backends):
            raise ValueError("Declare member_positions for every backend")
        self.backends, self.combine = dict(backends), combine
        self._positions = {}
        for name, backend in self.backends.items():
            positions = getattr(backend, "positions", backend.metadata.get("design_positions"))
            declared = None if member_positions is None else member_positions[name]
            if positions is not None and declared is not None and not np.array_equal(positions, declared):
                raise ValueError(f"{name} member_positions differs from native design row order")
            indices = positions if positions is not None else declared
            if indices is not None:
                indices = np.asarray(indices)
                if (indices.shape != (self.design_length,) or indices.dtype.kind not in "iu"
                        or (indices < 0).any() or len(np.unique(indices)) != self.design_length):
                    raise ValueError(f"{name} requires unique nonnegative native indices for each design row")
            self._positions[name] = indices.tolist() if indices is not None else None
        observed = {tuple(pos) for pos in self._positions.values() if pos is not None}
        if member_positions is None and len(observed) > 1:
            raise ValueError("Native design row orders differ; supply the intended member_positions alignment")
        self.metadata = {"backend": "composite", "variable_order": list(variable_order),
                         "member_positions": self._positions,
                         "members": {k: b.metadata for k, b in backends.items()},
                         "key_policy": "same key in each condition"}
        if all(callable(getattr(b, "probe_forward", None)) for b in backends.values()):
            self.probe_forward = self._probe_forward

    def sequence_from_logits(self, logits, controls=None):
        """Return the shared hard identities of design rows only."""
        shared = None
        for name, backend in self.backends.items():
            decoder = getattr(backend, "sequence_from_logits", None)
            if decoder is None:
                sequence = decode(logits)
            else:
                sequence = decoder(logits, controls)
                positions = self._positions[name]
                if not isinstance(sequence, str):
                    raise ValueError(f"{name} must decode a sequence string")
                if positions is not None:
                    if max(positions) >= len(sequence):
                        raise ValueError(f"{name} native decoder length differs from its residue map")
                    sequence = "".join(sequence[i] for i in positions)
            if len(sequence) != self.design_length or shared is not None and sequence != shared:
                raise ValueError(f"{name} native design-row identities differ from the shared sequence")
            shared = sequence
        return shared

    def evaluate(self, logits, key, controls=None, *, gradient=True, prediction=False):
        if prediction:
            raise ValueError("Retrieve condition predictions from the named member backends")
        return self._evaluate(logits, key, controls, gradient=gradient, probe=False)

    def _probe_forward(self, logits, key, controls=None):
        return self._evaluate(logits, key, controls, gradient=False, probe=True)

    def _evaluate(self, logits, key, controls, *, gradient, probe):
        import jax
        import jax.numpy as jnp
        from .common import inputs, evaluation
        logits, key, controls = inputs(logits, key, controls, self.design_length)
        self.sequence_from_logits(logits, controls)
        results = {}
        for name, backend in self.backends.items():
            if probe:
                results[name] = backend.probe_forward(logits, key, dict(controls))
            else:
                results[name] = backend.evaluate(logits, key, dict(controls), gradient=gradient)
        values = {name: jnp.asarray(result.loss) for name, result in results.items()}

        def aggregate(x):
            return objective_result(self.combine(x, controls))

        if gradient:
            (loss, metrics), coefficients = jax.value_and_grad(aggregate, has_aux=True)(values)
            grad = np.zeros_like(logits, dtype=np.float64)
            for name, result in results.items():
                if result.gradient is None or result.gradient.shape != logits.shape:
                    raise ValueError(f"{name} returned an invalid design-row gradient")
                grad += float(coefficients[name]) * np.asarray(result.gradient, dtype=np.float64)
        else:
            loss, metrics = aggregate(values)
            grad = None
        member_metrics = {f"condition/{k}": v.loss for k, v in results.items()}
        for name, result in results.items():
            nested = {f"condition/{name}/{k}": v for k, v in result.metrics.items()}
            if member_metrics.keys() & nested.keys():
                raise ValueError("Member metric name collides with a condition name")
            member_metrics.update(nested)
        if metrics.keys() & member_metrics.keys():
            raise ValueError("Combined metric name collides with a member metric")
        return evaluation(loss, {**member_metrics, **metrics}, grad)
