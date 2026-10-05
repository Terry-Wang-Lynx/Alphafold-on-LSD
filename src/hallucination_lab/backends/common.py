"""Backend boundary validation and host conversion."""
import numpy as np
from ..sequence import validate_controls
from ..types import Evaluation


def inputs(logits, key, controls, length):
    logits = np.asarray(logits, np.float32)
    key = np.asarray(key)
    if logits.shape != (length, 20) or not np.isfinite(logits).all():
        raise ValueError("logits must be finite [design_length,20]")
    if key.shape != (2,) or key.dtype != np.uint32:
        raise ValueError("key must be a uint32 array of shape (2,)")
    return logits, key, validate_controls(controls)


def evaluation(loss, metrics, gradient=None, prediction=None):
    import jax
    loss, metrics, gradient, prediction = jax.device_get((loss, metrics, gradient, prediction))
    result = Evaluation(float(loss), {k: float(v) for k, v in metrics.items()},
                        None if gradient is None else np.asarray(gradient, np.float32), prediction)
    if not np.isfinite(result.loss) or not all(np.isfinite(v) for v in result.metrics.values()):
        raise FloatingPointError("Nonfinite objective: check masks, bins, and model outputs")
    if result.gradient is not None and not np.isfinite(result.gradient).all():
        raise FloatingPointError("Nonfinite sequence gradient")
    return result


def design_indices(value, length):
    """Reject lossy or boolean position conversions before model preparation."""
    if value is not None and any(isinstance(item, (bool, np.bool_))
                                 for item in np.asarray(value, dtype=object).flat):
        raise ValueError("design_positions must be unique valid zero-based integer positions")
    positions = np.arange(length) if value is None else np.asarray(value)
    if (positions.ndim != 1 or not positions.size or positions.dtype.kind not in "iu"
            or (positions < 0).any() or (positions >= length).any()
            or (positions > np.iinfo(np.int32).max).any()
            or len(np.unique(positions)) != len(positions)):
        raise ValueError("design_positions must be unique valid zero-based integer positions")
    return positions.astype(np.int32)


def numerical_policy(precision):
    """Explicit matmul policy; model dtypes and non-matmul kernels remain separate."""
    import hashlib
    import os
    from importlib.metadata import version
    import jax
    if precision not in ("default", "high", "highest"):
        raise ValueError("matmul_precision must be default, high, or highest")
    return {"matmul_precision": precision, "jaxlib": version("jaxlib"),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "xla_flags_sha256": hashlib.sha256(os.environ.get("XLA_FLAGS", "").encode()).hexdigest(),
            "devices": [{"platform": d.platform, "kind": d.device_kind}
                        for d in jax.local_devices()]}


def with_precision(function):
    """Trace and execute under a local policy without changing caller JAX globals."""
    from functools import wraps
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        import jax
        with jax.default_matmul_precision(self._matmul_precision):
            return function(self, *args, **kwargs)
    return wrapped
