"""Explicit numerical gates, separate from optimization or biological success.

Audits evaluate fixed inputs and keys. They never modify gradients, retry with
new settings, or interpret a straight-through estimator as a true hard-forward
finite difference. Passing is evidence only for the sampled points and checks.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


def _json_array(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Cannot serialize audit value {type(value).__name__}')


@dataclass(frozen=True)
class Tolerances:
    loss_rtol: float = 1e-4
    loss_atol: float = 1e-4
    gradient_cosine_min: float = .999
    gradient_relative_l2_max: float = 1e-3
    normalized_update_rms_max: float = 1e-4
    learning_rate: float = .1

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(x) for x in values.values()):
            raise ValueError('Tolerances must be finite')
        if not -1 <= self.gradient_cosine_min <= 1:
            raise ValueError('gradient_cosine_min must lie in [-1,1]')
        if any(v < 0 for k, v in values.items() if k != 'gradient_cosine_min'):
            raise ValueError('Error tolerances and learning_rate must be nonnegative')


class NumericalAuditError(FloatingPointError):
    """A fixed-input numerical gate failed; this is not a loss-feasibility result."""
    def __init__(self, report, evaluation=None, input_logits=None):
        self.report = report
        self.evaluation = evaluation
        self.input_logits = None if input_logits is None else np.asarray(input_logits).copy()
        super().__init__('Numerical audit failed; inspect .report before interpreting optimization')


def compare_evaluations(reference, observed, *, tolerances=None, gradients=True):
    """Compare values and raw gradients, retaining direction AND magnitude tests.

    The normalized update uses sqrt(number of nonzero gradient rows) / L2 norm,
    matching the declared sequence-SGD convention. It is not a substitute for
    raw-gradient checks. Neither reference nor observed is assumed ground truth.
    """
    t = tolerances or Tolerances()
    left, right = float(reference.loss), float(observed.loss)
    finite = np.isfinite(left) and np.isfinite(right)
    result = {'loss_reference': left if np.isfinite(left) else None,
              'loss_observed': right if np.isfinite(right) else None,
              'loss_absolute_difference': abs(left-right) if finite else None,
              'checks': {'finite_loss': bool(finite),
                         'loss': bool(finite and np.isclose(left, right, rtol=t.loss_rtol,
                                                           atol=t.loss_atol))}}
    if gradients:
        a, b = reference.gradient, observed.gradient
        valid = (a is not None and b is not None and np.shape(a) == np.shape(b)
                 and np.ndim(a) == 2 and np.size(a) > 0
                 and np.isfinite(a).all() and np.isfinite(b).all())
        result['checks']['finite_matching_gradients'] = bool(valid)
        if valid:
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            an, bn = np.linalg.norm(a), np.linalg.norm(b)
            both_zero = an == bn == 0
            cosine = 1. if both_zero else (float(np.vdot(a, b)/(an*bn)) if an*bn else None)
            relative = 0. if both_zero else (float(np.linalg.norm(a-b)/an) if an else None)
            def update(g, norm):
                return t.learning_rate * g * np.sqrt(np.count_nonzero(np.square(g).sum(-1))) / (norm+1e-7)
            rms = float(np.sqrt(np.square(update(a, an)-update(b, bn)).mean()))
            result.update(gradient_reference_norm=float(an), gradient_observed_norm=float(bn),
                          gradient_cosine=cosine, gradient_relative_l2=relative,
                          normalized_update_rms=rms)
            result['checks'].update(
                gradient_cosine=cosine is not None and cosine >= t.gradient_cosine_min,
                gradient_relative_l2=relative is not None and relative <= t.gradient_relative_l2_max,
                normalized_update_rms=rms <= t.normalized_update_rms_max)
    result['passed'] = all(result['checks'].values())
    return result


def audit_backend(backend, logits, key, controls=None, *, reference=None,
                  tolerances=None, output=None, strict=True):
    """Gate a point before using its optimization result in a loss study.

    Checks repeated gradient evaluations, score vs gradient graph, and (where
    available) an independent native forward graph. An optional reference backend
    adds a paired comparison; it must implement the SAME intended objective and
    gradient semantics. Failing raises AFTER writing the JSON report. Forward/AD
    disagreement is numerical inconsistency, not proof that AD is wrong.
    Cost: two gradient calls plus score and optional probe/reference calls.
    """
    t = tolerances or Tolerances()
    x, k, c = np.asarray(logits).copy(), np.asarray(key).copy(), deepcopy(controls or {})
    if output is not None:
        output = Path(output)
        raw_path = output.with_suffix('.npz')
        if raw_path == output:
            raise ValueError('Audit report path must not use the reserved .npz suffix')
        if output.exists() or raw_path.exists():
            raise FileExistsError('Audit outputs already exist; use a new evidence path')
        output.parent.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Fixed-input numerical consistency; not a universal correctness certificate',
              'tolerances': asdict(t), 'controls': c, 'key': k.tolist(),
              'input_shape': list(x.shape), 'input_dtype': str(x.dtype),
              'input_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
              'backend_metadata': backend.metadata, 'checks': {}}
    arrays = {'logits': x, 'key': k}
    def evaluate(which, gradient):
        return which.evaluate(x.copy(), k.copy(), deepcopy(c), gradient=gradient)
    try:
        first = evaluate(backend, True)
        arrays['gradient'] = first.gradient
        report['baseline_metrics'] = dict(first.metrics)
        repeated = evaluate(backend, True)
        arrays['repeat_gradient'] = repeated.gradient
        report['checks']['repeat'] = compare_evaluations(first, repeated, tolerances=t)
        report['checks']['score'] = compare_evaluations(first, evaluate(backend, False),
                                                       tolerances=t, gradients=False)
        if hasattr(backend, 'probe_forward'):
            report['checks']['independent_forward'] = compare_evaluations(
                first, backend.probe_forward(x.copy(), k.copy(), deepcopy(c)),
                tolerances=t, gradients=False)
        if reference is not None:
            report['reference_metadata'] = reference.metadata
            reference_result = evaluate(reference, True)
            arrays['reference_gradient'] = reference_result.gradient
            report['checks']['reference'] = compare_evaluations(reference_result, first, tolerances=t)
    except NumericalAuditError as exc:
        report['numerical_exception'] = str(exc)
        report['backend_guard'] = exc.report
        if exc.evaluation is not None:
            arrays['rejected_gradient'] = exc.evaluation.gradient
    except FloatingPointError as exc:
        report['numerical_exception'] = str(exc)
    report['passed'] = bool(report['checks']) and 'numerical_exception' not in report and all(
        check['passed'] for check in report['checks'].values())
    if output is not None:
        with raw_path.open('xb') as handle:
            np.savez_compressed(handle, **{k: v for k, v in arrays.items() if v is not None})
        report['raw_arrays'] = str(raw_path)
        with output.open('x') as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False, default=_json_array) + '\n')
    if strict and not report['passed']:
        raise NumericalAuditError(report)
    return report
