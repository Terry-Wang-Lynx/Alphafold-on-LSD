"""Paired objective measurements under explicitly fixed evaluation conditions.

These measure model objectives, not biological function. Conditions should be
chosen before seeing the results; report every start and every failed pair.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np


def _json(value):
    def numpy_value(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize assessment value {type(item).__name__}")
    return json.dumps(value, sort_keys=True, allow_nan=False, default=numpy_value)


def _digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def paired_evaluate(backend, before_logits, after_logits, *, conditions, output=None):
    """Evaluate both endpoints with identical keys, controls, and backend.

    Each condition is {name: str, key: [uint32, uint32], controls: dict}.
    Ordinary backend exceptions are recorded, not omitted. Always inspect
    ``all_valid`` before using the report. Unexpected interrupts propagate.
    No gradients are requested; AF2's consistent score mode may compute them
    internally. Endpoints are supplied explicitly, never selected from a run.
    ``output`` writes a JSON report exclusively and refuses to overwrite.
    """
    endpoints = {}
    for name, value in (("before", before_logits), ("after", after_logits)):
        array = np.array(value, dtype=np.float32, copy=True)
        if array.shape != (backend.design_length, 20) or not np.isfinite(array).all():
            raise ValueError(f"{name}_logits must be finite [design_length,20]")
        endpoints[name] = array
    # JSON copying freezes the protocol and rejects non-serializable controls.
    protocol = json.loads(_json(list(conditions)))
    if not protocol:
        raise ValueError("At least one evaluation condition is required")
    names = set()
    for condition in protocol:
        if not isinstance(condition, dict) or set(condition) != {"name", "key", "controls"}:
            raise ValueError("Each condition requires exactly name, key, and controls")
        name, key, controls = (condition[k] for k in ("name", "key", "controls"))
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Condition names must be nonempty and unique")
        names.add(name)
        if (not isinstance(key, list) or len(key) != 2
                or any(type(k) is not int or not 0 <= k < 2**32 for k in key)):
            raise ValueError("Condition key must contain two uint32 integers")
        if not isinstance(controls, dict):
            raise ValueError("Condition controls must be a dictionary")
    destination = Path(output) if output else None
    if destination is not None and destination.exists():
        raise FileExistsError(f"Refusing to overwrite assessment: {destination}")
    metadata = json.loads(_json(backend.metadata))
    report = {
        "format": 1, "interpretation": "model_objective_comparison_not_biological_validation",
        "backend": metadata, "conditions": protocol,
        "protocol_sha256": hashlib.sha256(_json({"backend": metadata,
                                                  "conditions": protocol}).encode()).hexdigest(),
        "endpoint_shape": list(endpoints["before"].shape), "endpoint_dtype": "float32",
        "endpoint_sha256": {name: _digest(value) for name, value in endpoints.items()},
        "pairs": [],
    }
    for condition in protocol:
        pair = {"condition": condition["name"]}
        for name, values in endpoints.items():
            try:
                result = backend.evaluate(values.copy(), np.array(condition["key"], np.uint32),
                                          deepcopy(condition["controls"]), gradient=False)
                loss = float(result.loss)
                metrics = {k: float(v) for k, v in result.metrics.items()}
                if not np.isfinite(loss) or not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError("Nonfinite loss or metric")
                pair[name] = {"ok": True, "loss": loss, "metrics": metrics}
            except Exception as error:
                pair[name] = {"ok": False, "error_type": type(error).__name__,
                              "error": str(error)}
                from .numerics import NumericalAuditError
                if isinstance(error, NumericalAuditError):
                    pair[name]["audit"] = json.loads(_json(error.report))
        pair["ok"] = pair["before"]["ok"] and pair["after"]["ok"]
        if pair["ok"]:
            before, after = pair["before"], pair["after"]
            # Missing metrics are a contract mismatch, not a zero-valued metric.
            if before["metrics"].keys() != after["metrics"].keys():
                pair.update(ok=False, error="Endpoint metric names differ")
            else:
                delta = {"loss": after["loss"] - before["loss"], "metrics": {
                    k: after["metrics"][k] - before["metrics"][k] for k in before["metrics"]}}
                if (not np.isfinite(delta["loss"])
                        or not all(np.isfinite(v) for v in delta["metrics"].values())):
                    pair.update(ok=False, error="Nonfinite endpoint difference")
                else:
                    pair["delta"] = delta
        report["pairs"].append(pair)
    report["requested_pairs"] = len(protocol)
    report["complete_pairs"] = sum(pair["ok"] for pair in report["pairs"])
    report["improved_pairs"] = sum(pair["ok"] and pair["delta"]["loss"] < 0
                                   for pair in report["pairs"])
    report["all_valid"] = report["complete_pairs"] == report["requested_pairs"]
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as file:
            file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
