"""Small shared contracts; no model runtime dependency."""
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol
import numpy as np


class Prediction(NamedTuple):
    """Native bin boundaries in Å; logits [N,N,B]; sequence [D,20].

    Padding remains in the tensors. token_mask excludes it. Native fields
    differ by backend. Sequence rows refer to the declared design positions.
    """
    logits: Any
    bin_edges: Any
    token_mask: Any
    residue_index: Any
    asym_id: Any
    sequence: Any
    native: dict


@dataclass
class Evaluation:
    loss: float
    metrics: dict[str, float]
    gradient: np.ndarray | None = None
    prediction: Prediction | None = None


class Backend(Protocol):
    design_length: int
    metadata: dict

    def evaluate(self, logits, key, controls=None, *, gradient=True,
                 prediction=False) -> Evaluation: ...


def objective_result(result):
    """Normalize a caller objective: scalar or (scalar, scalar-metric dict)."""
    if isinstance(result, tuple):
        if len(result) != 2 or not isinstance(result[1], dict):
            raise TypeError("Objective must return scalar or (scalar, metrics dict)")
        loss, metrics = result
    else:
        loss, metrics = result, {}
    if getattr(loss, "ndim", 0) != 0:
        raise ValueError("Objective loss must be scalar")
    if any(getattr(v, "ndim", 0) != 0 for v in metrics.values()):
        raise ValueError("Objective metrics must be scalar")
    return loss, metrics
