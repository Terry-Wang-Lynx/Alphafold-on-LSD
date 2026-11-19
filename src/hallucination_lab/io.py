"""Portable exports; explicit selections keep model-sized arrays out of logs."""
from pathlib import Path
import numpy as np
from .checkpoint import save
from .types import Prediction


def save_prediction(path, prediction: Prediction, *, include_native=False):
    data = {name: np.asarray(value) for name, value in prediction._asdict().items() if name != "native"}
    if include_native:
        def arrays(value):
            if isinstance(value, dict):
                return {key: arrays(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [arrays(item) for item in value]
            return None if value is None else np.asarray(value)
        data["native"] = arrays(prediction.native)
    save(path, data)


def write_fasta(path, sequences):
    """Write named sequences explicitly supplied by the caller."""
    from .sequence import ALPHABET
    rows = []
    for name, sequence in sequences.items():
        if not name or "\r" in name or "\n" in name or not sequence or any(a not in ALPHABET for a in sequence):
            raise ValueError("FASTA names must be single-line and sequences standard uppercase amino acids")
        rows.extend([f">{name}", sequence])
    Path(path).write_text("\n".join(rows) + "\n")
