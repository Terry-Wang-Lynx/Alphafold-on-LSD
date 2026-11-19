"""Atomic, pickle-free array + JSON checkpoints for inspectable research state."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def encode(tree):
    """Validate and encode supported state without writing a checkpoint."""
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(arrays)}"
            if value.dtype.hasobject:
                raise TypeError("Object arrays are not checkpointable")
            arrays[key] = value
            return {"__array__": key}
        if isinstance(value, dict):
            if "__array__" in value:
                raise ValueError("__array__ is reserved in checkpoint dictionaries")
            if not all(isinstance(k, str) for k in value):
                raise TypeError("Checkpoint dictionary keys must be strings")
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [pack(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    arrays["manifest"] = np.asarray(json.dumps(pack(tree), allow_nan=False))
    return arrays


def save(path, tree):
    arrays = encode(tree)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
            temporary = file.name
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load(path):
    from zipfile import BadZipFile
    try:
        return _load(path)
    except (BadZipFile, EOFError) as error:
        raise ValueError(f"Invalid checkpoint {path}: damaged NPZ archive") from error


def _load(path):
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"Invalid checkpoint {path}: expected an NPZ archive")
    with archive:
        if "manifest" not in archive.files:
            raise ValueError(f"Invalid checkpoint {path}: missing manifest")
        manifest = archive["manifest"]
        if manifest.ndim != 0 or manifest.dtype.kind not in "US":
            raise ValueError(f"Invalid checkpoint {path}: manifest must be a JSON string")

        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    key = value["__array__"]
                    if not isinstance(key, str) or key == "manifest" or key not in archive.files:
                        raise ValueError(f"Invalid checkpoint {path}: missing or invalid array reference {key!r}")
                    return archive[key].copy()
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value
        return unpack(json.loads(manifest.item()))


def atomic_text(path, text):
    """Replace a text file only after its full replacement has been written."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = file.name
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
