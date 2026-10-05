"""Reproducibility helpers that avoid embedding machine-specific absolute paths."""
import hashlib
import json
from pathlib import Path
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_fingerprint(path):
    """Hash Python source contents with stable paths, including uncommitted edits.

    The source tree must stay unchanged while it is imported and used. This does
    not capture in-memory monkeypatches or caller-owned closures; record those
    explicitly in the experiment identity.
    """
    root = Path(path).resolve()
    files = {file.relative_to(root).as_posix(): sha256_file(file)
             for file in sorted(root.rglob("*.py"))
             if not any(part.startswith(".") or part == "__pycache__"
                        for part in file.relative_to(root).parts)}
    if not files:
        raise ValueError("Cannot fingerprint an empty Python source tree")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def source_revision(path):
    """Git revision only for a tracked source path, otherwise explicitly None.

    An installed runtime inside a project's .venv must not inherit the enclosing
    project's unrelated revision. Archive/wheel installations retain file hashes.
    """
    source = Path(path).resolve()
    if not source.exists():
        return None
    directory = source.parent if source.is_file() else source
    try:
        result = subprocess.run(["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True, timeout=5, check=True)
        root = Path(result.stdout.strip()).resolve()
        relative = str(source.relative_to(root))
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch",
                                  "--", relative], capture_output=True, text=True,
                                 timeout=5, check=True)
        if not tracked.stdout.strip():
            return None
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def experiment_id(*, files=(), settings=None, label="experiment"):
    """Content-derived ID for custom code, input files, and explicit settings.

    Supply every semantically relevant file, including objective/step modules
    and input structures. Function source alone cannot capture closure contents.
    """
    content = {"files": [sha256_file(path) for path in files], "settings": settings or {}}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return f"{label}:{digest}"
