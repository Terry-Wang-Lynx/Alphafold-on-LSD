"""Check the explicit public file list and basic source distribution hygiene."""
import argparse
from pathlib import Path, PurePosixPath
import json
import re
import sys
import tarfile


def audit(root):
    declared = set(json.loads((root / "release-files.json").read_text())["files"])
    excluded = {".git", "build", "dist", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "runs"}
    observed = set()
    issues = []
    patterns = [r"/Users/[^/]+/", r"/home/[^/]+/", r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                r"gh[pousr]_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9]{32,}"]
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(x in excluded or x.endswith(".egg-info") for x in relative.parts):
            continue
        if path.is_symlink():
            issues.append(f"Symlink: {relative}")
            continue
        if not path.is_file() or str(relative) == "PKG-INFO":
            continue
        observed.add(relative.as_posix())
        if path.suffix in {".npz", ".npy", ".pkl", ".bin", ".cif", ".pdb", ".pem", ".key"}:
            issues.append(f"Unexpected asset: {relative}")
        try:
            text = path.read_text()
        except UnicodeError:
            issues.append(f"Non-text source file: {relative}")
            continue
        if relative.as_posix() != "scripts/audit_release.py":
            if any(re.search(pattern, text) for pattern in patterns):
                issues.append(f"Private-path/credential pattern: {relative}")
    issues.extend(f"Undeclared public file: {x}" for x in sorted(observed - declared))
    issues.extend(f"Missing public file: {x}" for x in sorted(declared - observed))
    return issues


def audit_sdist(path, root):
    """Check archive members and source bytes without extracting the archive."""
    declared = set(json.loads((root / "release-files.json").read_text())["files"])
    generated = {"PKG-INFO", "setup.cfg"} | {
        "src/hallucination_lab.egg-info/" + name for name in
        ("PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
         "requires.txt", "top_level.txt")}
    issues, observed = [], set()
    with tarfile.open(path) as archive:
        roots = set()
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or member.name.startswith("/") or ".." in parts:
                issues.append(f"Unsafe archive path: {member.name}")
                continue
            roots.add(parts[0])
            if member.isdir():
                continue
            if not member.isfile() or len(parts) < 2:
                issues.append(f"Non-regular archive member: {member.name}")
                continue
            relative = "/".join(parts[1:])
            if relative in observed:
                issues.append(f"Duplicate archive member: {relative}")
            observed.add(relative)
            if relative not in declared | generated:
                issues.append(f"Undeclared archive member: {relative}")
            elif relative in declared:
                if archive.extractfile(member).read() != (root / relative).read_bytes():
                    issues.append(f"Archive source differs: {relative}")
        if len(roots) != 1:
            issues.append("Archive must have one top-level directory")
    issues.extend(f"Missing archive source: {name}" for name in sorted(declared - observed))
    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, nargs="*", default=[],
                        help="Also check source archives against the declared source and build metadata")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    problems = audit(root)
    for path in args.sdist:
        problems.extend(audit_sdist(path, root))
    print(json.dumps({"passed": not problems, "issues": problems,
                      "scope": "File allowlist and limited pattern checks; not a complete security audit"}, indent=2))
    sys.exit(bool(problems))
