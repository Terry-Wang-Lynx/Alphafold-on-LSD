"""Explicit environment preflight; this does not run or certify a model."""
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
from ._version import __version__


def package_versions(names):
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def preflight(*, runtime=None, require_gpu=False, weights=None, profile=None):
    """Inspect only requested runtime; never download weights or install code.

    A profile declares Python major.minor and exact distribution versions in
    ``packages``. It is data, never an import list or executable installer.
    GPU discovery initializes JAX and may allocate runtime resources.
    """
    if runtime not in (None, "af2", "af3"):
        raise ValueError("runtime must be af2 or af3")
    if (require_gpu or weights is not None) and runtime is None:
        raise ValueError("--require-gpu and --weights require --runtime")
    packages = package_versions(("hallucination-lab", "numpy", "jax", "jaxlib",
                                 "dm-haiku", "colabdesign", "alphafold3"))
    report = {"scope": "environment_preflight_only_no_model_execution", "code_version": __version__,
              "python": platform.python_version(), "executable": sys.executable,
              "packages": packages, "checks": [], "runtime": runtime}

    def check(name, ok, **detail):
        report["checks"].append({"name": name, "ok": bool(ok), **detail})

    if profile is not None:
        declared = json.loads(Path(profile).read_text())
        if not isinstance(declared, dict):
            raise ValueError("Profile must be a JSON object")
        expected_python = declared.get("python")
        expected_packages = declared.get("packages")
        if (not isinstance(expected_python, str) or not expected_python
                or not isinstance(expected_packages, dict) or not expected_packages):
            raise ValueError("Profile requires python and a nonempty exact packages mapping")
        if any(not isinstance(k, str) or not isinstance(v, str) or not v
               for k, v in expected_packages.items()):
            raise ValueError("Profile packages must map names to exact version strings")
        report["profile"] = {"name": declared.get("name"), "python": expected_python,
                             "packages": expected_packages}
        actual_python = platform.python_version()
        check("profile_python", actual_python.split(".")[:len(expected_python.split("."))]
              == expected_python.split("."), actual=actual_python, expected=expected_python)
        actual_packages = package_versions(expected_packages)
        for name, expected in expected_packages.items():
            check(f"package:{name}", actual_packages[name] == expected,
                  actual=actual_packages[name], expected=expected)
    if runtime:
        module_name = {"af2": "colabdesign", "af3": "alphafold3"}[runtime]
        for name in ("jax", module_name):
            try:
                module = importlib.import_module(name)
                check(f"import:{name}", True, path=getattr(module, "__file__", None))
                if name == "jax":
                    devices = [{"platform": d.platform, "kind": d.device_kind}
                               for d in module.local_devices()]
                    report["devices"] = devices
                    if require_gpu:
                        check("gpu", any(d["platform"] == "gpu" for d in devices),
                              devices=devices)
            except Exception as error:
                check(f"import:{name}", False, error=f"{type(error).__name__}: {error}")
    if weights is not None:
        path = Path(weights)
        if runtime == "af2":
            files = sorted(path.glob("params_*.npz")) if path.is_dir() else []
        else:
            files = [path / "af3.bin"] if path.is_dir() else [path]
        valid = [p for p in files if p.is_file() and p.stat().st_size > 0]
        check("weight_files", bool(valid), files=[{"name": p.name, "bytes": p.stat().st_size}
                                                 for p in valid],
              interpretation="Presence only; completeness, hash and model loading not checked")
    report["ok"] = all(check["ok"] for check in report["checks"])
    return report
