import json
from types import SimpleNamespace
import pytest
from hallucination_lab import diagnostics
from hallucination_lab.cli import main


def test_metadata_default_does_not_initialize_runtime(monkeypatch):
    def no_import(*args):
        raise AssertionError("Default doctor must not initialize models/JAX")
    monkeypatch.setattr(diagnostics.importlib, "import_module", no_import)
    report = diagnostics.preflight()
    assert report["ok"] and "devices" not in report
    assert report["checks"] == []


def test_missing_gpu_runtime_and_weights_fail_explicitly(monkeypatch, tmp_path):
    def importer(name):
        if name == "jax":
            return SimpleNamespace(__file__="jax.py", local_devices=lambda: [
                SimpleNamespace(platform="cpu", device_kind="cpu")])
        raise ImportError("Not installed")
    monkeypatch.setattr(diagnostics.importlib, "import_module", importer)
    report = diagnostics.preflight(runtime="af2", require_gpu=True, weights=tmp_path)
    checks = {item["name"]: item for item in report["checks"]}
    assert not report["ok"]
    assert not checks["gpu"]["ok"] and not checks["weight_files"]["ok"]
    assert not checks["import:colabdesign"]["ok"]


def test_profile_is_exact_and_inert(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostics, "package_versions", lambda names: {name: "1.0" for name in names})
    monkeypatch.setattr(diagnostics.platform, "python_version", lambda: "3.12.11")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"python": "3.12", "packages": {"jax": "1.0"}}))
    assert diagnostics.preflight(profile=path)["ok"]
    path.write_text(json.dumps({"python": "3.1", "packages": {"jax": "1.0"}}))
    assert not diagnostics.preflight(profile=path)["ok"]
    path.write_text(json.dumps({"python": "3.12", "packages": {"jax": "2.0"}}))
    assert not diagnostics.preflight(profile=path)["ok"]


def test_cli_diagnostic_failure_is_nonzero(capsys):
    with pytest.raises(SystemExit) as error:
        main(["doctor", "--require-gpu"])
    assert error.value.code == 2
    assert "require --runtime" in capsys.readouterr().err


def test_weight_layouts_and_device_error_are_reported_separately(monkeypatch, tmp_path):
    def failed_devices():
        raise RuntimeError("no device")
    monkeypatch.setattr(diagnostics.importlib, "import_module",
                        lambda name: SimpleNamespace(local_devices=failed_devices))
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "params_model_1.npz").write_bytes(b"1")
    (tmp_path / "af3.bin.zst.00").write_bytes(b"1")
    for runtime in ("af2", "af3"):
        report = diagnostics.preflight(runtime=runtime, weights=tmp_path)
        assert not report["ok"]
        checks = {item["name"]: item["ok"] for item in report["checks"]}
        assert checks["weight_files"] and checks["import:jax"] and not checks["devices"]
        assert len(checks) == len(report["checks"])
