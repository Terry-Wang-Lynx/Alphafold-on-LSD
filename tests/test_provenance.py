from pathlib import Path
import shutil
import subprocess
import pytest
from hallucination_lab.provenance import source_revision, source_tree_fingerprint


@pytest.fixture
def checkout(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    (tmp_path / "source.py").write_text("value = 1\n")
    git("add", "source.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "fixture")
    return tmp_path, git("rev-parse", "HEAD")


def test_tracked_source_reports_its_revision(checkout):
    root, revision = checkout
    assert source_revision(root / "source.py") == revision


def test_installed_package_inside_checkout_does_not_inherit_parent_revision(checkout):
    root, _ = checkout
    module = root / ".venv" / "site-packages" / "upstream" / "model.py"
    module.parent.mkdir(parents=True)
    module.write_text("value = 2\n")
    assert source_revision(module) is None
    assert source_revision(Path(root) / "missing.py") is None


def test_source_fingerprint_tracks_nested_edits_independent_of_install_path(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    (first / "model").mkdir(parents=True)
    (first / "model" / "geometry.py").write_text("precision = 32\n")
    (first / "__init__.py").write_text("")
    shutil.copytree(first, second)
    original = source_tree_fingerprint(first)
    assert source_tree_fingerprint(second) == original
    (second / "model" / "ignored.pyc").write_bytes(b"cache")
    assert source_tree_fingerprint(second) == original
    (second / "model" / "geometry.py").write_text("precision = 64\n")
    changed = source_tree_fingerprint(second)
    assert changed["sha256"] != original["sha256"]
    assert changed["files"]["model/geometry.py"] != original["files"]["model/geometry.py"]
    assert changed["files"]["__init__.py"] == original["files"]["__init__.py"]
    with pytest.raises(ValueError, match="empty"):
        source_tree_fingerprint(tmp_path / "absent")


def test_checkpoint_rejects_source_edit_without_version_bump(quadratic, tmp_path, monkeypatch):
    from hallucination_lab import Run, Stage, initial_logits
    from hallucination_lab import provenance
    source = tmp_path / "source"
    source.mkdir()
    module = source / "engine.py"
    module.write_text("moment_dtype = 'float32'\n")
    monkeypatch.setattr(provenance, "source_tree_fingerprint",
                        lambda root: source_tree_fingerprint(source))

    def run():
        return Run(quadratic, initial_logits(2), [Stage("test", 2)], problem_id="unchanged")

    original = run()
    original.run(max_steps=1)
    checkpoint = tmp_path / "old.npz"
    original.save(checkpoint)
    unchanged = run().restore(checkpoint)
    assert unchanged.state.step == 1
    module.write_text("moment_dtype = 'float64'\n")
    modified = run()
    assert modified.manifest["tool_version"] == original.manifest["tool_version"]
    with pytest.raises(ValueError, match="code/runtime"):
        modified.restore(checkpoint)
    assert modified.state.step == 0
