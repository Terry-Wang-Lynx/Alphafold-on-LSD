"""A public allowlist must reject accidental additional source files."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile


def test_unlisted_file_is_rejected_and_missing_file_is_reported(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/audit_release.py"
    spec = importlib.util.spec_from_file_location("release_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "release-files.json").write_text(json.dumps({
        "files": ["release-files.json", "expected.txt"]}))
    (tmp_path / "expected.txt").write_text("public")
    assert module.audit(tmp_path) == []
    (tmp_path / "unlisted.txt").write_text("unlisted")
    issues = module.audit(tmp_path)
    assert "Undeclared public file: unlisted.txt" in issues
    (tmp_path / "expected.txt").unlink()
    assert "Missing public file: expected.txt" in module.audit(tmp_path)


def test_source_archive_rejects_extra_and_modified_source(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/audit_release.py"
    spec = importlib.util.spec_from_file_location("release_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    root.mkdir()
    (root / "release-files.json").write_text(json.dumps({"files": ["public.txt"]}))
    (root / "public.txt").write_text("public")

    def bundle(name, files):
        archive = tmp_path / name
        with tarfile.open(archive, "w:gz") as tar:
            for relative, content in files.items():
                data = content.encode()
                member = tarfile.TarInfo("package/" + relative)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        return archive

    valid = bundle("good.tar.gz", {"public.txt": "public", "PKG-INFO": "Name: package"})
    assert module.audit_sdist(valid, root) == []
    invalid = bundle("bad.tar.gz", {"public.txt": "changed",
                                    "src/hallucination_lab.egg-info/private.txt": "private"})
    issues = module.audit_sdist(invalid, root)
    assert "Archive source differs: public.txt" in issues
    assert "Undeclared archive member: src/hallucination_lab.egg-info/private.txt" in issues
