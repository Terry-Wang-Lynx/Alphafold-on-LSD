import json
import subprocess
import sys
import numpy as np
import pytest
from hallucination_lab.sequence import initial_logits, decode, relax
from hallucination_lab.checkpoint import save, load


def test_import_does_not_load_jax():
    subprocess.run([sys.executable, "-c", "import hallucination_lab, sys; assert 'jax' not in sys.modules"], check=True)


def test_alphabet_roundtrip():
    sequence = "ARNDCQEGHILKMFPSTWYV"
    assert decode(initial_logits(sequence=sequence)) == sequence


def test_straight_through_has_hard_values_and_gradient():
    import jax
    import jax.numpy as jnp
    logits = jnp.asarray(initial_logits(3))
    hard = relax(logits, {"hard": 1.0})
    np.testing.assert_allclose(hard.sum(-1), 1)
    assert np.all((np.asarray(hard) == 0) | (np.asarray(hard) == 1))
    grad = jax.grad(lambda x: relax(x, {"hard": 1.0})[:, 0].sum())(logits)
    assert np.linalg.norm(grad) > 0


def test_checkpoint_roundtrip_without_pickle(tmp_path):
    path = tmp_path / "state.npz"
    tree = {"data": np.arange(5), "nested": {"value": None, "list": [1, "x"]}}
    save(path, tree)
    restored = load(path)
    np.testing.assert_array_equal(restored["data"], tree["data"])
    assert restored["nested"] == tree["nested"]
    with pytest.raises(TypeError):
        save(path, {"unsafe": np.array([object()], dtype=object)})
    np.testing.assert_array_equal(load(path)["data"], tree["data"])


@pytest.mark.parametrize("case", ["missing_manifest", "missing_array", "bad_manifest", "npy", "truncated_zip", "not_run"])
def test_cli_rejects_malformed_checkpoint_without_traceback(tmp_path, case):
    path = tmp_path / "invalid.npz"
    if case == "missing_manifest":
        np.savez(path, a=np.zeros(1))
    elif case == "missing_array":
        np.savez(path, manifest=json.dumps({"x": {"__array__": "absent"}}))
    elif case == "bad_manifest":
        np.savez(path, manifest=np.zeros(2))
    elif case == "truncated_zip":
        path.write_bytes(b"PK\x03\x04truncated")
    elif case == "npy":
        with path.open("wb") as handle:
            np.save(handle, np.zeros(1))
    else:
        save(path, {"x": np.zeros(1)})
    result = subprocess.run([sys.executable, "-m", "hallucination_lab.cli", "run",
                             "hallucination_lab.toy:build", "--resume", str(path)],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 2, result.stderr
    assert "Invalid" in result.stderr and "checkpoint" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "runs").exists()


def test_controls_fail_before_compute(quadratic, key):
    with pytest.raises(ValueError):
        quadratic.evaluate(initial_logits(2), key, {"temperature": 0})
    with pytest.raises(ValueError):
        quadratic.evaluate(initial_logits(2), key, {"hard": 1.1})
    with pytest.raises(ValueError):
        quadratic.evaluate(initial_logits(3), key)


@pytest.mark.parametrize("kwargs", [{"length": True}, {"length": 2, "scale": True},
                                    {"length": True, "sequence": "A"}])
def test_initial_logits_rejects_booleans(kwargs):
    with pytest.raises(ValueError):
        initial_logits(**kwargs)


@pytest.mark.parametrize("controls", [
    {"temperature": True}, {"alpha": True}, {"soft": True}, {"hard": False},
    {"dropout": 1}, {"temperature": "1.0"},
])
def test_validate_controls_rejects_invalid_types(controls):
    from hallucination_lab.sequence import validate_controls
    with pytest.raises(ValueError):
        validate_controls(controls)


def test_prediction_export_roundtrip(prediction, tmp_path):
    from hallucination_lab.io import save_prediction
    path = tmp_path / "prediction.npz"
    save_prediction(path, prediction)
    restored = load(path)
    np.testing.assert_array_equal(restored["logits"], prediction.logits)
    assert "native" not in restored


def test_experiment_id_tracks_contents(tmp_path):
    from hallucination_lab.provenance import experiment_id
    path = tmp_path / "objective.py"
    path.write_text("x = 1\n")
    first = experiment_id(files=[path], settings={"seed": 1})
    path.write_text("x = 2\n")
    assert experiment_id(files=[path], settings={"seed": 1}) != first
