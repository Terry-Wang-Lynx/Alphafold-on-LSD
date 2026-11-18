"""Test example configuration and generated inputs without loading native models."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def load_example(name, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(f"public_example_{name}", root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_stage_boundaries_and_step_counts(monkeypatch):
    module = load_example("stages", monkeypatch)
    stages = module.make_stages({"stage_steps": [3, 5, 2], "learning_rate": 0.04})
    assert [s.steps for s in stages] == [3, 5, 2]
    assert stages[0].at(0)["soft"] == 0
    assert stages[0].at(2)["soft"] == 1
    assert stages[1].at(0)["temperature"] == 1
    assert stages[1].at(4)["temperature"] == pytest.approx(0.2)
    assert stages[2].at(0)["hard"] == stages[2].at(1)["hard"] == 1
    assert all(s.learning_rate == 0.04 for s in stages)


@pytest.mark.parametrize("counts", [[0, 4, 2], [4, 2], [True, 4, 2], [1.5, 4, 2]])
def test_invalid_stages_rejected(monkeypatch, counts):
    with pytest.raises(ValueError, match="three positive integers"):
        load_example("stages", monkeypatch).make_stages({"stage_steps": counts})


def test_artificial_af3_input_identity_and_no_overwrite(monkeypatch, tmp_path):
    module = load_example("make_af3_input", monkeypatch)
    output = tmp_path / "artificial"
    module.create(output, 12)
    data = json.loads((output / "input.json").read_text())
    chain = data["sequences"][0]["protein"]
    assert chain["id"] == "A" and chain["sequence"] == "A" * 12
    assert chain["templates"] == [] and chain["unpairedMsa"] == chain["pairedMsa"] == ""
    mask = np.load(output / "pair_mask.npy", allow_pickle=False)
    assert mask.shape == (12, 12) and mask[0, 8] == mask[8, 0] == 1
    assert mask[0, 7] == mask[0, 0] == 0
    before = (output / "input.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.create(output, 16)
    assert (output / "input.json").read_bytes() == before


@pytest.mark.parametrize("backend_name,atom_mode", [("af2", None), ("af3", "hard"), ("af3", "soft")])
def test_model_factory_uses_configured_stages_without_runtime(monkeypatch, tmp_path, backend_name, atom_mode):
    module = load_example(backend_name, monkeypatch)

    class FakeBackend:
        def __init__(self, **kwargs):
            assert kwargs.get("atom_mode") == atom_mode
            if atom_mode == "soft":
                assert kwargs["soft_atom_alphabets"] == ["AG"]
            self.design_length = 16
            self.metadata = {"backend": "synthetic_configuration_test"}

    monkeypatch.setattr(module, backend_name.upper(), FakeBackend)
    config = {"output": str(tmp_path / "run"), "stage_steps": [2, 3, 1],
              "length": 16, "data_dir": str(tmp_path)}
    if backend_name == "af3":
        load_example("make_af3_input", monkeypatch).create(tmp_path / "input", 16)
        config.update(input_json=str(tmp_path / "input/input.json"),
                      pair_mask=str(tmp_path / "input/pair_mask.npy"),
                      model_dir=str(tmp_path), design_chain="A")
    if atom_mode == "soft":
        config.update(atom_mode="soft", design_positions=[3], soft_atom_alphabets=["AG"])
    run = module.build(config)
    assert run.total_steps == 6
    if atom_mode == "soft":
        stage, = run.stages
        assert stage.name == "soft"
        assert stage.at(0)["temperature"] == 1.
        assert stage.at(5)["temperature"] == pytest.approx(.05)
        assert all(stage.at(i)["hard"] == 0 and stage.at(i)["soft"] == 1 for i in range(stage.steps))
    else:
        assert [s.name for s in run.stages] == ["logits", "soft", "hard"]
