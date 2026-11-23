"""Numerical adapter contracts without installing the AF3 runtime."""
from types import SimpleNamespace
from contextlib import contextmanager
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hallucination_lab.backends._af3.features import inject_soft_sequence, create_msa_feat_with_soft_query


def test_fixed_rows_preserved_and_gradient_excluded():
    baseline = jax.nn.one_hot(jnp.array([0, 1, 2]), 31)
    profile = baseline * 0.5
    pseudo = jnp.ones((3, 31)) / 31
    mask = jnp.array([False, True, False])
    target, updated_profile = inject_soft_sequence(baseline, profile, pseudo, mask)
    np.testing.assert_array_equal(np.asarray(target)[[0, 2]], np.asarray(baseline)[[0, 2]])
    np.testing.assert_array_equal(np.asarray(updated_profile)[[0, 2]], np.asarray(profile)[[0, 2]])
    np.testing.assert_allclose(target[1], pseudo[1])
    gradient = jax.grad(lambda x: sum(v.sum() for v in inject_soft_sequence(baseline, profile, x, mask)))(pseudo)
    np.testing.assert_array_equal(np.asarray(gradient)[[0, 2]], 0)
    np.testing.assert_array_equal(gradient[1], 2)


def test_msa_only_designed_query_positions_change():
    msa = SimpleNamespace(rows=jnp.array([[0, 1, 2], [3, 4, 5]]),
                          deletion_matrix=jnp.zeros((2, 3)))
    pseudo = jnp.ones((3, 31)) / 31
    mask = jnp.array([False, True, False])
    features = create_msa_feat_with_soft_query(msa, pseudo, mask)
    assert features.shape == (2, 3, 34)
    np.testing.assert_array_equal(features[1, :, :32], jax.nn.one_hot(msa.rows[1], 32))
    np.testing.assert_array_equal(features[0, 0, :32], jax.nn.one_hot(0, 32))
    np.testing.assert_allclose(features[0, 1, :31], pseudo[1])
    np.testing.assert_array_equal(features[:, :, 32:], 0)


def test_forward_only_export_retains_padding_masks():
    from hallucination_lab.backends.af3 import AF3
    backend = object.__new__(AF3)
    backend._matmul_precision = "highest"
    backend.design_length = 2
    backend._atom_names = np.array([["N", "CA"], ["", ""]])
    backend.token_chain_ids = np.array(["A", ""])
    backend.token_residue_ids = np.array([1, 0])
    batch = {"pred_dense_atom_mask": np.array([[1, 1], [0, 0]]),
             "seq_mask": np.array([1, 0]), "asym_id": np.array([1, 0]),
             "residue_index": np.array([0, 0])}
    backend._carrier = lambda logits, controls: batch
    backend._features = lambda logits, controls: (logits, logits)
    backend._full = lambda batch, full, key, diffusion_key, native_input: {"diffusion_atom_positions": jnp.zeros((2, 2, 3))}
    result = backend.full_prediction(np.zeros((2, 20), np.float32), np.array([0, 1], np.uint32), autotune=False)
    np.testing.assert_array_equal(result["atom_mask"], batch["pred_dense_atom_mask"])
    np.testing.assert_array_equal(result["token_mask"], batch["seq_mask"])
    np.testing.assert_array_equal(result["atom_names"], [["N", "CA"], ["", ""]])
    np.testing.assert_array_equal(result["token_chain_ids"], ["A", ""])
    np.testing.assert_array_equal(result["token_residue_ids"], [1, 0])
    result["atom_names"][0, 0] = "X"
    assert backend._atom_names[0, 0] == "N"


def test_recycle_last_gradient_preserves_forward_but_cuts_all_previous_passes():
    from hallucination_lab.backends._af3.features import recycle_embeddings
    def result(sequence, detach):
        def body(previous, key):
            return 2*previous + sequence, key
        return recycle_embeddings(body, jnp.array(0.), None, 2, detach=detach)
    assert float(result(jnp.array(3.), True)) == 21.
    assert float(result(jnp.array(3.), False)) == 21.
    assert float(jax.grad(lambda s: result(s, True))(jnp.array(3.))) == 1.
    assert float(jax.grad(lambda s: result(s, False))(jnp.array(3.))) == 7.


@pytest.mark.parametrize('fails', [False, True])
def test_full_prediction_default_tuning_is_local_even_on_failure(monkeypatch, fails):
    from hallucination_lab.backends.af3 import AF3
    active = []
    events = []

    @contextmanager
    def tuning(mode):
        assert mode == 'autotune'
        active.append(mode)
        events.append('enter')
        try:
            yield
        finally:
            active.pop()
            events.append('exit')

    monkeypatch.setitem(sys.modules, 'tokamax', SimpleNamespace(
        config=SimpleNamespace(autotuning_cache_miss_fallback=tuning)))
    backend = object.__new__(AF3)
    backend._matmul_precision = 'highest'
    backend.design_length = 1
    backend._atom_names = np.array([['CA']])
    backend.token_chain_ids = np.array(['A'])
    backend.token_residue_ids = np.array([1])
    batch = {'pred_dense_atom_mask': np.ones((1, 1)), 'seq_mask': np.ones(1),
             'asym_id': np.ones(1), 'residue_index': np.zeros(1)}
    backend._carrier = lambda logits, controls: batch
    backend._features = lambda logits, controls: (logits, logits)

    def forward(batch, full, key, diffusion_key, native_input):
        assert active == ['autotune']
        if fails:
            raise RuntimeError('kernel failure')
        return {'diffusion_atom_positions': np.zeros((1, 1, 3))}

    backend._full = forward
    args = (np.zeros((1, 20), np.float32), np.array([0, 1], np.uint32))
    if fails:
        with pytest.raises(RuntimeError, match='kernel failure'):
            backend.full_prediction(*args)
    else:
        backend.full_prediction(*args)
    assert active == [] and events == ['enter', 'exit']


def test_hard_string_rebuilds_native_inputs_and_keeps_separate_sampling_key():
    from hallucination_lab.backends.af3 import AF3
    backend = AF3.__new__(AF3)
    backend._matmul_precision = "highest"
    backend.sequence = "GPA"
    backend.design_length, backend.positions = 2, np.array([2, 0])
    backend._atom_names = np.array([["CA"]] * 3)
    backend.token_chain_ids = np.array(["A"] * 3)
    backend.token_residue_ids = np.arange(1, 4)
    batch = {"pred_dense_atom_mask": np.ones((3, 1)), "seq_mask": np.ones(3),
             "asym_id": np.ones(3), "residue_index": np.arange(3)}
    calls = []
    def build(sequence):
        calls.append(sequence)
        return batch
    def full(target, pseudo, key, diffusion, native_input):
        assert target is batch and pseudo is None and native_input is True
        return {"readout": np.asarray(key), "sample": np.asarray(diffusion)}
    backend._batch, backend._full = build, full
    key = np.array([0, 3], np.uint32)
    first = backend.predict_hard_sequence("RPG", key, autotune=False)
    np.testing.assert_array_equal(first["sample"], jax.random.fold_in(key, 1))
    assert first["sequence_representation"] == "native_hard" and first["sequence"] == "RPG"
    other = np.array([9, 8], np.uint32)
    second = backend.predict_hard_sequence("RPG", key, autotune=False, diffusion_key=other)
    np.testing.assert_array_equal(first["readout"], second["readout"])
    np.testing.assert_array_equal(second["sample"], other)
    assert second["rng_provenance"] == dict(model_key=[0, 3], diffusion_key=[9, 8],
                                            diffusion_key_source="explicit")
    for invalid in ("RG", "RAG", "RPX"):
        with pytest.raises(ValueError):
            backend.predict_hard_sequence(invalid, key)
    assert calls == ["RPG", "RPG"]


def test_homolog_profile_and_shuffled_query_copy_rows():
    from hallucination_lab.backends.af3 import _check_carrier, _profile_mode
    protein = {"sequence": "AR", "unpairedMsa": ">q\nAR\n>other\nAA"}
    assert _profile_mode([protein], "auto") == "native"
    with pytest.raises(ValueError, match="query_only"):
        _profile_mode([protein], "query_only")
    assert _profile_mode([{"unpairedMsa": ""}], "auto") == "query_only"
    assert _profile_mode([{"unpairedMsa": ""}, protein], "auto") == "native"
    onehot = jax.nn.one_hot(jnp.array([0, 1]), 31)
    profile = onehot.at[1, :2].set(.5)
    pseudo = jnp.ones((2, 31)) / 31
    mask = jnp.array([True, True])
    _, unchanged = inject_soft_sequence(onehot, profile, pseudo, mask, profile_mode="native")
    np.testing.assert_array_equal(unchanged, profile)
    rows = jnp.array([[0, 1], [0, 1], [0, 0]])
    active = jnp.array([[True, True], [True, True], [False, False]])
    msa = SimpleNamespace(rows=rows, mask=active, deletion_matrix=jnp.zeros_like(rows))
    batch = dict(msa=rows, msa_mask=active, profile=onehot, aatype=np.array([0, 1]))
    _check_carrier(batch, np.arange(2), "AR", "query_only")
    with pytest.raises(ValueError, match="query_only"):
        _check_carrier({**batch, "profile": profile}, np.arange(2), "AR", "query_only")
    with pytest.raises(ValueError, match="identities"):
        _check_carrier(batch, np.arange(2), "RA", "native")
    feat = create_msa_feat_with_soft_query(msa, pseudo, mask, query_only=True)
    np.testing.assert_allclose(feat[1, :, :31], pseudo)
    np.testing.assert_array_equal(feat[2, :, :32], jax.nn.one_hot(rows[2], 32))
    shuffled = create_msa_feat_with_soft_query(msa, pseudo, mask, source_rows=jnp.array([4, 0, 7]))
    np.testing.assert_array_equal(shuffled[0, :, :32], jax.nn.one_hot(rows[0], 32))
    np.testing.assert_allclose(shuffled[1, :, :31], pseudo)


def test_compressed_and_sharded_weights_use_native_file_order(tmp_path):
    import hashlib
    from hallucination_lab.backends.af3 import _weight_provenance
    files = [tmp_path / "weights.bin.zst.00", tmp_path / "weights.bin.zst.01"]
    files[0].write_bytes(b"x")
    files[1].write_bytes(b"y")
    assert _weight_provenance(files) == {"weight_sha256": hashlib.sha256(b"xy").hexdigest(),
                                         "weight_files": [p.name for p in files]}


def test_true_logit_winner_and_extreme_finite_scaling():
    from hallucination_lab.backends.af3 import AF3
    from hallucination_lab.sequence import relax, validate_controls
    model = AF3.__new__(AF3)
    model.positions, model.design_length, model.sequence = np.array([0]), 1, "A"
    logits = np.ones((1, 20), np.float32)
    logits[0, -1] += 1e-6
    controls = validate_controls({"hard": 1., "temperature": 100.})
    np.testing.assert_array_equal(relax(logits, controls), np.eye(20)[[-1]])
    assert model.sequence_from_logits(logits, controls) == "V"
    x = jnp.full((1, 20), -3e38, jnp.float32).at[0, -1].set(3e38)
    # Multiplying the positive peak first would overflow; sharp output is finite.
    for c in ({"alpha": 2., "temperature": 1.}, {"alpha": 1., "temperature": 2e37}):
        y, back = jax.vjp(lambda a: relax(a, c), x)
        assert np.isfinite(y).all() and np.isfinite(back(jnp.ones_like(y))[0]).all()
        if c["temperature"] > 1:
            np.testing.assert_allclose(float(y[0, 0]), np.exp(-30), rtol=2e-5, atol=0)


def test_jitted_hard_and_training_frontdoors_use_distinct_features(monkeypatch, tmp_path):
    """Execute actual AF3 adapter JIT wiring on CPU with small deterministic stand-ins."""
    import importlib.metadata
    import json
    from hallucination_lab.backends.af3 import AF3
    from hallucination_lab.sequence import ALPHABET, initial_logits

    def transform(fn):
        def apply(weights, rng, *args):
            return fn(*args)
        return SimpleNamespace(apply=apply)

    class Config:
        def __init__(self):
            self.global_config = SimpleNamespace()
            self.heads = SimpleNamespace(diffusion=SimpleNamespace(eval=SimpleNamespace()))
        def as_dict(self):
            return {}

    class Gate:
        def __init__(self, config, remat, mask, **kw):
            self.native_input = kw.get("native_sequence_input", False)
        def __call__(self, batch, full, profile, key, diffusion_key=None):
            if diffusion_key is None:
                scored = full[:, :20]
                d = {"distogram": scored, "bin_edges": jnp.array([1.])}
                return {"distogram": d}
            tokens = (jax.nn.one_hot(batch["aatype"], 31) if self.native_input else full)
            return {"observed": tokens, "actual_profile": batch["profile"],
                    "diffusion_operand": diffusion_key}

    code = tmp_path / "library/model/model.py"
    code.parent.mkdir(parents=True)
    code.write_text("# test AF3 boundary\n")
    archive = tmp_path / "weights.bin.zst"
    archive.write_bytes(b"synthetic params")
    monkeypatch.setitem(sys.modules, "haiku", SimpleNamespace(transform=transform))
    monkeypatch.setitem(sys.modules, "alphafold3.model", SimpleNamespace(
        model=SimpleNamespace(Model=SimpleNamespace(Config=Config), __file__=str(code)),
        params=SimpleNamespace(get_model_haiku_params=lambda model_dir: {},
                               select_model_files=lambda model_dir: ([archive], True))))
    monkeypatch.setitem(sys.modules, "alphafold3.model.network", SimpleNamespace(
        evoformer=SimpleNamespace(__file__=str(code))))
    monkeypatch.setitem(sys.modules, "absl", SimpleNamespace(flags=SimpleNamespace(
        FLAGS=SimpleNamespace(is_parsed=lambda: True))))
    monkeypatch.setitem(sys.modules, "hallucination_lab.backends._af3.hybrid", SimpleNamespace(
        HybridDesignTrunk=Gate, HybridFullGate=Gate))
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "fixture" if name == "alphafold3" else version(name))
    def batch(self, sequence):
        self.token_chain_ids = np.array(["A", "A"])
        self.token_residue_ids = np.array([1, 2])
        self._atom_names = np.array([["CA"], ["CA"]])
        ids = jnp.array([ALPHABET.index(a) for a in sequence])
        return {"aatype": ids, "residue_index": jnp.arange(2), "asym_id": jnp.ones(2),
                "profile": jnp.full((2, 31), 1 / 31), "seq_mask": jnp.ones(2),
                "pred_dense_atom_mask": jnp.ones((2, 1))}
    monkeypatch.setattr(AF3, "_batch", batch)
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps({"sequences": [{"protein": {
        "id": "A", "sequence": "AG", "unpairedMsa": ">q\nAG\n>homolog\nGG\n"}}]}))
    model = AF3(input_json=input_json, design_chain="A", model_dir=tmp_path, bucket=2,
                objective=lambda p, c: p.sequence[0, 0])
    key = np.array([0, 71], np.uint32)
    try:
        x = initial_logits(sequence="AG")
        train = model.evaluate(x, key)
        hybrid = model.full_prediction(x, key, autotune=False)
        hard = model.predict_hard_sequence("AR", key, autotune=False, diffusion_key=key)
        assert model.metadata["weight_files"] == ["weights.bin.zst"]
        assert model.metadata["profile_mode"] == "native"
        np.testing.assert_allclose(train.loss, hybrid["observed"][0, 0])
        assert train.gradient.shape == (2, 20) and train.gradient[0, 0] > 0
        assert np.count_nonzero(hybrid["observed"][0]) == 20
        np.testing.assert_array_equal(hard["observed"], np.eye(31)[[0, 1]])
        np.testing.assert_allclose(hard["actual_profile"], 1 / 31)
        np.testing.assert_array_equal(hard["diffusion_operand"], key)
        with pytest.raises(ValueError, match="bucket"):
            AF3(input_json=input_json, design_chain="A", model_dir=tmp_path, bucket=True,
                objective=lambda p, c: p.sequence[0, 0])
        with pytest.raises(ValueError, match="recycles"):
            AF3(input_json=input_json, design_chain="A", model_dir=tmp_path, bucket=2,
                recycles=True, objective=lambda p, c: p.sequence[0, 0])
    finally:
        model.close()
