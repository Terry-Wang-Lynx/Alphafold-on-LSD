"""Chain identity regression for nonalphabetical, including equal-length, inputs."""
from types import SimpleNamespace
import numpy as np
import pytest
from hallucination_lab.backends._af3.mapping import atom_names, token_identity, protein_tokens


def batch(chains, residues):
    layout = SimpleNamespace(atom_name=np.array([["N", "CA"] if c else ["", ""] for c in chains]),
                             chain_id=np.repeat(np.array(chains)[:, None], 2, axis=1),
                             res_id=np.repeat(np.array(residues)[:, None], 2, axis=1))
    return {"token_atoms_layout": np.array(layout, object), "seq_mask": np.array([bool(c) for c in chains])}


def test_non_input_order_unequal_lengths():
    # Caller lists B then A, native layout lists A then B. Never infer asym from JSON.
    chains, residues = token_identity(batch(["A", "A", "A", "B", ""], [1, 2, 3, 1, 0]))
    np.testing.assert_array_equal(protein_tokens(chains, residues, "A", 3), [0, 1, 2])
    np.testing.assert_array_equal(protein_tokens(chains, residues, "B", 1), [3])


def test_equal_lengths_select_identity_not_length_or_input_order():
    chains, residues = token_identity(batch(["A", "A", "B", "B"], [1, 2, 1, 2]))
    # Different sequences; selecting B by input index would silently overwrite A.
    sequence_by_token = np.array(list("ARWC"))
    a = protein_tokens(chains, residues, "A", 2)
    b = protein_tokens(chains, residues, "B", 2)
    assert "".join(sequence_by_token[a]) == "AR"
    assert "".join(sequence_by_token[b]) == "WC"
    assert not set(a) & set(b)


def test_native_residue_ids_define_sequence_order():
    chains, residues = token_identity(batch(["Z", "Z", ""], [2, 1, 0]))
    np.testing.assert_array_equal(protein_tokens(chains, residues, "Z", 2), [1, 0])


@pytest.mark.parametrize("residue_ids", [[1, 1], [1, 3]])
def test_duplicate_or_missing_residue_identity_rejected(residue_ids):
    chains, residues = token_identity(batch(["A", "A"], residue_ids))
    with pytest.raises(ValueError, match="map once"):
        protein_tokens(chains, residues, "A", 2)


def test_missing_native_layout_fails_closed():
    with pytest.raises(ValueError, match="token_atoms_layout"):
        token_identity({"seq_mask": np.ones(2)})


def test_native_atom_names_preserved_with_padding_and_no_truncation():
    value = batch(["A", ""], [1, 0])
    layout = value["token_atoms_layout"].item()
    layout.atom_name = np.array([["LONG_NAME", "CA"], ["", ""]])
    names = atom_names(value)
    np.testing.assert_array_equal(names, [["LONG_NAME", "CA"], ["", ""]])
    names[0, 0] = "changed"
    assert layout.atom_name[0, 0] == "LONG_NAME"


def test_mixed_chain_token_rejected():
    value = batch(["A"], [1])
    value["token_atoms_layout"].item().chain_id[0, 1] = "B"
    with pytest.raises(ValueError, match="one chain"):
        token_identity(value)


def test_carrier_refeaturisation_rejects_changed_identity(monkeypatch, tmp_path):
    """Exercise AF3._batch itself; native featurizer is replaced by two layouts."""
    import sys
    from hallucination_lab.backends.af3 import AF3
    first = {**batch(["A", "A", "B", "B"], [1, 2, 1, 2]), "aatype": np.zeros(4, int),
             "pred_dense_atom_mask": np.ones((4, 2), bool)}
    native_batches = iter([first, batch(["B", "B", "A", "A"], [1, 2, 1, 2])])
    monkeypatch.setitem(sys.modules, "alphafold3.common", SimpleNamespace(
        folding_input=SimpleNamespace(load_fold_inputs_from_path=lambda path: iter([SimpleNamespace(user_ccd=None)]))))
    monkeypatch.setitem(sys.modules, "alphafold3.constants", SimpleNamespace(
        chemical_components=SimpleNamespace(Ccd=lambda user_ccd: object())))
    monkeypatch.setitem(sys.modules, "alphafold3.data", SimpleNamespace(
        featurisation=SimpleNamespace(featurise_input=lambda **kwargs: [next(native_batches)])))
    monkeypatch.setitem(sys.modules, "alphafold3.model.components", SimpleNamespace(
        utils=SimpleNamespace(remove_invalidly_typed_feats=lambda value: {"seq_mask": value["seq_mask"]})))
    backend = object.__new__(AF3)
    backend.input_data = {"sequences": [{"protein": {"id": "A", "sequence": "AA"}}]}
    backend.chain, backend.bucket, backend._profile_mode = "A", 4, "native"
    backend._auto_profile = False
    backend._cached_sequence = backend._cached_batch = None
    backend._scratch = SimpleNamespace(name=str(tmp_path))
    backend._batch("AA")
    np.testing.assert_array_equal(backend.token_chain_ids, ["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="changed token identity/order"):
        backend._batch("RR")
    assert backend._cached_sequence == "AA"


def test_a3m_query_preamble_and_explicit_homolog_identity(monkeypatch):
    import sys
    from hallucination_lab.backends.af3 import _rewrite_query
    msa = "\n #A3M#\n; comment\n>q\nA\nG\n>same-sequence-homolog\nAG\n>q\nAG\n>other\nA-g\n"
    assert _rewrite_query(msa, "VR", "AG") == (
        ">q\nVR\n>same-sequence-homolog\nAG\n>q\nVR\n>other\nA-g\n")
    assert _rewrite_query("  \n", "VR", "AG") == ""
    for invalid in ("AG\n>q\nAG", "#A3M#", ">q\n>other\nAG"):
        with pytest.raises(ValueError, match="A3M"):
            _rewrite_query(invalid, "VR", "AG")
    monkeypatch.setitem(sys.modules, "alphafold3.constants", SimpleNamespace(
        mmcif_names=SimpleNamespace(PROTEIN_CHAIN="protein")))
    monkeypatch.setitem(sys.modules, "alphafold3.data", SimpleNamespace(
        msa=SimpleNamespace(sequences_are_feature_equivalent=lambda a, b, p: a == b)))
    with pytest.raises(ValueError, match="original design chain"):
        _rewrite_query(">q\nAA", "VR", "AG")


def test_rebuilt_batch_export_uses_new_atom_names_and_cli_cutoff(monkeypatch, tmp_path):
    from datetime import date
    import json
    import sys
    from hallucination_lab.backends.af3 import AF3
    from hallucination_lab.sequence import ALPHABET

    requests = []
    def fold_inputs(path):
        content = json.loads(path.read_text())
        return iter([SimpleNamespace(protein=content["sequences"][0]["protein"], user_ccd="ccd")])

    def featurise_input(*, fold_input, ccd, buckets, ref_max_modified_date):
        prot = fold_input.protein
        seq = prot["sequence"]
        requests.append(seq)
        assert buckets == [1] and ccd == "ccd" and ref_max_modified_date == date(2021, 9, 30)
        assert prot["unpairedMsa"] == f">q\n{seq}\n>homolog\nR\n"
        native = batch(["A"], [1])
        name = {"A": "CB", "G": "", "V": "CB"}[seq]
        atom_names = np.array([["CA", name]])
        native["token_atoms_layout"].item().atom_name = atom_names
        mask = atom_names != ""
        if seq == "V":
            mask[0, 1] = False
        native.update(aatype=np.array([ALPHABET.index(seq)]), pred_dense_atom_mask=mask,
                      asym_id=np.ones(1), residue_index=np.arange(1))
        return [native]

    monkeypatch.setitem(sys.modules, "alphafold3.common", SimpleNamespace(
        folding_input=SimpleNamespace(load_fold_inputs_from_path=fold_inputs)))
    monkeypatch.setitem(sys.modules, "alphafold3.constants", SimpleNamespace(
        chemical_components=SimpleNamespace(Ccd=lambda user_ccd: user_ccd)))
    monkeypatch.setitem(sys.modules, "alphafold3.data", SimpleNamespace(
        featurisation=SimpleNamespace(featurise_input=featurise_input)))
    monkeypatch.setitem(sys.modules, "alphafold3.model.components", SimpleNamespace(
        utils=SimpleNamespace(remove_invalidly_typed_feats=lambda b: {
            k: v for k, v in b.items() if k != "token_atoms_layout"})))
    b = AF3.__new__(AF3)
    b.input_data = {"sequences": [{"protein": {"id": "A", "sequence": "A",
                                               "unpairedMsa": "\n#A3M#\n>q\nA\n>homolog\nR"}}]}
    b.sequence, b.positions, b.design_length = "A", np.array([0]), 1
    b.chain, b.bucket, b._matmul_precision = "A", 1, "highest"
    b._profile_mode, b._auto_profile = "native", True
    b._cached_sequence = b._cached_batch = None
    b._scratch = SimpleNamespace(name=str(tmp_path))
    b._full = lambda batch, pseudo, key, diffusion_key, native_input: {}
    key = np.array([0, 2], np.uint32)

    first = b.predict_hard_sequence("A", key, autotune=False)
    second = b.predict_hard_sequence("G", key, autotune=False)
    with pytest.raises(ValueError, match="dense atom mask"):
        b.predict_hard_sequence("V", key, autotune=False)
    cached = b.predict_hard_sequence("G", key, autotune=False)
    assert requests == ["A", "G", "V"]
    for result, names, mask in ((first, ["CA", "CB"], [1, 1]), (second, ["CA", ""], [1, 0]),
                                (cached, ["CA", ""], [1, 0])):
        np.testing.assert_array_equal(result["atom_names"], [names])
        np.testing.assert_array_equal(result["atom_mask"], [mask])
    assert b.input_data["sequences"][0]["protein"]["sequence"] == "A"
