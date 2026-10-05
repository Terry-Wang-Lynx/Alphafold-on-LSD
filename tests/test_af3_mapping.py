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
    native_batches = iter([batch(["A", "A", "B", "B"], [1, 2, 1, 2]),
                           batch(["B", "B", "A", "A"], [1, 2, 1, 2])])
    monkeypatch.setitem(sys.modules, "alphafold3.common", SimpleNamespace(
        folding_input=SimpleNamespace(load_fold_inputs_from_path=lambda path: iter([None]))))
    monkeypatch.setitem(sys.modules, "alphafold3.constants", SimpleNamespace(
        chemical_components=SimpleNamespace(Ccd=object)))
    monkeypatch.setitem(sys.modules, "alphafold3.data", SimpleNamespace(
        featurisation=SimpleNamespace(featurise_input=lambda **kwargs: [next(native_batches)])))
    monkeypatch.setitem(sys.modules, "alphafold3.model.components", SimpleNamespace(
        utils=SimpleNamespace(remove_invalidly_typed_feats=lambda value: {"seq_mask": value["seq_mask"]})))
    backend = object.__new__(AF3)
    backend.input_data = {"sequences": [{"protein": {"id": "A", "sequence": "AA"}}]}
    backend.chain, backend.bucket = "A", 4
    backend._cached_sequence = backend._cached_batch = None
    backend._scratch = SimpleNamespace(name=str(tmp_path))
    backend._batch("AA")
    np.testing.assert_array_equal(backend.token_chain_ids, ["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="changed token identity/order"):
        backend._batch("RR")
    assert backend._cached_sequence == "AA"
