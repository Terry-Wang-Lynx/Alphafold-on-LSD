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
    backend._full = lambda batch, full, key: {"diffusion_atom_positions": jnp.zeros((2, 2, 3))}
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

    def forward(batch, full, key):
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
