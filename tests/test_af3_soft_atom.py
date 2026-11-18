"""CPU checks of the opt-in AF3 atom-mode contract."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hallucination_lab.backends._af3.features import create_msa_feat_with_soft_query
from hallucination_lab.backends._af3.soft_atom import (
    _conditional_layout_weights,
    _log_weighted_softmax,
    _pooling_mask,
    assemble_bank,
    candidate_layouts,
    f32_controls,
    input_chains,
    joint_log_probabilities,
    relax_atom_sequence,
    require_stable_entities,
)
from hallucination_lab.backends.af3 import AF3
from hallucination_lab.sequence import ALPHABET, validate_controls


def test_sites_choose_from_one_shared_distribution_at_soft_and_hard_endpoints():
    sequences, combinations, mask = candidate_layouts('GPA', [2, 0], ['VT', 'AG'])
    assert sequences == ('APV', 'GPV', 'APT', 'GPT')
    vals = np.zeros((2, 20), dtype=np.float32)
    vals[:, ALPHABET.index('W')] = 1000  # Excluded despite the highest raw score.
    vals[0, ALPHABET.index('T')] = 2
    vals[1, ALPHABET.index('G')] = 3
    p, logs = relax_atom_sequence(vals, validate_controls(None), mask)
    layouts = np.exp(joint_log_probabilities(logs, combinations))
    np.testing.assert_allclose(layouts, p[0, combinations[:, 0]] * p[1, combinations[:, 1]], rtol=1e-6)
    assert p[0, ALPHABET.index('W')] == 0
    locked, locked_logs = relax_atom_sequence(vals, validate_controls({'hard': 1}), mask)
    np.testing.assert_array_equal(np.exp(joint_log_probabilities(locked_logs, combinations)), [0, 0, 0, 1])
    np.testing.assert_array_equal(np.argmax(locked, axis=1), combinations[-1])
    assert sequences[-1] == 'GPT'


@pytest.mark.parametrize('temperature', [1., .05])
def test_small_joint_probabilities_keep_accurate_gradients(temperature):
    _, combinations, mask = candidate_layouts('AAA', [0, 1, 2], ['AV'] * 3)
    vals = jnp.zeros((3, 20), jnp.float32).at[:, ALPHABET.index('V')].set(-40 * temperature)
    member = np.stack([np.ones(8, bool), np.arange(8) == 7, np.zeros(8, bool)], axis=1)

    def score(logits):
        _, same_logs = relax_atom_sequence(logits, validate_controls({'temperature': temperature}), mask)
        logs = joint_log_probabilities(same_logs, combinations)
        log_mass, conditional = _conditional_layout_weights(logs, member)
        return log_mass[1] + jnp.sum(conditional[:, 2])

    value, grad = jax.jit(jax.value_and_grad(score))(vals)
    assert float(value) == pytest.approx(-120.)
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(grad[:, ALPHABET.index('V')], 1 / temperature)


@pytest.mark.parametrize('controls', [dict(alpha=1e-50), dict(temperature=1e40),
                                      dict(alpha=1e20, temperature=1e-20),
                                      dict(alpha=1e-20, temperature=1e20)])
def test_unrepresentable_f32_soft_atom_controls_raise(controls):
    with pytest.raises(ValueError, match='normal finite F32'):
        f32_controls(validate_controls(controls))


def test_soft_atom_controls_do_not_promote_probability_to_f64():
    _, combinations, allowed = candidate_layouts('A', [0], ['AG'])
    enable_x64 = jax.enable_x64 if hasattr(jax, 'enable_x64') else jax.experimental.enable_x64
    with enable_x64(True):
        logits = np.zeros((1, 20), np.float32)
        controls = validate_controls({'alpha': np.float64(1.), 'temperature': np.float64(.05)})
        p, logs = jax.jit(relax_atom_sequence)(logits, controls, allowed)
        assert p.dtype == logs.dtype == joint_log_probabilities(logs, combinations).dtype == jnp.float32
        np.testing.assert_array_equal(p[allowed], [.5, .5])


def test_soft_atom_input_requires_explicit_query_only_protein_and_bounded_layouts():
    protein = dict(id='A', sequence='AAA', unpairedMsa='', pairedMsa='', templates=[])
    assert input_chains({'sequences': [{'protein': protein}]}) == {'A': 'AAA'}
    with pytest.raises(ValueError, match='query_only'):
        input_chains({'sequences': [{'protein': {**protein, 'unpairedMsa': '>q\nAAA\n'}}]})
    with pytest.raises(ValueError, match='protein chains'):
        input_chains({'sequences': [{'protein': protein}, {'ligand': {'id': 'B'}}]})
    with pytest.raises(ValueError, match='at most 27'):
        candidate_layouts('AAAA', [0, 1, 2, 3], ['AVF'] * 4)


def test_atom_mode_constructor_rejects_invalid_options_before_loading_af3():
    kwargs = dict(input_json='unused', design_chain='A', objective=None, model_dir='unused')
    with pytest.raises(ValueError, match='must be hard or soft'):
        AF3(**kwargs, atom_mode='other')
    with pytest.raises(ValueError, match='requires atom_mode'):
        AF3(**kwargs, soft_atom_alphabets=['AG'])


def two_chain_batch(sequence):
    """Complete native-shaped query rows and fully populated key windows."""
    names = np.array([['N', 'CA', 'C', 'O', ''], ['N', 'CA', 'C', 'O', 'CB']])
    if sequence[0] != 'G':
        names[0, -1] = 'CB'
    real = names != ''
    query = np.flatnonzero(real)[None, :].astype(np.int32)
    aatype = np.array([ALPHABET.index(aa) for aa in sequence], dtype=np.int32)
    return {
        **{name: np.zeros(2, dtype=bool) for name in
           ('is_rna', 'is_dna', 'is_ligand', 'is_water', 'is_nonstandard_polymer_chain')},
        'seq_mask': np.ones(2, bool), 'seq_length': np.array(2), 'is_protein': np.ones(2, bool),
        'token_index': np.arange(2), 'residue_index': np.ones(2, np.int32),
        'asym_id': np.array([1, 2]), 'entity_id': np.array([1, 2]), 'sym_id': np.ones(2, np.int32),
        'aatype': aatype, 'msa': np.tile(aatype, (2, 1)), 'msa_mask': np.ones((2, 2), bool),
        'profile': np.eye(31, dtype=np.float32)[aatype], 'deletion_mean': np.zeros(2),
        'deletion_matrix': np.zeros((2, 2)), 'template_atom_mask': np.zeros((2, 5), bool),
        'token_atoms_layout': SimpleNamespace(atom_name=names,
                                              chain_id=np.broadcast_to(np.array([['A'], ['B']]), real.shape),
                                              res_id=np.ones_like(real, dtype=np.int32)),
        'pred_dense_atom_mask': real, 'ref_mask': real, 'ref_pos': np.zeros((*real.shape, 3), np.float32),
        'ref_element': np.full(real.shape, 6, np.int32), 'ref_charge': np.zeros(real.shape, np.float32),
        'ref_atom_name_chars': np.zeros((*real.shape, 4), np.int32),
        'ref_space_uid': np.broadcast_to(np.arange(2)[:, None], real.shape),
        'token_atoms_to_queries:gather_idxs': query, 'token_atoms_to_queries:gather_mask': np.ones_like(query, bool),
        'queries_to_keys:gather_idxs': np.arange(4, dtype=np.int32)[None, :],
        'queries_to_keys:gather_mask': np.ones((1, 4), bool),
    }


def test_two_chain_bank_requires_native_query_rows_and_key_windows_and_preserves_pooling():
    raw, mutant = two_chain_batch('GR'), two_chain_batch('AR')
    by_chain = [{'A': 'G', 'B': 'R'}, {'A': 'A', 'B': 'R'}]
    bank = assemble_bank(raw, [raw, mutant], by_chain, [0])
    for layout, expected in zip(bank['exists'], ([4, 5], [5, 5]), strict=True):
        actual = _pooling_mask(bank, layout.astype(np.float32))
        np.testing.assert_array_equal(actual.sum(axis=1), expected)
        assert set(np.unique(actual)) == {0, 1}
    broken_keys = {**mutant, 'queries_to_keys:gather_mask': np.array([[1, 1, 1, 0]], bool)}
    with pytest.raises(ValueError, match='complete native key windows'):
        assemble_bank(raw, [raw, broken_keys], by_chain, [0])
    gap = mutant['msa'].copy()
    gap[1, 0] = 21
    with pytest.raises(ValueError, match='non-query effective MSA row'):
        assemble_bank(raw, [raw, {**mutant, 'msa': gap}], by_chain, [0])
    require_stable_entities(by_chain[0], 'A', ('G', 'A'))
    with pytest.raises(ValueError, match='preserve native chain entity assignments'):
        require_stable_entities(by_chain[0], 'A', ('R',))
    with pytest.raises(ValueError, match='preserve native chain entity assignments'):
        require_stable_entities({'A': 'G', 'B': 'G'}, 'A', ('A',))


def test_log_attention_prior_and_empty_query_have_finite_gradient():
    prior = jnp.array([[-122., -120.], [-jnp.inf, -jnp.inf]], dtype=jnp.float32)
    logits = jnp.zeros_like(prior)
    weights = jax.jit(_log_weighted_softmax)(logits, prior)
    np.testing.assert_allclose(weights[0], jax.nn.softmax(jnp.array([-2., 0.])))
    np.testing.assert_array_equal(weights[1], [0, 0])
    grad = jax.grad(lambda x: _log_weighted_softmax(x, prior)[0, 0])(logits)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(grad[1], [0, 0])


def test_query_only_injection_uses_duplicate_queries_and_preserves_padding():
    msa = SimpleNamespace(rows=jnp.array([[0, 1], [0, 1], [2, 2]]),
                          mask=jnp.array([[1, 1], [1, 1], [0, 0]]),
                          deletion_matrix=jnp.zeros((3, 2)))
    pseudo = jnp.zeros((2, 31)).at[0, 0].set(.25).at[0, 3].set(.75)
    batch = create_msa_feat_with_soft_query(msa, pseudo, np.array([True, False]), query_only=True)
    np.testing.assert_array_equal(batch[:2, 0, :31], jnp.tile(pseudo[0], (2, 1)))
    np.testing.assert_array_equal(batch[2, :, :32], jax.nn.one_hot(msa.rows[2], 32))
    np.testing.assert_array_equal(batch[:, 1, :32], jax.nn.one_hot(msa.rows[:, 1], 32))


def test_evaluate_uses_fixed_soft_bank_and_decodes_only_allowed_identities(key):
    b = AF3.__new__(AF3)
    b.atom_mode = 'soft'
    b.design_length, b.sequence, b.positions = 1, 'AAA', np.array([1])
    b._matmul_precision = 'highest'
    _, _, b._allowed_amino_acids = candidate_layouts('AAA', [1], ['FY'])
    b._soft_reference, b._soft_atom_bank, b._full_parameters = object(), object(), object()
    b._batch = lambda seq: pytest.fail(f'Unexpected hard re-featurisation of {seq}')

    def score(values, batch, key, controls, bank, weights):
        assert batch is b._soft_reference and bank is b._soft_atom_bank
        assert weights is b._full_parameters
        locked, _ = relax_atom_sequence(values, controls, b._allowed_amino_acids)
        assert b.sequence_from_logits(values, controls) == 'A' + ALPHABET[np.argmax(locked)] + 'A'
        return 2., {}

    b._score = score
    logits = np.zeros((1, 20), np.float32)
    logits[0, ALPHABET.index('W')] = 100
    logits[0, ALPHABET.index('Y')] = 10
    assert b.sequence_from_logits(logits) == 'AYA'
    assert b.evaluate(logits, key, {'hard': 1}, gradient=False).loss == 2.
    almost_tied = logits.astype(np.float64)
    almost_tied[0, ALPHABET.index('F')] = 1
    almost_tied[0, ALPHABET.index('Y')] = 1 + 1e-8
    # Distinct F64 scores collapse to a tie at the model's F32 boundary.
    assert b.sequence_from_logits(almost_tied) == 'AFA'
    assert b.evaluate(almost_tied, key, {'hard': 1}, gradient=False).loss == 2.
    subnormal = np.zeros((1, 20), np.float32)
    subnormal[0, ALPHABET.index('Y')] = np.nextafter(np.float32(0), np.float32(1))
    assert b.sequence_from_logits(subnormal) == 'AFA'
    assert b.evaluate(subnormal, key, {'hard': 1}, gradient=False).loss == 2.
    with pytest.raises(ValueError, match='gradient=False'):
        b.evaluate(logits, key, {'hard': 1})
    with pytest.raises(ValueError, match='soft=1'):
        b.evaluate(logits, key, {'soft': .5})
    with pytest.raises(ValueError, match='normal finite F32'):
        b.evaluate(logits, key, {'alpha': 1e-50})


def test_soft_full_prediction_prepares_locked_native_atoms(key):
    b = AF3.__new__(AF3)
    b.atom_mode = 'soft'
    b.design_length, b.sequence, b.positions = 1, 'A', np.array([0])
    b._matmul_precision = 'highest'
    _, _, b._allowed_amino_acids = candidate_layouts('A', [0], ['FY'])
    b.token_chain_ids = np.array(['A'])
    b.token_residue_ids = np.array([1])
    b._atom_names = np.array([['CA']])

    def batch(sequence):
        assert sequence == 'Y'
        return {name: np.ones(1) for name in ('pred_dense_atom_mask', 'seq_mask', 'asym_id', 'residue_index')}

    b._batch = batch
    b._features = lambda values, controls: relax_atom_sequence(values, controls, b._allowed_amino_acids)

    def full(batch, locked_logs, key):
        np.testing.assert_array_equal(np.exp(locked_logs), jax.nn.one_hot([ALPHABET.index('Y')], 20))
        return {'diffusion_atom_positions': np.zeros((1, 1, 3))}

    b._full = full
    logits = np.zeros((1, 20), np.float32)
    logits[0, ALPHABET.index('Y')] = 1
    assert b.full_prediction(logits, key, autotune=False)['atom_names'][0, 0] == 'CA'
