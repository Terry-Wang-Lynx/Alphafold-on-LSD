"""CPU contract tests with a tiny differentiable stand-in for the AF2 runtime."""
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hallucination_lab.backends.af2 import AF2


class NativeModel:
    def __init__(self, **kwargs):
        self._model_names = kwargs['model_names']
        self._model_params = [jnp.array(i + 1., dtype=jnp.float32)
                              for i in range(len(self._model_names))]
        self._args = {'use_dgram': False}
        self._cfg = SimpleNamespace(to_dict=lambda: {})
        self.opt = {'weights': {'native': 9.}}
        self.traces = 0

        def forward(params, weights, inputs, key):
            self.traces += 1
            value = params['seq'].sum() * weights + inputs['prev']['prev_pos'].sum()
            value += jnp.where(inputs['opt']['pssm_hard'], 7., 0.)
            aux = {'plddt': value, 'pae': value, 'ptm': value, 'i_ptm': value,
                   'seq': {k: params['seq'] for k in ('pseudo', 'logits', 'pssm', 'hard')},
                   'losses': {'native': value, 'unselected': value * 1000},
                   'prev': {'prev_pos': jnp.ones((1, 37, 3)) * weights * params['seq'].sum(),
                            'prev_pair': inputs['prev']['prev_pair'],
                            'prev_msa_first_row': inputs['prev']['prev_msa_first_row']}}
            outputs = {'distogram': {'logits': jnp.ones((1, 1, 2)) * value,
                                     'bin_edges': jnp.array([8.])}}
            kwargs['post_callback'](inputs, outputs, aux, params['seq'])
            total = sum(v * inputs['opt']['weights'].get(k, 1.) for k, v in aux['losses'].items())
            return total, aux
        self._model = {'fn': jax.jit(forward),
                       'grad_fn': jax.jit(jax.value_and_grad(forward, has_aux=True))}

    def prep_inputs(self, **kwargs):
        self._inputs = {'seq_mask': jnp.ones(1), 'residue_index': jnp.arange(1),
                        'asym_id': jnp.zeros(1)}

    def set_seq(self, sequence):
        self._params = {'seq': jnp.ones((1, 1, 20))}


@pytest.fixture
def backend(monkeypatch):
    import hallucination_lab.backends.af2 as module
    monkeypatch.setitem(sys.modules, 'colabdesign', SimpleNamespace(mk_afdesign_model=NativeModel,
                                                                  __file__=__file__))
    monkeypatch.setattr(module, 'version', lambda name: 'mock')
    def objective(p, controls):
        return p.native['losses']['native'] * p.native['loss_weights']['native'], {}
    def prepare(model):
        model.opt['weights']['native'] = 2.
    return AF2(sequence='A', objective=objective, data_dir='.', prepare=prepare,
               model_names=['model_1_ptm', 'model_2_ptm'])


def test_bank_parameter_switch_reuses_trace_and_explicit_objective(backend):
    args = (np.ones((1, 20)), np.array([0, 1], np.uint32))
    first = backend.evaluate(*args, prediction=True)
    traces = backend.model.traces
    second = backend.evaluate(*args, {'model_index': 1}, prediction=True)
    assert backend.model.traces == traces
    assert first.loss == 40.
    assert second.loss == 80.
    np.testing.assert_array_equal(first.gradient, np.full((1, 20), 2.))
    np.testing.assert_array_equal(second.gradient, np.full((1, 20), 4.))
    assert second.metrics['af2_model_index'] == 1
    assert 'unselected' in second.prediction.native['losses']
    assert second.prediction.native['loss_weights']['native'] == 2.


def test_recycle_override_and_detached_prior_pass(backend):
    result = backend.evaluate(np.ones((1, 20)), np.array([0, 1], np.uint32),
                              {'model_index': 1, 'recycles': 1})
    assert result.loss == 8960.
    np.testing.assert_array_equal(result.gradient, np.full((1, 20), 4.))
    assert result.metrics['af2_recycles'] == 1


@pytest.mark.parametrize('controls', [{'model_index': 2}, {'model_index': -1},
                                     {'model_index': True}, {'recycles': 1.5}, {'recycles': -1}])
def test_invalid_runtime_selection(backend, controls):
    with pytest.raises(ValueError):
        backend.evaluate(np.ones((1, 20)), np.array([0, 1], np.uint32), controls)


def test_pssm_hard_and_native_sequence(backend):
    result = backend.evaluate(np.ones((1, 20)), np.array([0, 1], np.uint32),
                              {'pssm_hard': True}, prediction=True)
    assert result.loss == 54.
    assert result.metrics['af2_pssm_hard'] == 1.
    assert set(result.prediction.native['sequence']) == {'logits', 'pssm', 'hard'}
    with pytest.raises(ValueError, match='pssm_hard'):
        backend.evaluate(np.ones((1, 20)), np.array([0, 1], np.uint32), {'pssm_hard': 1})


def test_native_hard_decode_includes_bias_alpha_and_fixed_positions(backend):
    values = np.zeros((1, 20))
    values[0, 0] = 2.
    backend._sequence_bias[0, 1] = 3.
    assert backend.sequence_from_logits(values) == 'R'
    assert backend.sequence_from_logits(values, {'alpha': 2.}) == 'A'
    backend.sequence = 'NA'
    backend.positions = np.array([1])
    backend._sequence_bias = np.concatenate([np.zeros((1, 20)), backend._sequence_bias])
    assert backend.sequence_from_logits(values) == 'NR'


def test_host_recycles_reuse_native_graphs(backend):
    args = (np.ones((1, 20)), np.array([0, 1], np.uint32))
    backend.evaluate(*args, {'recycles': 1})
    traces = backend.model.traces
    backend.evaluate(*args, {'recycles': 3})
    assert backend.model.traces == traces
    assert backend.metadata['execution'] == 'native_single_pass_host_recycles'


@pytest.mark.parametrize('options', [{'use_initial_guess': True}, {'initial_guess': True},
                                    {'use_initial_atom_pos': True}, {'clear_prev': False}])
def test_unsupported_recycle_state_rejected_before_preparation(monkeypatch, options):
    def forbidden(**kwargs):
        raise AssertionError('Unsupported stateful mode must fail before native construction')
    monkeypatch.setitem(sys.modules, 'colabdesign', SimpleNamespace(mk_afdesign_model=forbidden))
    with pytest.raises(ValueError, match='independent evaluations'):
        AF2(sequence='A', objective=lambda p, c: 0., data_dir='.', model_options=options)


def test_prepare_cannot_silently_enable_unsupported_state(monkeypatch):
    import hallucination_lab.backends.af2 as module
    monkeypatch.setitem(sys.modules, 'colabdesign', SimpleNamespace(mk_afdesign_model=NativeModel))
    monkeypatch.setattr(module, 'version', lambda name: 'mock')
    def prepare(model):
        model._args['use_initial_guess'] = True
    with pytest.raises(ValueError, match='independent evaluations'):
        AF2(sequence='A', objective=lambda p, c: 0., data_dir='.', prepare=prepare)


def test_consistent_score_uses_gradient_graph_and_probe_is_explicit(backend):
    backend._numerical_guard = False
    args = (np.ones((1, 20)), np.array([0, 1], np.uint32))
    expected = backend.evaluate(*args)
    original = backend.model._model['fn']
    def divergent(*args):
        loss, aux = original(*args)
        return loss + 1., aux
    backend.model._model['fn'] = divergent
    score = backend.evaluate(*args, gradient=False)
    assert score.loss == expected.loss
    assert score.gradient is None
    assert backend.probe_forward(*args).loss == expected.loss + 1.
    assert backend.metadata['score_mode'] == 'consistent'


def test_explicit_forward_score_mode(backend):
    backend._score_mode = 'forward'
    args = (np.ones((1, 20)), np.array([0, 1], np.uint32))
    def forbidden(*args):
        raise AssertionError('Explicit forward mode must not run backward')
    backend.model._model['grad_fn'] = forbidden
    assert backend.evaluate(*args, gradient=False).loss == 40.


def test_guard_refuses_graph_mismatch_without_changing_input(backend):
    from hallucination_lab.numerics import NumericalAuditError
    x = np.ones((1, 20))
    key = np.array([0, 1], np.uint32)
    original = backend.model._model['fn']
    def divergent(*args):
        loss, aux = original(*args)
        return loss + 1., aux
    backend.model._model['fn'] = divergent
    with pytest.raises(NumericalAuditError) as exc:
        backend.evaluate(x, key)
    assert not exc.value.report['passed']
    assert exc.value.report['key'] == [0, 1]
    assert exc.value.evaluation.loss == 40.
    np.testing.assert_array_equal(exc.value.input_logits, x)
    assert exc.value.input_logits is not x
    np.testing.assert_array_equal(x, np.ones((1, 20)))
    backend._numerical_guard = False
    assert backend.evaluate(x, key).loss == 40.


def test_forward_mode_requires_explicit_guard_opt_out():
    with pytest.raises(ValueError, match='numerical_guard=False'):
        AF2(sequence='A', objective=lambda p, c: 0., data_dir='.', score_mode='forward')


def test_guard_checks_the_same_final_state_and_key_without_extra_recycle(backend):
    calls = []
    for mode in ('fn', 'grad_fn'):
        original = backend.model._model[mode]
        def wrapped(params, weights, batch, key, original=original, mode=mode):
            calls.append((mode, batch['prev'], np.asarray(key)))
            return original(params, weights, batch, key)
        backend.model._model[mode] = wrapped
    backend.evaluate(np.ones((1, 20)), np.array([0, 1], np.uint32), {'recycles': 1})
    assert [c[0] for c in calls] == ['fn', 'grad_fn', 'fn']
    assert calls[1][1] is calls[2][1]
    np.testing.assert_array_equal(calls[1][2], calls[2][2])
