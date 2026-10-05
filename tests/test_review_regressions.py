"""Independent-review regressions: failure isolation and truthful contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from hallucination_lab import Run, Stage, initial_logits
from hallucination_lab.engine import StepResult
from hallucination_lab.backends.common import design_indices
from hallucination_lab.steps import mutation_step


def make_run(backend, **options):
    return Run(backend, initial_logits(2), [Stage('test', 2)],
               problem_id='review-regressions', **options)


@pytest.mark.parametrize('failure', ['exception', 'nan', 'overflow', 'unserializable'])
def test_failed_custom_step_is_transactional(quadratic, failure):
    def bad(state, backend, controls, key):
        state.logits += 1
        state.optimizer['nested'][0] = 999
        state.extra['nested']['value'] = 999
        if failure == 'exception':
            raise RuntimeError('custom failure')
        result = StepResult(state.logits, 0., {}, state.logits,
                            state.optimizer, state.extra)
        if failure == 'nan':
            result.loss = float('nan')
        elif failure == 'overflow':
            result.logits = np.full((2, 20), 1e100)
        else:
            result.extra['opaque'] = object()
        return result
    run = make_run(quadratic, step_fn=bad)
    run.state.optimizer = {'nested': np.array([1])}
    run.state.extra = {'nested': {'value': 1}}
    before = run.state.logits.copy()
    with pytest.raises((RuntimeError, FloatingPointError, TypeError)):
        run.run()
    assert run.state.step == 0
    np.testing.assert_array_equal(run.state.logits, before)
    assert run.state.optimizer['nested'][0] == 1
    assert run.state.extra['nested']['value'] == 1


def test_failed_restore_preserves_state_and_log(quadratic, tmp_path):
    make_run(quadratic, output=tmp_path).run(max_steps=1)
    trajectory = tmp_path / 'trajectory.jsonl'
    malformed = 'broken\n{}\n'
    trajectory.write_text(malformed)
    run = make_run(quadratic, output=tmp_path)
    before = run.state.logits.copy()
    with pytest.raises(ValueError, match='Malformed'):
        run.restore(tmp_path / 'checkpoint.npz')
    assert run.state.step == 0 and not run._restored
    np.testing.assert_array_equal(run.state.logits, before)
    assert trajectory.read_text() == malformed


def test_mutation_logs_actual_controls(quadratic, tmp_path):
    run = Run(quadratic, initial_logits(2),
              [Stage('mutation', 1, controls={'soft': 0., 'hard': 0.})],
              problem_id='mutation-controls', step_fn=mutation_step, output=tmp_path)
    run.run()
    row = json.loads((tmp_path / 'trajectory.jsonl').read_text())
    assert row['controls']['hard'] == row['controls']['soft'] == 1.
    assert row['requested_controls']['hard'] == 0.
    actual = quadratic.evaluate(run.state.logits, np.array([0, 0], np.uint32),
                                row['controls'], gradient=False)
    assert actual.loss == row['loss']


@pytest.mark.parametrize('value', [[.9], [-.1], [True], [False, 2], [True, 2], [0, 0], [2**32], [], [[0]], ['0']])
def test_design_indices_reject_lossy_or_invalid_values(value):
    with pytest.raises(ValueError, match='integer positions'):
        design_indices(value, 3)


def test_design_indices_preserve_order():
    np.testing.assert_array_equal(design_indices([2, 0], 3), [2, 0])
    np.testing.assert_array_equal(design_indices(None, 3), [0, 1, 2])


def test_full_af3_rejects_dropout_before_model_call():
    from hallucination_lab.backends.af3 import AF3
    backend = AF3.__new__(AF3)
    backend._matmul_precision = "highest"
    backend.design_length = 2
    with pytest.raises(ValueError, match='dropout'):
        backend.full_prediction(initial_logits(2), np.array([0, 0], np.uint32),
                                {'dropout': True})


def test_topk_does_not_drop_unsatisfied_valid_rows():
    import jax.numpy as jnp
    from hallucination_lab import Prediction
    from hallucination_lab.losses import contact
    pred = Prediction(jnp.zeros((4, 4, 3)), jnp.array([4., 8.]),
                      jnp.array([1, 1, 1, 0]), jnp.arange(4), jnp.ones(4),
                      jnp.zeros((4, 20)), {})
    mask = np.zeros((4, 4))
    mask[0, [1, 3]] = 1
    mask[1, [0, 2]] = 1
    loss = contact(mask, [4, 8], cutoff=8, partners=2)
    assert np.isnan(loss(pred, {}))
    # Padding rows themselves do not constitute an unsatisfied constraint.
    mask[0] = 0
    mask[3, [0, 1]] = 1
    assert np.isfinite(contact(mask, [4, 8], cutoff=8, partners=2)(pred, {}))


def test_missing_cli_factory_has_concise_error():
    result = subprocess.run([sys.executable, '-m', 'hallucination_lab.cli',
                             'run', 'math:missing_factory'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Factory' in result.stderr and 'Traceback' not in result.stderr


@pytest.mark.parametrize('factory', ['math', 'math:sin:extra', ':sin', 'math:'])
def test_invalid_cli_factory_format_has_actionable_error(factory):
    result = subprocess.run([sys.executable, '-m', 'hallucination_lab.cli',
                             'run', factory], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'module:function' in result.stderr and 'unpack' not in result.stderr
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize(('example', 'expected'), [
    ('af2:build', 'AF2 example config requires: data_dir'),
    ('af3:build', 'AF3 example config requires: input_json, pair_mask, model_dir, design_chain'),
])
def test_example_missing_config_is_actionable(example, expected):
    root = Path(__file__).parents[1]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), str(root / 'examples')])
    result = subprocess.run([sys.executable, '-m', 'hallucination_lab.cli',
                             'run', example, '--max-steps', '0'],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert expected in result.stderr and 'Traceback' not in result.stderr


def test_installed_numerical_toy_runs_without_examples_path(tmp_path):
    from hallucination_lab.toy import build
    run = build({'output': str(tmp_path)})
    run.run(max_steps=1)
    assert run.state.step == 1
    assert (tmp_path / 'checkpoint.npz').exists()
