from functools import partial
import json
import numpy as np
import pytest
from hallucination_lab import Run, Stage, initial_logits, gradient_step
from hallucination_lab.steps import mutation_step
from hallucination_lab.backends.composite import Composite


def make_run(backend, **kwargs):
    return Run(backend, initial_logits(2, seed=11),
               [Stage("warm", 3, 0.05), Stage("cool", 3, 0.01, {"temperature": 1.}, {"temperature": 0.2})],
               problem_id="test-v1", seed=9, **kwargs)


def test_gradient_and_loss_improve(quadratic, key):
    run = make_run(quadratic)
    before = quadratic.evaluate(run.state.logits, key).loss
    run.run()
    assert quadratic.evaluate(run.state.logits, key).loss < before
    assert run.state.step == 6


def test_checkpoint_resume_equivalence(quadratic, tmp_path):
    original = make_run(quadratic)
    original.run()
    interrupted = make_run(quadratic, output=tmp_path)
    interrupted.run(max_steps=2)
    restored = make_run(quadratic, output=tmp_path).restore(tmp_path / "checkpoint.npz")
    restored.run()
    np.testing.assert_array_equal(original.state.logits, restored.state.logits)
    np.testing.assert_array_equal(original.state.optimizer["first"], restored.state.optimizer["first"])
    rows = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(6))


def test_checkpoint_rejects_different_experiment(quadratic, tmp_path):
    run = make_run(quadratic)
    run.save(tmp_path / "state.npz")
    different = Run(quadratic, run.state.logits, run.stages, problem_id="different", seed=9)
    with pytest.raises(ValueError, match="fingerprint"):
        different.restore(tmp_path / "state.npz")


def test_failed_incompatible_restore_does_not_reserve_output(quadratic, tmp_path):
    stages = [Stage("warm", 3, 0.05),
              Stage("cool", 3, 0.01, {"temperature": 1.}, {"temperature": 0.2})]
    checkpoint = tmp_path / "other.npz"
    Run(quadratic, initial_logits(2, seed=11), stages,
        problem_id="other", seed=9).save(checkpoint)
    output = tmp_path / "attempt"

    attempted = make_run(quadratic, output=output)
    assert not output.exists()
    with pytest.raises(ValueError, match="fingerprint"):
        attempted.restore(checkpoint)
    assert not (output / "manifest.json").exists()

    corrected = Run(quadratic, initial_logits(2, seed=11), stages,
                    problem_id="other", seed=9, output=output)
    corrected.restore(checkpoint).run(max_steps=0)
    assert (output / "manifest.json").exists()


def test_observer_stops_after_committed_step(quadratic):
    run = make_run(quadratic, observers=[lambda run, row: False])
    run.run()
    assert run.state.step == 1


def test_best_sequence_is_evaluated_not_updated(quadratic, key):
    run = make_run(quadratic)
    before = run.state.logits.copy()
    run.run(max_steps=1)
    np.testing.assert_array_equal(run.state.best_logits, before)
    assert not np.array_equal(run.state.logits, before)
    assert run.state.best_loss == quadratic.evaluate(before, key).loss


def test_user_step_and_gradient_transform(quadratic):
    def zero(gradient, state, controls):
        return gradient * 0
    run = make_run(quadratic, step_fn=partial(gradient_step, optimizer="sgd", transform=zero))
    before = run.state.logits.copy()
    run.run()
    np.testing.assert_array_equal(before, run.state.logits)


def test_composite_chain_rule(quadratic, key):
    composite = Composite({"a": quadratic, "b": quadratic},
                          lambda scores, controls: scores["a"]**2 + 3*scores["b"],
                          variable_order=["x", "y"])
    logits = initial_logits(2, seed=4)
    base = quadratic.evaluate(logits, key)
    result = composite.evaluate(logits, key)
    np.testing.assert_allclose(result.loss, base.loss**2 + 3*base.loss, rtol=1e-6)
    np.testing.assert_allclose(result.gradient, (2*base.loss+3)*base.gradient, rtol=1e-6)


def test_mutation_algorithm_and_resume(quadratic, tmp_path):
    run = make_run(quadratic, step_fn=mutation_step)
    run.run(max_steps=2)
    run.save(tmp_path / "mutation.npz")
    resumed = make_run(quadratic, step_fn=mutation_step).restore(tmp_path / "mutation.npz")
    run.run()
    resumed.run()
    np.testing.assert_array_equal(run.state.logits, resumed.state.logits)


def test_invalid_gradient_does_not_advance(quadratic):
    def invalid(g, state, controls):
        return g * np.nan
    run = make_run(quadratic, step_fn=partial(gradient_step, transform=invalid))
    before = run.state.logits.copy()
    with pytest.raises(FloatingPointError):
        run.run()
    assert run.state.step == 0
    np.testing.assert_array_equal(before, run.state.logits)


def test_ramp_endpoints():
    stage = Stage("test", 3, controls={"temperature": 1}, end={"temperature": 0.1})
    assert stage.at(0)["temperature"] == 1
    assert stage.at(2)["temperature"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        Stage("test", 0)


def test_stage_algorithm_dispatch(quadratic):
    calls = []
    def first(state, backend, controls, key):
        calls.append("a")
        return gradient_step(state, backend, controls, key)
    def second(state, backend, controls, key):
        calls.append("b")
        return gradient_step(state, backend, controls, key)
    run = Run(quadratic, initial_logits(2), [Stage("a", 1), Stage("b", 1)],
              problem_id="dispatch", step_fn={"a": first, "b": second})
    run.run()
    assert calls == ["a", "b"]


def test_output_reuse_requires_resume(quadratic, tmp_path):
    make_run(quadratic, output=tmp_path).run(max_steps=1)
    with pytest.raises(FileExistsError, match="restore"):
        make_run(quadratic, output=tmp_path).run()


def test_partial_trajectory_record_recovered(quadratic, tmp_path):
    make_run(quadratic, output=tmp_path).run(max_steps=2)
    with (tmp_path / "trajectory.jsonl").open("a") as handle:
        handle.write('{"step":2')
    run = make_run(quadratic, output=tmp_path).restore(tmp_path / "checkpoint.npz")
    run.run()
    lines = (tmp_path / "trajectory.jsonl").read_text().splitlines()
    assert len(lines) == 6
    assert all(json.loads(line)["step"] == i for i, line in enumerate(lines))


def test_tuple_metadata_normalized_on_resume(quadratic, tmp_path):
    quadratic.metadata["tuple_config"] = ((1, 2), (None, 4))
    make_run(quadratic, output=tmp_path).run(max_steps=1)
    resumed = make_run(quadratic, output=tmp_path).restore(tmp_path / "checkpoint.npz")
    resumed.run()
    assert resumed.state.step == 6


def test_generator_schedule_captured_and_external_mutation_isolated(quadratic):
    stage = Stage("a", 2, controls={"temperature": 1.0})
    run = Run(quadratic, initial_logits(2), (s for s in [stage]), problem_id="generator")
    stage.controls["temperature"] = 0.01
    assert len(run.manifest["stages"]) == 1
    assert run.controls_at(0)[1]["temperature"] == 1.0
