"""A categorical step must not silently alter native sequence identities."""
import numpy as np
import pytest
from hallucination_lab.backends.af2 import AF2
from hallucination_lab.backends.composite import Composite
from hallucination_lab.engine import State
from hallucination_lab.steps import mutation_step
from hallucination_lab.types import Evaluation


def native_decoder_backend(bias):
    # Use the actual AF2 decoder and partial-design mapping; no model runtime.
    backend = AF2.__new__(AF2)
    backend.design_length = 2
    backend.positions = np.array([0, 2])
    backend.sequence = "ACA"
    backend._sequence_bias = np.zeros((3, 20), np.float32)
    backend._sequence_bias[backend.positions, 1] = bias
    backend.metadata = {"backend": "decoder-test"}
    calls = []

    def score(logits, key, controls, *, gradient):
        assert gradient is False
        calls.append(backend.sequence_from_logits(logits, controls))
        return Evaluation(1., {})

    backend.evaluate = score
    return backend, calls


@pytest.mark.parametrize("composite", [False, True])
def test_canonicalization_cannot_change_native_residues(composite):
    backend, calls = native_decoder_backend(.4)
    x = np.zeros((2, 20), np.float32)
    x[:, 0], x[:, 1] = .8, .7
    assert backend.sequence_from_logits(x) == "RCR"
    if composite:
        backend = Composite({"condition": backend}, lambda scores, c: scores["condition"],
                            variable_order=["first", "last"])
    with pytest.raises(ValueError, match="one-hot conversion changes native sequence"):
        mutation_step(State(x), backend, {}, np.array([0, 9], np.uint32))
    assert calls == []  # Reject the changed baseline before model scoring.
    np.testing.assert_array_equal(x[:, :2], np.tile([.8, .7], (2, 1)).astype(np.float32))


def test_bias_blocked_proposal_is_not_scored_as_a_mutation():
    backend, calls = native_decoder_backend(2.)
    x = np.eye(20, dtype=np.float32)[[1, 1]]
    with pytest.raises(ValueError, match="not one native residue change"):
        mutation_step(State(x), backend, {"trials": 4}, np.array([0, 9], np.uint32))
    assert calls == ["RCR"]  # Baseline only; every other residue is bias-blocked.


@pytest.mark.parametrize("bias", [0., .2])
def test_compatible_encoding_keeps_native_single_mutation_semantics(bias):
    backend, calls = native_decoder_backend(bias)
    x = np.zeros((2, 20), np.float32)
    x[:, 0] = 3.
    result = mutation_step(State(x), backend, {"trials": 20}, np.array([0, 9], np.uint32))
    assert calls[0] == "ACA"
    assert len(calls) == 21
    for sequence in calls[1:]:
        assert sequence[1] == "C"  # The non-designed position is never touched.
        assert sum(a != b for a, b in zip(sequence, calls[0])) == 1
    assert result.metrics["accepted"] == 0.
    assert backend.sequence_from_logits(result.logits) == "ACA"
