"""Optional algorithm examples using only the public backend contract."""
import numpy as np
from .engine import StepResult


def _identities(backend, logits, controls):
    """Native hard identities, including every member of a composite model."""
    from .backends.composite import Composite
    from .sequence import decode
    if isinstance(backend, Composite):
        return tuple(identity for member in backend.backends.values()
                     for identity in _identities(member, logits, controls))
    decoder = getattr(backend, "sequence_from_logits", None)
    identity = decoder(logits, controls) if decoder is not None else decode(logits)
    if not isinstance(identity, str) or not identity:
        raise ValueError("Mutation requires a nonempty decoded sequence")
    return (identity,)


def mutation_step(state, backend, controls, key):
    """Greedy single-position mutation proposals scored with the same model key.

    Uses no gradients and can therefore optimize forward-only caller-defined
    scores via a custom backend. trials is controlled by controls['trials'].
    The entire evaluated design sequence is hard one-hot; fixed backend positions
    remain fixed. Native decoders must preserve identities under one-hot
    conversion and map each proposal to exactly one residue change. Unsupported
    encodings (for example, a conflicting prepared bias) are rejected, not
    inverted or bypassed. Without a decoder, raw categorical identity is assumed.
    This is a reference algorithm, not a performance claim.
    """
    trials = controls.get("trials", 4)
    if isinstance(trials, (bool, np.bool_)) or not isinstance(trials, (int, np.integer)) or trials < 1:
        raise ValueError("trials must be a positive integer")
    rng = np.random.default_rng(np.random.SeedSequence(np.asarray(key).tolist()))
    hard = np.eye(20, dtype=np.float32)[state.logits.argmax(-1)]
    controls = {**controls, "soft": 1.0, "hard": 1.0}
    original = _identities(backend, state.logits, controls)
    if _identities(backend, hard, controls) != original:
        raise ValueError("mutation_step one-hot conversion changes native sequence identity; "
                         "use a step that supports this backend's sequence encoding/bias")
    baseline = backend.evaluate(hard, key, controls, gradient=False)
    best_logits, best = hard, baseline
    for _ in range(trials):
        proposal = hard.copy()
        index = rng.integers(len(proposal))
        old = int(proposal[index].argmax())
        residue = int(rng.integers(19))
        residue += residue >= old
        proposal[index] = np.eye(20, dtype=np.float32)[residue]
        identities = _identities(backend, proposal, controls)
        if len(identities) != len(original) or any(
                len(a) != len(b) or sum(x != y for x, y in zip(a, b)) != 1
                for a, b in zip(original, identities)):
            raise ValueError("mutation_step proposal is not one native residue change; "
                             "use a step that supports this backend's sequence encoding/bias")
        result = backend.evaluate(proposal, key, controls, gradient=False)
        if result.loss < best.loss:
            best_logits, best = proposal, result
    return StepResult(best_logits, best.loss,
                      {**best.metrics, "accepted": float(best.loss < baseline.loss),
                       "evaluations": float(trials + 1)}, best_logits.copy(), {}, dict(state.extra),
                      effective_controls=controls)
