"""A single declared residue order and shared sequence relaxation."""
import numpy as np

ALPHABET = "ARNDCQEGHILKMFPSTWYV"


def initial_logits(length=None, *, sequence=None, seed=0, scale=0.1):
    if sequence is not None:
        if not sequence or any(a not in ALPHABET for a in sequence):
            raise ValueError("sequence must contain standard uppercase amino acids")
        if length is not None and length != len(sequence):
            raise ValueError("length and sequence disagree")
        return np.eye(20, dtype=np.float32)[[ALPHABET.index(a) for a in sequence]]
    if not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be positive and finite")
    return np.random.default_rng(seed).normal(0, scale, (length, 20)).astype(np.float32)


def decode(logits):
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
        raise ValueError("expected finite [D,20] logits")
    return "".join(ALPHABET[i] for i in values.argmax(-1))


def relax(logits, controls):
    """Raw/soft/straight-through mixture (ColabDesign semantics; see NOTICE)."""
    import jax
    import jax.numpy as jnp
    temperature = controls.get("temperature", 1.0)
    soft = controls.get("soft", 1.0)
    hard = controls.get("hard", 0.0)
    scaled = logits * controls.get("alpha", 1.0)
    probabilities = jax.nn.softmax(scaled / temperature, axis=-1)
    onehot = jax.nn.one_hot(probabilities.argmax(-1), 20)
    straight_through = probabilities + jax.lax.stop_gradient(onehot - probabilities)
    pseudo = (1 - soft) * logits + soft * probabilities
    return jnp.asarray((1 - hard) * pseudo + hard * straight_through)


def validate_controls(controls):
    c = {"temperature": 1.0, "soft": 1.0, "hard": 0.0, "alpha": 1.0,
         "dropout": False, **(controls or {})}
    for name in ("temperature", "alpha"):
        if not np.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("soft", "hard"):
        if not np.isfinite(c[name]) or not 0 <= c[name] <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    return c
