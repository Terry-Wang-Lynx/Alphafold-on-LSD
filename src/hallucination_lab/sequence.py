"""A single declared residue order and shared sequence relaxation."""
import numpy as np

ALPHABET = "ARNDCQEGHILKMFPSTWYV"


def _real_number(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)))


def initial_logits(length=None, *, sequence=None, seed=0, scale=0.1):
    if sequence is not None:
        if not sequence or any(a not in ALPHABET for a in sequence):
            raise ValueError("sequence must contain standard uppercase amino acids")
        if length is not None and (type(length) is not int or length != len(sequence)):
            raise ValueError("length and sequence disagree")
        return np.eye(20, dtype=np.float32)[[ALPHABET.index(a) for a in sequence]]
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer")
    if not _real_number(scale) or scale <= 0 or not np.isfinite(scale):
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
    logits = jnp.asarray(logits)
    gain = jnp.asarray(controls.get("alpha", 1.0), logits.dtype) / temperature
    peak = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    delta = logits - peak
    # An opposite-sign pair can overflow on subtraction even when the scaled
    # difference is modest. For that case only, attenuate before subtracting.
    safe_gain = jnp.minimum(gain, 1.0)
    attenuated = logits * safe_gain - peak * safe_gain
    scaled = jnp.where((gain < 1) & ~jnp.isfinite(delta), attenuated, delta * gain)
    probabilities = jax.nn.softmax(scaled, axis=-1)
    onehot = jax.nn.one_hot(logits.argmax(-1), 20, dtype=logits.dtype)
    straight_through = onehot + (probabilities - jax.lax.stop_gradient(probabilities))
    pseudo = (1 - soft) * logits + soft * probabilities
    return jnp.asarray((1 - hard) * pseudo + hard * straight_through)


def validate_controls(controls):
    c = {"temperature": 1.0, "soft": 1.0, "hard": 0.0, "alpha": 1.0,
         "dropout": False, **(controls or {})}
    for name in ("temperature", "alpha"):
        if not _real_number(c[name]) or not np.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    # Backend forwards and gradients operate in F32. Reject effective scales
    # that would silently flush to zero or infinity when passed through XLA.
    info = np.finfo(np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        scales = np.asarray([c["alpha"], c["temperature"]], np.float32)
        gain = scales[0] / scales[1]
    if not (np.isfinite(scales).all() and (scales >= info.tiny).all()
            and info.tiny <= gain <= info.max):
        raise ValueError("alpha, temperature and their ratio must be representable normal float32")
    for name in ("soft", "hard"):
        if not _real_number(c[name]) or not np.isfinite(c[name]) or not 0 <= c[name] <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if not isinstance(c["dropout"], (bool, np.bool_)):
        raise ValueError("dropout must be boolean")
    return c
