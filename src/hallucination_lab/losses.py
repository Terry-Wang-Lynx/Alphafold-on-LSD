"""Distogram loss constructors. No implicit terms, targets, weights, or symmetry.

    objective = weighted({"shape": (1.0, distribution(target, mask, edges=edges)),
                          "contact": (0.2, contact(mask, edges, cutoff=8.0))})

Masks are nonnegative pair weights. The caller chooses directed/undirected
counting. Padding is always removed. Distances are Å. These are mathematical
objectives, not validated proxies for binding affinity or biological function.
"""
import numpy as np


def _mask(value):
    mask = np.asarray(value, np.float32)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("pair mask must be square [N,N]")
    if not np.isfinite(mask).all() or (mask < 0).any() or not (mask > 0).any():
        raise ValueError("pair mask must be finite, nonnegative, and nonempty")
    return mask


def _weights(prediction, mask):
    import jax.numpy as jnp
    n = prediction.logits.shape[0]
    if mask.shape[0] > n:
        raise ValueError("pair mask exceeds prediction token count")
    padded = jnp.pad(jnp.asarray(mask), ((0, n - len(mask)), (0, n - len(mask))))
    valid = prediction.token_mask.astype(padded.dtype)
    return padded * valid[:, None] * valid[None, :]


def _mean(values, weights):
    import jax.numpy as jnp
    # Empty after padding/filtering is an error, not an apparent zero loss.
    return jnp.where(weights.sum() > 0, (values * weights).sum() / weights.sum(), jnp.nan)


def bin_event(edges, *, lower=0.0, upper=np.inf, mode="contained"):
    """Select entire bins inside a range, or bins intersecting it (explicit).

    Bins are [0,e0), [e0,e1), ..., [e_last,inf). The tail is never
    assigned an invented finite distance. Boundaries alone cannot resolve
    probabilities within one bin; contained and overlap bracket that ambiguity.
    """
    edges = np.asarray(edges, np.float32)
    if (edges.ndim != 1 or not len(edges) or not np.isfinite(edges).all()
            or (np.diff(edges) <= 0).any() or edges[0] <= 0):
        raise ValueError("edges must be positive finite strictly increasing boundaries")
    if not 0 <= lower < upper:
        raise ValueError("require 0 <= lower < upper")
    lo, hi = np.r_[0, edges], np.r_[edges, np.inf]
    if mode == "contained":
        selected = (lo >= lower) & (hi <= upper)
    elif mode == "overlap":
        selected = (lo < upper) & (hi > lower)
    else:
        raise ValueError("mode must be contained or overlap")
    if not selected.any():
        raise ValueError("No bins represent this interval; widen it or explicitly use overlap")
    return selected


def distribution(target, pair_mask, *, edges):
    """Masked cross-entropy against a caller-defined target distribution."""
    import jax
    import jax.numpy as jnp
    mask = _mask(pair_mask)
    target = np.asarray(target, np.float32)
    if target.ndim != 3 or target.shape[:2] != mask.shape:
        raise ValueError("target must be [N,N,B], matching pair_mask")
    bin_event(edges)  # Validate the declared target bin boundaries.
    if target.shape[-1] != len(edges) + 1:
        raise ValueError("target bin count must equal len(edges)+1")
    active = mask > 0
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("target probabilities must be finite and nonnegative")
    if not np.allclose(target.sum(-1)[active], 1, atol=1e-5):
        raise ValueError("active target distributions must sum to one")

    def loss(prediction, controls):
        n, _, bins = prediction.logits.shape
        if target.shape[-1] != bins:
            raise ValueError("target bin count differs; rebin explicitly")
        padded = jnp.pad(jnp.asarray(target), ((0, n-len(target)), (0, n-len(target)), (0, 0)))
        same_edges = jnp.all(jnp.isclose(prediction.bin_edges, jnp.asarray(edges), atol=1e-6))
        log_prob = jax.nn.log_softmax(prediction.logits, -1)
        # Zero target mass contributes zero even if a finite-logit difference overflows.
        score = _mean(-jnp.sum(padded * jnp.where(padded > 0, log_prob, 0), -1),
                      _weights(prediction, mask))
        return jnp.where(same_edges, score, jnp.nan)
    return loss


def interval(pair_mask, edges, *, lower=0.0, upper=np.inf, mode="contained",
             exclude=False):
    """Negative log event probability, or its complement (exclude=True)."""
    import jax
    import jax.numpy as jnp
    mask, selected = _mask(pair_mask), bin_event(edges, lower=lower, upper=upper, mode=mode)
    reference_edges = jnp.asarray(edges)

    def loss(prediction, controls):
        if prediction.logits.shape[-1] != len(selected):
            raise ValueError("bin count differs from event definition")
        same_edges = jnp.all(jnp.isclose(prediction.bin_edges, reference_edges, atol=1e-6))
        event = ~selected if exclude else selected
        log_prob = jax.nn.log_softmax(prediction.logits, -1)
        # logsumexp avoids zero-probability clipping and its vanishing gradient.
        log_mass = jax.scipy.special.logsumexp(jnp.where(event, log_prob, -jnp.inf), axis=-1)
        return jnp.where(same_edges, _mean(-log_mass, _weights(prediction, mask)), jnp.nan)
    if exclude and selected.all():
        raise ValueError("Cannot exclude every distance bin")
    return loss


def contact(pair_mask, edges, *, cutoff=8.0, partners=None, mode="contained"):
    """Optimize all eligible pairs, or each row's best k eligible partners.

    partners=None means weighted mean over all pairs. Integer k means each
    active row needs at least k candidate partners; selection can change during
    optimization. This is contact probability, not physical interaction energy.
    """
    import jax
    import jax.numpy as jnp
    mask = _mask(pair_mask)
    selected = bin_event(edges, upper=cutoff, mode=mode)
    if partners is None:
        return interval(mask, edges, upper=cutoff, mode=mode)
    if not isinstance(partners, int) or partners < 1:
        raise ValueError("partners must be a positive integer")
    counts = (mask > 0).sum(-1)
    if (counts[counts > 0] < partners).any():
        raise ValueError("Every active row needs at least partners eligible pairs")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("top-k contact requires a binary mask")

    def loss(prediction, controls):
        if prediction.logits.shape[-1] != len(selected):
            raise ValueError("bin count differs from contact definition")
        log_prob = jax.nn.log_softmax(prediction.logits, -1)
        log_mass = jax.scipy.special.logsumexp(jnp.where(selected, log_prob, -jnp.inf), -1)
        weights = _weights(prediction, mask)
        values, _ = jax.lax.top_k(jnp.where(weights > 0, log_mass, -jnp.inf), partners)
        active = jnp.pad(jnp.asarray(counts > 0),
                         (0, prediction.logits.shape[0] - len(counts)))
        active = active & prediction.token_mask.astype(bool)
        enough = weights.sum(-1) >= partners
        feasible = jnp.all(~active | enough)
        valid = active & enough
        row_loss = jnp.where(valid, -jnp.mean(values, -1), 0)
        same_edges = jnp.all(jnp.isclose(prediction.bin_edges, jnp.asarray(edges), atol=1e-6))
        return jnp.where(same_edges & feasible, _mean(row_loss, valid), jnp.nan)
    return loss


def weighted(terms):
    """Combine named (weight, callable) terms. Override via controls[name].

    Each term callable takes (Prediction, controls) and returns a scalar.
    Metrics report unweighted terms so weight schedules remain interpretable.
    """
    if not terms or any(not isinstance(k, str) for k in terms):
        raise ValueError("provide named objective terms")
    if any(not np.isfinite(weight) for weight, _ in terms.values()):
        raise ValueError("weights must be finite")

    def objective(prediction, controls):
        metrics = {name: function(prediction, controls) for name, (_, function) in terms.items()}
        total = sum(controls.get(name, weight) * metrics[name]
                    for name, (weight, _) in terms.items())
        return total, metrics
    return objective
