"""Full-chain soft atom probabilities with native AF3 layout forwards."""
import numpy as np


def require_stable_entities(chains, design_chain, positions):
    base = chains[design_chain]
    mutable = np.zeros(len(base), bool)
    mutable[positions] = True
    for chain, sequence in chains.items():
        if (chain != design_chain and len(sequence) == len(base)
                and all(change or a == b for change, a, b in zip(mutable, base, sequence))):
            raise ValueError("Sampled Soft Atom sequences must preserve native chain entities")


def draw_identities(key, log_p, samples):
    import jax

    # The forward model gets its original key for every layout. Domain-separate
    # the categorical draws so the sampled layout and model RNG are independent.
    layout_key = jax.random.fold_in(key, 0x41544F4D)
    return jax.random.categorical(
        layout_key, log_p, axis=-1, shape=(samples, log_p.shape[0]), mode="high",
    )


def mean_gradient(losses, direct, draws, probability, scale):
    """Differentiate E[L(soft tokens, native atoms of S)], S ~ product(p).

    Each direct gradient holds the drawn layouts fixed. The leave-one-out
    baseline is independent of its corresponding draw, so the additional
    likelihood-ratio term estimates the remaining derivative without bias.
    """
    losses = np.asarray(losses, np.float64)
    direct = np.asarray(direct, np.float64).mean(axis=0)
    n = len(losses)
    advantage = (losses - losses.mean()) * (n / (n - 1))
    indicator = np.eye(probability.shape[1], dtype=np.float64)[draws]
    score = indicator - probability
    return direct + scale * np.einsum("s,sia->ia", advantage / n, score)
