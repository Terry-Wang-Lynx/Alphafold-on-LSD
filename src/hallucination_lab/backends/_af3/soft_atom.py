"""Finite-layout continuous reference atom encoding for AF3 3.0.3.

This module targets the input atom encoder only. Static native layout and
chemical features are built offline. Every train-time routing weight comes
from the joint sequence distribution, without decoding a hard carrier.
"""

from collections.abc import Mapping, Sequence
from itertools import product
from math import prod

import numpy as np

from ...sequence import ALPHABET
from .mapping import protein_tokens, token_identity
from .query_only import audit_query_only_batch, require_query_only_input


MAX_LAYOUTS = 27
NATIVE_ENCODER_SHA256 = "bb69e504f38bf7d1b25f5e89a86b31493aaa962335f71a0292d91dd4bed873e0"


def f32_logits(logits):
    """Use the same F32 tie semantics on the host and on XLA devices."""
    values = np.asarray(logits, dtype=np.float32)
    # XLA devices can flush subnormals. Normalize at the host boundary so
    # ties agree without masking the derivative at a zero design logit.
    return np.where(np.abs(values) < np.finfo(np.float32).tiny, np.float32(0), values)


def f32_controls(controls):
    """Reject positive controls whose F32 representation would break scaling."""
    limits = np.finfo(np.float32)
    lower, upper = float(limits.tiny), float(limits.max)
    alpha, temperature = float(controls["alpha"]), float(controls["temperature"])
    if not all(lower <= x <= upper for x in (alpha, temperature)):
        raise ValueError("Soft Atom alpha and temperature must be normal finite F32 values")
    alpha, temperature = np.float32(alpha), np.float32(temperature)
    if not lower <= float(alpha) / float(temperature) <= upper:
        raise ValueError("Soft Atom alpha / temperature must be a normal finite F32 value")
    return {**controls, "alpha": alpha, "temperature": temperature}


def input_chains(data):
    """Read canonical protein chains and verify the no-template/query-only contract."""
    require_query_only_input(data)
    chains = {}
    if data.get("bondedAtomPairs"):
        raise ValueError("Soft Atom mode requires protein inputs without explicit covalent bonds")
    for entry in data["sequences"]:
        if set(entry) != {"protein"} or entry["protein"].get("modifications"):
            raise ValueError("Soft Atom mode requires unmodified canonical protein chains")
        protein = entry["protein"]
        sequence = protein["sequence"]
        ids = protein["id"]
        if isinstance(ids, str):
            ids = [ids]
        if not ids or not sequence or set(sequence) - set(ALPHABET):
            raise ValueError("Soft Atom mode requires canonical protein chains with IDs")
        for chain in ids:
            if not isinstance(chain, str) or not chain or chain in chains:
                raise ValueError("Soft Atom mode requires distinct protein chain IDs")
            chains[chain] = sequence
    return chains


def candidate_layouts(sequence, positions, alphabets):
    """Enumerate the explicit local AA search space and its train-time mask."""
    if (not isinstance(alphabets, (list, tuple)) or len(alphabets) != len(positions)
            or any(not isinstance(a, str) or not a or set(a) - set(ALPHABET)
                   or len(set(a)) != len(a) for a in alphabets)):
        raise ValueError("Provide one nonempty canonical soft_atom_alphabets string per design position")
    if prod(map(len, alphabets)) > MAX_LAYOUTS:
        raise ValueError(f"Soft Atom mode supports at most {MAX_LAYOUTS} native layouts per run")
    choices = tuple(product(*alphabets))
    indices = np.asarray([[ALPHABET.index(a) for a in choice] for choice in choices], np.int32)
    allowed = np.zeros((len(positions), len(ALPHABET)), bool)
    allowed[np.arange(len(positions))[None, :], indices] = True
    sequences = []
    for choice in choices:
        candidate = list(sequence)
        for position, amino_acid in zip(positions, choice, strict=True):
            candidate[position] = amino_acid
        sequences.append("".join(candidate))
    return tuple(sequences), indices, allowed


def require_stable_entities(chains, design_chain, sequences):
    """Reject mutations that split or merge native sequence-grouped entities."""
    base = chains[design_chain]
    for chain, other in chains.items():
        if chain != design_chain and any((seq == other) != (base == other) for seq in sequences):
            raise ValueError("Soft Atom layouts must preserve native chain entity assignments")


def assemble_bank(raw_reference, raw_batches, chains_by_layout, design_tokens):
    """Use native token order and check all unchanged trunk conditioning."""
    tokens = np.flatnonzero(raw_reference["seq_mask"])
    chain_ids, residues = token_identity(raw_reference)
    unaffected = np.ones(raw_reference["seq_mask"].shape, bool)
    unaffected[design_tokens] = False
    token_order = np.flatnonzero(unaffected)
    fixed = {"token_index", "residue_index", "seq_mask", "asym_id", "entity_id", "sym_id",
             "msa_mask", "deletion_matrix", "deletion_mean", "is_protein", "is_rna", "is_dna",
             "is_ligand", "is_nonstandard_polymer_chain", "is_water", "seq_length"}
    sequences = []
    for batch, by_chain in zip(raw_batches, chains_by_layout, strict=True):
        actual_chains, actual_residues = token_identity(batch)
        if not np.array_equal(actual_chains, chain_ids) or not np.array_equal(actual_residues, residues):
            raise ValueError("Soft Atom candidate changes native token order")
        keys = fixed | {key for key in raw_reference.keys() | batch.keys()
                        if key.startswith("template_") or "bond" in key}
        for name in sorted(keys):
            if name not in raw_reference or name not in batch or not np.array_equal(raw_reference[name], batch[name]):
                raise ValueError(f"Soft Atom candidate changes static AF3 conditioning: {name}")
        for name in ("aatype", "msa", "profile"):
            axis = 1 if name == "msa" else 0
            if not np.array_equal(np.take(raw_reference[name], token_order, axis=axis),
                                  np.take(batch[name], token_order, axis=axis)):
                raise ValueError(f"Soft Atom candidate changes conditioning outside its design mask: {name}")
        if len(tokens) != sum(map(len, by_chain.values())):
            raise ValueError("Soft Atom bank requires one native token per amino acid")
        for chain, sequence in by_chain.items():
            indices = protein_tokens(chain_ids, residues, chain, len(sequence))
            audit_query_only_batch(batch, indices, sequence)
        sequences.append("".join(by_chain[chain_ids[t]][residues[t] - 1] for t in tokens))
    return build_atom_bank(raw_batches, sequences)


def relax_atom_sequence(logits, controls, allowed):
    """A common log probability for both input channels, with a read-only hard endpoint."""
    import jax
    import jax.numpy as jnp

    masked = jnp.where(allowed, logits, -jnp.inf)
    alpha = jnp.asarray(controls["alpha"], dtype=jnp.float32)
    temperature = jnp.asarray(controls["temperature"], dtype=jnp.float32)
    center = jnp.max(masked, axis=-1, keepdims=True)
    scores = jnp.where(allowed, (logits - center) * (alpha / temperature), -jnp.inf)
    log_p = jax.nn.log_softmax(scores, axis=-1)
    hard = jax.nn.one_hot(masked.argmax(axis=-1), len(ALPHABET), dtype=bool)
    log_p = jnp.where(controls["hard"] == 1, jnp.where(hard, 0.0, -jnp.inf), log_p)
    return jnp.exp(log_p), log_p


_REF_KEYS = ("ref_pos", "ref_mask", "ref_element", "ref_charge",
             "ref_atom_name_chars", "ref_space_uid")


def _atom_layout_names(batch: Mapping) -> np.ndarray:
    layout = batch["token_atoms_layout"]
    if isinstance(layout, np.ndarray) and layout.ndim == 0:
        layout = layout.item()
    return np.asarray(layout.atom_name, dtype=str)


def _native_edges(
    batch: Mapping, real_atoms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the native flattened query/key membership from actual gathers."""
    qids = np.asarray(batch["token_atoms_to_queries:gather_idxs"])
    qmask = np.asarray(batch["token_atoms_to_queries:gather_mask"], dtype=bool)
    kids = np.asarray(batch["queries_to_keys:gather_idxs"])
    kmask = np.asarray(batch["queries_to_keys:gather_mask"], dtype=bool)
    if qids.ndim != 2 or kids.ndim != 2 or qids.shape[0] != kids.shape[0]:
        raise ValueError("Expected native two-dimensional atom attention windows")
    if qids.shape != qmask.shape or kids.shape != kmask.shape:
        raise ValueError("Native atom attention gather indices and masks disagree")
    if ((qids < 0).any() or (qids >= real_atoms.size).any()
            or (kids < 0).any() or (kids >= qids.size).any()):
        raise ValueError("Native atom attention gathers point outside their source layouts")

    # A padding UID may still have a matching gather. The actual native query
    # mask also consults pred_dense_atom_mask.
    qmask = qmask & real_atoms[qids]
    nreal = int(real_atoms.sum())
    query_rows = qmask.any(axis=1)
    # The pinned native attention uses an unusual additive mask. With real
    # queries and full real key windows the bias is exactly zero. Qualify only
    # this domain until padded-key attention has its own endpoint test.
    if not np.all(kmask[query_rows]):
        raise ValueError(
            "Soft atom qualification requires complete native key windows for real queries"
        )
    hard_qids = qids[qmask]
    if (hard_qids.size != nreal or (hard_qids < 0).any()
            or not np.array_equal(np.sort(hard_qids), np.flatnonzero(real_atoms))):
        raise ValueError("Each real AF3 atom must be present as an attention query exactly once")
    selected_keys = kids[query_rows]
    if (selected_keys < 0).any() or (selected_keys >= qids.size).any():
        raise ValueError("Native key gather outside the query layout")
    if not np.all(qmask.ravel()[selected_keys]):
        raise ValueError("Real native queries attend non-real key slots")
    if np.any(np.diff(np.sort(selected_keys, axis=-1), axis=-1) == 0):
        raise ValueError("Unexpected duplicate key in a native atom attention window")

    rows, _ = np.nonzero(qmask)
    repeat_count = kids.shape[1]
    key_query_slots = kids[rows].ravel()
    return np.repeat(hard_qids, repeat_count), qids.ravel()[key_query_slots], key_query_slots


def build_atom_bank(raw_batches: Sequence[Mapping], sequences: Sequence[str]) -> dict[str, np.ndarray]:
    """Build a numeric PyTree from ordered, native AF3 feature dictionaries.

    Requires equal-length canonical token sequences with a stable token map.
    Slot identities retain the amino-acid variant, so competing side chains
    remain mutually exclusive even when both have positive probability.
    """
    if not raw_batches or len(raw_batches) != len(sequences):
        raise ValueError("Give one raw native AF3 batch per hard sequence")
    if len(set(sequences)) != len(sequences) or any(
        len(sequence) != len(sequences[0]) or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
        for sequence in sequences
    ):
        raise ValueError("Expected distinct equal-length canonical protein sequences")
    nres = len(sequences[0])
    seq_mask = np.asarray(raw_batches[0]["seq_mask"], dtype=bool)
    token_ids = np.flatnonzero(seq_mask)
    if seq_mask.ndim != 1 or token_ids.size != nres or nres == 0:
        raise ValueError("Expected one active token per canonical protein residue")

    layouts, old_edges, all_atom_keys = [], [], set()
    identities, variant_counts = {}, {}
    for batch, sequence in zip(raw_batches, sequences, strict=True):
        names = _atom_layout_names(batch)
        real = np.asarray(batch["pred_dense_atom_mask"], dtype=bool)
        if (names.shape != real.shape or real.ndim != 2 or real.shape[0] != seq_mask.size
                or not np.array_equal(real, names != "")
                or not np.array_equal(np.asarray(batch["seq_mask"], dtype=bool), seq_mask)
                or not np.array_equal(real.any(axis=1), seq_mask)):
            raise ValueError("Inconsistent AF3 protein token and dense atom layouts")
        tok, dense = np.nonzero(real)
        flat_ids = np.flatnonzero(real)
        token_counts = real.sum(axis=1)
        for token, aa in zip(token_ids, sequence, strict=True):
            identity = (int(token), aa)
            size = int(token_counts[token])
            if variant_counts.setdefault(identity, size) != size:
                raise ValueError("Atom count changes across native states of the same residue identity")
        # Dense index preserves the native order inside every residue.
        keys = [(int(t), sequence[i], int(d))
                for t, d, i in zip(tok, dense, np.searchsorted(token_ids, tok), strict=True)]
        for key, label in zip(keys, names[tok, dense], strict=True):
            if key in identities and identities[key] != label:
                raise ValueError("Atom names change across native states of the same residue identity")
            identities[key] = label
        all_atom_keys.update(keys)
        layouts.append((real.shape, flat_ids, keys))
        old_edges.append(_native_edges(batch, real.ravel()))

    all_atom_keys = sorted(all_atom_keys)
    slot_by_key = {key: n for n, key in enumerate(all_atom_keys)}
    nslots = len(all_atom_keys)
    all_edges = []
    native_key_uid = []
    branch_slots = []
    for batch, layout, edge_data in zip(raw_batches, layouts, old_edges, strict=True):
        dense_shape, flat_ids, keys = layout
        queries, keys_flat, key_query_slots = edge_data
        slots = np.fromiter((slot_by_key[key] for key in keys), dtype=np.int32, count=len(keys))
        branch_slots.append(slots)
        native_to_union = np.full(np.prod(dense_shape), -1, dtype=np.int32)
        native_to_union[flat_ids] = slots
        # int64 for the flattened map even though the individual slots are int32.
        edges = native_to_union[queries].astype(np.int64) * nslots + native_to_union[keys_flat]
        order = np.argsort(edges)
        all_edges.append(edges[order])
        # The pinned runtime gathers reference-space UIDs from the dense
        # source array with query-layout key indices. Other AF3 revisions
        # gather UIDs from the packed query array instead. Reproduce the
        # pinned source exactly, including this distinction.
        dense_uid = np.asarray(batch["ref_space_uid"]).ravel()
        if (key_query_slots >= dense_uid.size).any():
            raise ValueError("Native key UID gather exceeds the pinned dense reference array")
        native_key_uid.append(dense_uid[key_query_slots[order]])

    union_edges = np.unique(np.concatenate(all_edges))
    source, dest = (union_edges // nslots).astype(np.int32), (union_edges % nslots).astype(np.int32)
    nlayouts = len(raw_batches)
    nedge = union_edges.size
    exists = np.zeros((nlayouts, nslots), dtype=bool)
    edge_exists = np.zeros((nlayouts, nedge), dtype=bool)
    # Pair geometry is kept per native branch. Averaging reference positions
    # across separately augmented residues before forming distances is invalid.
    pair_offset = np.zeros((nlayouts, nedge, 3), dtype=np.float32)
    pair_inverse_distance = np.zeros((nlayouts, nedge, 1), dtype=np.float32)
    pair_same_space = np.zeros((nlayouts, nedge, 1), dtype=bool)
    refs = {}
    for key in _REF_KEYS:
        first = np.asarray(raw_batches[0][key])
        refs[key] = np.zeros((nlayouts, nslots, *first.shape[2:]), dtype=first.dtype)

    for i, (batch, layout, slots, edges, key_uids) in enumerate(
        zip(raw_batches, layouts, branch_slots, all_edges, native_key_uid, strict=True)
    ):
        shape, flat_ids, _ = layout
        for name, target in refs.items():
            arr = np.asarray(batch[name])
            if arr.shape[:2] != shape or arr.shape[2:] != target.shape[2:]:
                raise ValueError(f"Invalid native atom reference {name}: {arr.shape}")
            target[i, slots] = arr.reshape(-1, *arr.shape[2:])[flat_ids]
        exists[i, slots] = True
        ids = np.searchsorted(union_edges, edges)
        edge_exists[i, ids] = True
        src, dst = source[ids], dest[ids]
        uid = refs["ref_space_uid"][i]
        valid = uid[src] == key_uids
        pos = refs["ref_pos"][i]
        delta = pos[src] - pos[dst]
        pair_offset[i, ids] = delta * valid[:, None]
        pair_inverse_distance[i, ids, 0] = valid / (1 + np.sum(delta**2, axis=-1))
        pair_same_space[i, ids, 0] = valid

    slots_by_token = [[] for _ in seq_mask]
    for union_slot, (token, _, _) in enumerate(all_atom_keys):
        slots_by_token[token].append(union_slot)
    pool_size = max(map(len, slots_by_token))
    pool_slots = np.zeros((seq_mask.size, pool_size), dtype=np.int32)
    pool_valid = np.zeros_like(pool_slots, dtype=bool)
    for i, slots in enumerate(slots_by_token):
        pool_slots[i, :len(slots)] = slots
        pool_valid[i, :len(slots)] = True
    return {
        **{key: value for key, value in refs.items() if key != "ref_space_uid"},
        "exists": exists,
        "edge_exists": edge_exists,
        "edge_src": source,
        "edge_dst": dest,
        "pair_offset": pair_offset,
        "pair_inverse_distance": pair_inverse_distance,
        "pair_same_space": pair_same_space,
        "pool_slots": pool_slots,
        "pool_valid": pool_valid,
        "variant_count": np.asarray(
            [variant_counts[(token, aa)] for token, aa, _ in all_atom_keys], dtype=np.int32,
        ),
    }


def joint_log_probabilities(log_probability, combinations):
    """Accumulate the joint distribution in log space before small products round to zero."""
    import jax.numpy as jnp

    return log_probability[jnp.arange(log_probability.shape[0])[None, :], combinations].sum(axis=-1)


def _conditional_layout_weights(layout_logs, member):
    """Return occurrence log-mass and conditional p for a token or edge bank."""
    import jax
    import jax.numpy as jnp

    active = member & jnp.isfinite(layout_logs[:, None])
    populated = jnp.any(active, axis=0, keepdims=True)
    logits = jnp.where(active, layout_logs[:, None], -jnp.inf)
    # A structurally absent row is constant zero; avoid evaluating a
    # differentiable normalization of an all-negative-infinity column.
    safe = jnp.where(populated, logits, 0.0)
    pivot = jax.lax.stop_gradient(jnp.max(safe, axis=0, keepdims=True))
    relative = jnp.exp(safe - pivot)
    norm = relative.sum(axis=0, keepdims=True)
    log_occurrence = pivot + jnp.log(norm)
    conditional = jnp.where(populated, relative / norm, 0.0)
    return jnp.where(populated, log_occurrence, -jnp.inf)[0], conditional


def _log_weighted_softmax(logits, log_mass):
    """Use log occurrence as attention prior, with zero for empty query rows."""
    import jax
    import jax.numpy as jnp

    present = jnp.any(jnp.isfinite(log_mass), axis=-1, keepdims=True)
    safe_scores = jnp.where(present, logits + log_mass, 0.0)
    return jnp.where(present, jax.nn.softmax(safe_scores, axis=-1), 0.0)


def _pooling_mask(bank, occupancy):
    """Pool each AA's atom mean at its probability, preserving native endpoints."""
    slots = bank["pool_slots"]
    native_mass = occupancy[slots] * bank["pool_valid"]
    expected_count = native_mass.sum(axis=-1, keepdims=True)
    # Divide counts before multiplying probability. At a one-hot endpoint the
    # live AA's count ratio is exactly 1, reproducing the native atomic mask.
    variant_count = bank["variant_count"][slots].astype(occupancy.dtype)
    return native_mass * (expected_count / variant_count)


def _cross_transformer(config, global_config, single, pair, log_mass, src, dst, name):
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.network import diffusion_transformer as dtx

    class WeightedCrossTransformer(hk.Module):
        """One state vector per atom slot, weighted union of native windows."""

        def __call__(self):
            attention = config.attention
            nslots, channels = single.shape
            kdim = attention.key_dim or channels
            vdim = attention.value_dim or channels
            heads = attention.num_head
            qshape, vshape = (heads, kdim // heads), (heads, vdim // heads)
            qscale = qshape[1] ** -0.5
            scope = self.name

            def block(atom_state, edge_logits):
                x_q = dtx.adaptive_layernorm(atom_state, single, name=f"{scope}q")
                x_k = dtx.adaptive_layernorm(atom_state, single, name=f"{scope}k")
                q = hm.Linear(qshape, use_bias=True, name=f"{scope}q_projection")(x_q)
                k = hm.Linear(qshape, name=f"{scope}k_projection")(x_k)
                values = hm.Linear(vshape, name=f"{scope}v_projection")(x_k)
                # Project each union slot once. Dense GEMMs avoid expanding
                # one value embedding per query/key edge in every block.
                q = q.astype(jnp.float32)
                k = k.astype(jnp.float32)
                logits = jnp.einsum("qhd,khd->hqk", q * qscale, k)
                logits = logits.at[:, src, dst].add(edge_logits)
                weights = _log_weighted_softmax(logits, log_mass[None]).astype(x_q.dtype)
                pooled = jnp.einsum("hqk,khd->qhd", weights, values)
                pooled = pooled.reshape(nslots, -1)
                gates = hm.Linear(
                    vdim, bias_init=1.0, initializer="zeros",
                    name=f"{scope}gating_query",
                )(x_q)
                pooled *= jax.nn.sigmoid(gates)
                next_state = atom_state + dtx.adaptive_zero_init(
                    pooled, channels, single, global_config, scope
                )
                next_state += dtx.transition_block(
                    next_state, config.num_intermediate_factor, global_config,
                    single, name=scope,
                )
                return next_state, None

            normalized_pair = hm.LayerNorm(
                name="pair_input_layer_norm", use_fast_variance=False, create_offset=False
            )(pair)
            edge_logits = hm.Linear(
                (config.num_blocks, heads), name="pair_logits_projection"
            )(normalized_pair)
            edge_logits = jnp.transpose(edge_logits, (1, 2, 0))
            return hk.experimental.layer_stack(
                config.num_blocks, with_per_layer_inputs=True
            )(block)(single, edge_logits)[0]

    return WeightedCrossTransformer(name=name)()


def soft_atom_token_act(bank, layout_logs, config, global_config, name="evoformer_conditioning"):
    """Encode atom slots from continuously weighted native layout alternatives."""
    import jax
    import jax.numpy as jnp
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.components import utils

    if layout_logs.ndim != 1 or layout_logs.shape[0] != bank["exists"].shape[0]:
        raise ValueError("Expected one weight per native atom layout")
    if global_config.bfloat16 != "none":
        raise ValueError("Soft atom encoder is qualified in AF3 F32 only")
    layout_logs = layout_logs.astype(jnp.float32)
    nslots = bank["exists"].shape[1]

    atom_log_mass, atom_w = _conditional_layout_weights(layout_logs, bank["exists"])
    occupancy = jnp.exp(atom_log_mass)
    atom_w = atom_w * bank["ref_mask"]
    element = jax.nn.one_hot(bank["ref_element"], 128, dtype=jnp.float32)
    chars = jax.nn.one_hot(bank["ref_atom_name_chars"], 64, dtype=jnp.float32)
    chars = chars.reshape(layout_logs.shape[0], nslots, -1)
    charge = jnp.arcsinh(bank["ref_charge"])[..., None]
    width = config.per_atom_channels
    # Native input projections are bias-free. Combine the conditionally
    # masked feature channels before applying their original parameter sets.
    pos = (bank["ref_pos"] * atom_w[..., None]).sum(axis=0)
    single = hm.Linear(width, precision="highest", name=f"{name}_embed_ref_pos")(pos)
    single += hm.Linear(width, name=f"{name}_embed_ref_mask")(atom_w.sum(axis=0)[:, None])
    elements = (element * atom_w[..., None]).sum(axis=0)
    single += hm.Linear(width, name=f"{name}_embed_ref_element")(elements)
    charges = (charge * atom_w[..., None]).sum(axis=0)
    single += hm.Linear(width, name=f"{name}_embed_ref_charge")(charges)
    letters = (chars * atom_w[..., None]).sum(axis=0)
    single += hm.Linear(width, name=f"{name}_embed_ref_atom_name")(letters)

    src, dst = bank["edge_src"], bank["edge_dst"]
    edge_log_mass, edge_w = _conditional_layout_weights(layout_logs, bank["edge_exists"])
    edge_w = edge_w[..., None]
    pair_width = config.per_atom_pair_channels
    singles_relu = jax.nn.relu(single)
    # The native encoder allocates four same-named projections in its discarded
    # local pair path first. Use the resolved cross-window checkpoint names.
    query_pair = hm.Linear(pair_width, name=f"{name}_single_to_pair_cond_row_1")(singles_relu)
    key_pair = hm.Linear(pair_width, name=f"{name}_single_to_pair_cond_col_1")(singles_relu)
    pair = query_pair[src] + key_pair[dst]
    offset = (edge_w * bank["pair_offset"]).sum(axis=0)
    pair += hm.Linear(pair_width, precision="highest", name=f"{name}_embed_pair_offsets_1")(offset)
    distance = (edge_w * bank["pair_inverse_distance"]).sum(axis=0)
    pair += hm.Linear(pair_width, name=f"{name}_embed_pair_distances_1")(distance)
    valid = (edge_w * bank["pair_same_space"]).sum(axis=0)
    pair += hm.Linear(pair_width, name=f"{name}_embed_pair_offsets_valid")(valid)

    update = hm.Linear(pair_width, initializer="relu", name=f"{name}_pair_mlp_1")(
        jax.nn.relu(pair)
    )
    update = hm.Linear(pair_width, initializer="relu", name=f"{name}_pair_mlp_2")(
        jax.nn.relu(update)
    )
    pair += hm.Linear(pair_width, initializer=global_config.final_init, name=f"{name}_pair_mlp_3")(
        jax.nn.relu(update)
    )

    log_mass = jnp.full((nslots, nslots), -jnp.inf, dtype=layout_logs.dtype)
    log_mass = log_mass.at[src, dst].set(edge_log_mass)
    atoms = _cross_transformer(
        config.atom_transformer, global_config, single, pair, log_mass, src, dst,
        name=f"{name}_atom_transformer_encoder",
    )
    projected = hm.Linear(config.per_token_channels, name=f"{name}_project_atom_features_for_aggr")(
        atoms
    )
    slots = bank["pool_slots"]
    mask = _pooling_mask(bank, occupancy)
    return utils.mask_mean(mask[..., None], jax.nn.relu(projected[slots]), axis=-2)
