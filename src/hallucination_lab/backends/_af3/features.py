# Copyright 2026 AlphaFold3 Hallucination Contributors
# Modifications Copyright 2026 Hallucination Lab contributors
# SPDX-License-Identifier: Apache-2.0
"""Soft query feature injection; adapted from AlphaFold3 Hallucination (NOTICE)."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def inject_soft_sequence(aatype_one_hot: jnp.ndarray, profile: jnp.ndarray,
                         pseudo_seq: jnp.ndarray, design_mask: jnp.ndarray):
    """Write the soft sequence into the target-feature channels.

    Args:
      aatype_one_hot: [N, width] fixed one-hot of the featurised sequence.
      profile: [N, width] msa profile (== query one-hot under empty MSA).
      pseudo_seq: [N, width] soft/pseudo distribution from soft_seq()["pseudo"],
        defined on every token but only used where design_mask is True.
      design_mask: [N] bool, mutable (binder) tokens.

    Returns (aatype_one_hot, profile) with design rows replaced by pseudo_seq.
    The target rows are untouched, so the gradient reaches only design tokens.
    """
    m = design_mask[:, None].astype(aatype_one_hot.dtype)
    new_aatype = aatype_one_hot * (1 - m) + pseudo_seq * m
    new_profile = profile * (1 - m) + pseudo_seq * m
    return new_aatype, new_profile


def create_msa_feat_with_soft_query(msa, pseudo_seq: jnp.ndarray,
                                    design_mask: jnp.ndarray) -> jnp.ndarray:
    """Create AF3 MSA features while soft-replacing the query row.

    This is the AF3 equivalent of the ColabDesign update_seq() write:

      msa_feat[..., 0:22] = seq_1hot

    AF3 does not store a floating msa_feat in the batch. It stores integer
    `msa.rows`, and the official network creates `msa_1hot` inside
    featurization.create_msa_feat(). We recreate the same feature locally and
    replace only row 0 at design positions with the differentiable pseudo
    sequence before the MSA stack sees it.

    Args:
      msa: alphafold3.model.features.MSA with rows/mask/deletion_matrix.
      pseudo_seq: [N, 31] soft/pseudo distribution for all tokens.
      design_mask: [N] bool mask for design tokens.

    Returns:
      [num_msa, N, 34] = one_hot(msa.rows, 32) + deletion channels.
    """
    msa_width = pseudo_seq.shape[-1] + 1
    msa_1hot = jax.nn.one_hot(msa.rows, msa_width)

    soft_query = jnp.pad(pseudo_seq, ((0, 0), (0, 1)))
    row_mask = (jnp.arange(msa_1hot.shape[0]) == 0)[:, None, None]
    pos_mask = design_mask[None, :, None]
    m = (row_mask & pos_mask).astype(msa_1hot.dtype)
    msa_1hot = msa_1hot * (1 - m) + soft_query[None, :, :] * m

    deletion_matrix = msa.deletion_matrix
    has_deletion = jnp.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (jnp.arctan(deletion_matrix / 3.0) * (2.0 / jnp.pi))[..., None]
    return jnp.concatenate([msa_1hot, has_deletion, deletion_value], axis=-1)


def recycle_embeddings(body, initial, key, recycles, *, detach):
    """Run recycles+1 passes; detach previous outputs BEFORE the final pass too."""
    state = initial
    for _ in range(recycles + 1):
        if detach:
            state = jax.tree_util.tree_map(jax.lax.stop_gradient, state)
        state, key = body(state, key)
    return state
