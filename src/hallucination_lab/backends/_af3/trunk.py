# Copyright 2024 DeepMind Technologies Limited
# Copyright 2026 AlphaFold3 Hallucination Contributors
# Modifications Copyright 2026 Hallucination Lab contributors
# SPDX-License-Identifier: Apache-2.0
"""Modified AF3 Evoformer adapter. Upstream attribution and changes: NOTICE."""
from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from alphafold3.model.network import (
    featurization,
    modules,
)
from alphafold3.model.network import evoformer as evoformer_network

from . import features as featmod
from . import remat as rematmod

WIDTH = 31


class DesignEvoformer(evoformer_network.Evoformer):
    """Evoformer with design-specific sequence injection and local remat.

    Based on Evoformer.__call__ / _embed_process_msa (AF3 b2f3d45). Every
    embedding helper is inherited and called via self; name defaults to
    'evoformer' so the parameter scope matches af3.bin (subclass name would
    otherwise be 'design_evoformer').

    Intentional deviations:
      - wrap pairformer and MSA stack per-block fns in remat.maybe_remat;
      - optionally replace the MSA query-row one-hot feature with soft pseudo
        sequence, matching ColabDesign update_seq() semantics.
    """

    def __init__(self, config, global_config, remat_cfg, pseudo_full,
                 design_mask, soft_msa_query=True, name='evoformer'):
        super().__init__(config, global_config, name=name)
        self._remat_cfg = remat_cfg
        self._pseudo_full = pseudo_full
        self._design_mask = design_mask
        self._soft_msa_query = soft_msa_query

    @hk.transparent
    def _embed_process_msa(self, msa_batch, pair_activations, pair_mask, key,
                           target_feat):
        """Process MSA with local remat and optional soft query-row injection.

        @hk.transparent is preserved so submodule param scopes (msa_activations,
        extra_msa_target_feat, msa_stack) stay identical to af3.bin.
        """
        dtype = pair_activations.dtype
        if self._soft_msa_query:
            # Match ColabDesign update_seq() semantics without writing soft
            # values into AF3's integer msa.rows. Build the floating msa_feat
            # before shuffle, inject query row 0, then apply the same shuffle
            # and truncation order to both msa_batch and msa_feat.
            key, sample_key = jax.random.split(key)
            logits = (jnp.clip(jnp.sum(msa_batch.mask, axis=-1), 0.0, 1.0) - 1.0) * 1e6
            index_order = featurization.gumbel_argsort_sample_idx(sample_key, logits)
            msa_feat = featmod.create_msa_feat_with_soft_query(
                msa_batch, self._pseudo_full, self._design_mask)
            msa_batch = msa_batch.index_msa_rows(index_order)
            msa_feat = msa_feat[index_order]
            indices = jnp.arange(self.config.num_msa)
            msa_batch = msa_batch.index_msa_rows(indices)
            msa_feat = msa_feat[indices].astype(dtype)
        else:
            msa_batch, key = featurization.shuffle_msa(key, msa_batch)
            msa_batch = featurization.truncate_msa_batch(msa_batch, self.config.num_msa)
            msa_feat = featurization.create_msa_feat(msa_batch).astype(dtype)

        msa_activations = hm.Linear(
            self.config.msa_channel, name='msa_activations')(msa_feat)
        msa_activations += hm.Linear(
            self.config.msa_channel, name='extra_msa_target_feat')(target_feat)[None]
        msa_mask = msa_batch.mask.astype(dtype)

        evoformer_input = {'msa': msa_activations, 'pair': pair_activations}
        masks = {'msa': msa_mask, 'pair': pair_mask}

        def evoformer_fn(x):
            return modules.EvoformerIteration(
                self.config.msa_stack, self.global_config, name='msa_stack')(
                    activations=x, masks=masks)

        # ---- LOCAL remat wiring (the only change vs upstream) ----
        evoformer_fn = rematmod.maybe_remat(evoformer_fn, self._remat_cfg)
        evoformer_stack = hk.experimental.layer_stack(
            self.config.msa_stack.num_layer)(evoformer_fn)
        # ----------------------------------------------------------

        evoformer_output = evoformer_stack(evoformer_input)
        return evoformer_output['pair'], key

    def __call__(self, batch, prev, target_feat, key):
        if self.global_config.bfloat16 not in {"all", "none"}:
            raise ValueError("global_config.bfloat16 must be 'all' or 'none'")
        num_residues = target_feat.shape[0]
        if batch.token_features.aatype.shape != (num_residues,):
            raise ValueError("aatype shape differs from target_feat token count")
        dtype = jnp.bfloat16 if self.global_config.bfloat16 == 'all' else jnp.float32

        with utils.bfloat16_context():
            pair_activations, pair_mask = self._seq_pair_embedding(
                batch.token_features, target_feat)

            pair_activations += hm.Linear(
                pair_activations.shape[-1], name='prev_embedding',
                initializer=self.global_config.final_init,
            )(hm.LayerNorm(name='prev_embedding_layer_norm')(
                prev['pair'].astype(pair_activations.dtype)))

            pair_activations = self._relative_encoding(batch, pair_activations)
            pair_activations = self._embed_bonds(
                batch=batch, pair_activations=pair_activations)
            pair_activations, key = self._embed_template_pair(
                batch=batch, pair_activations=pair_activations,
                pair_mask=pair_mask, key=key)
            pair_activations, key = self._embed_process_msa(
                msa_batch=batch.msa, pair_activations=pair_activations,
                pair_mask=pair_mask, key=key, target_feat=target_feat)
            del key

            single_activations = hm.Linear(
                self.config.seq_channel, name='single_activations')(target_feat)
            single_activations += hm.Linear(
                single_activations.shape[-1], name='prev_single_embedding',
                initializer=self.global_config.final_init,
            )(hm.LayerNorm(name='prev_single_embedding_layer_norm')(
                prev['single'].astype(single_activations.dtype)))

            def pairformer_fn(x):
                pairformer_iteration = modules.PairFormerIteration(
                    self.config.pairformer, self.global_config,
                    with_single=True, name='trunk_pairformer')
                pair_act, single_act = x
                return pairformer_iteration(
                    act=pair_act, single_act=single_act, pair_mask=pair_mask,
                    seq_mask=batch.token_features.mask.astype(dtype))

            # ---- LOCAL remat wiring (the only change vs upstream __call__) ----
            pairformer_fn = rematmod.maybe_remat(pairformer_fn, self._remat_cfg)
            pairformer_stack = hk.experimental.layer_stack(
                self.config.pairformer.num_layer)(pairformer_fn)
            # -------------------------------------------------------------------

            pair_activations, single_activations = pairformer_stack(
                (pair_activations, single_activations))

            output = {
                'single': single_activations,
                'pair': pair_activations,
                'target_feat': target_feat,
            }
        return output
