# Copyright 2026 AlphaFold3 Hallucination Contributors
# Modifications Copyright 2026 Hallucination Lab contributors
# SPDX-License-Identifier: Apache-2.0
"""AF3 hybrid carrier model; adapted from AlphaFold3 Hallucination (NOTICE)."""
import functools
import haiku as hk
import jax
import jax.numpy as jnp
from alphafold3.model import feat_batch
from alphafold3.model import model as af_model
from alphafold3.model.components import mapping, utils
from alphafold3.model.network import atom_cross_attention, confidence_head, diffusion_head, distogram_head
from .trunk import WIDTH, DesignEvoformer
from .features import recycle_embeddings

class DetachedTemplateEvoformer(DesignEvoformer):
    """Preserve template forward context while explicitly detaching its increment."""
    @hk.transparent
    def _embed_template_pair(self, batch, pair_activations, pair_mask, key):
        result, key = super()._embed_template_pair(batch, pair_activations, pair_mask, key)
        return pair_activations + jax.lax.stop_gradient(result - pair_activations), key


class _HybridEmbeddingModel(af_model.Model):
    def __init__(
        self,
        config,
        remat_cfg,
        design_mask,
        *,
        num_design_recycles=0,
        detach_recycles=False,
        template_gradient=False,
        profile_mode="replace",
        native_sequence_input=False,
        name="diffuser",
    ):
        super().__init__(config, name=name)
        self._remat_cfg = remat_cfg
        self._design_mask = design_mask
        self._num_design_recycles = int(num_design_recycles)
        self._detach_recycles = bool(detach_recycles)
        self._template_gradient = bool(template_gradient)
        self._profile_mode = profile_mode
        self._native_sequence_input = native_sequence_input

    @hk.transparent
    def _hybrid_embeddings(self, batch_dict, pseudo_full, profile_full, key):
        batch = feat_batch.Batch.from_data_dict(batch_dict)
        evo_cfg = self.config.evoformer
        global_config = self.global_config
        dtype = jnp.bfloat16 if global_config.bfloat16 == "all" else jnp.float32

        with utils.bfloat16_context():
            aatype_onehot = jax.nn.one_hot(batch.token_features.aatype, WIDTH)
            if self._native_sequence_input:
                hybrid_aatype, hybrid_profile = aatype_onehot, batch.msa.profile
            else:
                from .features import inject_soft_sequence
                hybrid_aatype, hybrid_profile = inject_soft_sequence(
                    aatype_onehot, batch.msa.profile, pseudo_full, self._design_mask,
                    profile_mode=self._profile_mode, profile_full=profile_full)
            target_feat = jnp.concatenate(
                [hybrid_aatype, hybrid_profile, batch.msa.deletion_mean[..., None]], axis=-1
            ).astype(dtype)
            atom_encoding = atom_cross_attention.atom_cross_att_encoder(
                token_atoms_act=None,
                trunk_single_cond=None,
                trunk_pair_cond=None,
                config=evo_cfg.per_atom_conditioning,
                global_config=global_config,
                batch=batch,
                name="evoformer_conditioning",
            )
            target_feat = jnp.concatenate(
                [target_feat, atom_encoding.token_act], axis=-1
            ).astype(dtype)

        embedding_type = DesignEvoformer if self._template_gradient else DetachedTemplateEvoformer
        embedding_module = embedding_type(
            evo_cfg,
            global_config,
            self._remat_cfg,
            pseudo_full,
            self._design_mask,
            soft_msa_query=not self._native_sequence_input,
            query_only=self._profile_mode == "query_only",
        )
        prev = {
            "pair": jnp.zeros(
                [batch.num_res, batch.num_res, evo_cfg.pair_channel], jnp.float32
            ),
            "single": jnp.zeros([batch.num_res, evo_cfg.seq_channel], jnp.float32),
            "target_feat": target_feat,
        }

        def body(previous, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            embeddings = embedding_module(
                batch=batch, prev=previous, target_feat=target_feat, key=subkey
            )
            embeddings = {
                "pair": embeddings["pair"].astype(jnp.float32),
                "single": embeddings["single"].astype(jnp.float32),
                "target_feat": target_feat,
            }
            return embeddings, rng_key

        embeddings = recycle_embeddings(
            body, prev, key, self._num_design_recycles, detach=self._detach_recycles
        )
        return batch, embeddings


class HybridDesignTrunk(_HybridEmbeddingModel):
    """Hybrid AF3 trunk returning only model-native distogram logits."""

    def __call__(self, batch, pseudo_full, profile_full, key=None):
        if key is None:
            key = hk.next_rng_key()
        batch_obj, embeddings = self._hybrid_embeddings(batch, pseudo_full, profile_full, key)
        distogram = distogram_head.DistogramHead(
            self.config.heads.distogram, self.global_config
        )(batch_obj, embeddings, return_distogram=True)
        return {"distogram": distogram, "pair": embeddings["pair"], "single": embeddings["single"]}


class HybridFullGate(_HybridEmbeddingModel):
    """One official AF3 diffusion sample plus ConfidenceHead for a hybrid input."""

    def __init__(self, *args, return_embeddings=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._return_embeddings = bool(return_embeddings)

    def __call__(self, batch, pseudo_full, profile_full, key=None, diffusion_key=None):
        if key is None:
            key = hk.next_rng_key()
        if diffusion_key is None:
            diffusion_key = jax.random.fold_in(key, 1)
        batch_obj, embeddings = self._hybrid_embeddings(batch, pseudo_full, profile_full, key)
        denoising_step = functools.partial(
            self.diffusion_module,
            batch=batch_obj,
            embeddings=embeddings,
            use_conditioning=True,
        )
        samples = diffusion_head.sample(
            denoising_step=denoising_step,
            batch=batch_obj,
            key=diffusion_key,
            config=self.config.heads.diffusion.eval,
        )
        positions = samples["atom_positions"][0]
        confidence_batched = mapping.sharded_map(
            lambda dense_atom_positions: confidence_head.ConfidenceHead(
                self.config.heads.confidence, self.global_config
            )(
                dense_atom_positions=dense_atom_positions,
                embeddings=embeddings,
                seq_mask=batch_obj.token_features.mask,
                token_atoms_to_pseudo_beta=batch_obj.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch_obj.token_features.asym_id,
            ),
            in_axes=0,
        )(samples["atom_positions"])
        confidence = jax.tree_util.tree_map(lambda value: value[0], confidence_batched)
        distogram = distogram_head.DistogramHead(
            self.config.heads.distogram, self.global_config
        )(batch_obj, embeddings, return_distogram=True)
        output = {
            "diffusion_atom_positions": positions,
            "distogram": distogram,
            **confidence,
        }
        if self._return_embeddings:
            output["single_embeddings"] = embeddings["single"]
            output["pair_embeddings"] = embeddings["pair"]
        return output
