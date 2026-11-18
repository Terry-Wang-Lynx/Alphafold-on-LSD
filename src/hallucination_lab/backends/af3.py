"""AF3 trunk differentiation with hard or probability-weighted atom inputs."""
from pathlib import Path
import hashlib
import json
import tempfile
import numpy as np
from .common import inputs, evaluation, design_indices, numerical_policy, with_precision
from ..sequence import ALPHABET, relax
from ..types import Prediction, objective_result


class AF3:
    """One designed protein chain. Soft Atom requires query-only protein inputs.

    Native AF3 JSON must contain inline MSA/template data, already prepared for
    inference. Positions are zero-based within design_chain. No diffusion
    backpropagation: full_prediction() is explicitly forward-only.
    """

    atom_mode = "hard"

    def __init__(self, *, input_json, design_chain, objective, model_dir,
                 design_positions=None, bucket=256, recycles=0, remat=True,
                 configure=None, attention="xla", template_gradient=False, matmul_precision="highest",
                 atom_mode="hard", soft_atom_alphabets=None):
        if atom_mode not in ("hard", "soft"):
            raise ValueError("atom_mode must be hard or soft")
        soft = atom_mode == "soft"
        if not soft and soft_atom_alphabets is not None:
            raise ValueError("soft_atom_alphabets requires atom_mode='soft'")
        self.atom_mode = atom_mode
        import haiku as hk
        import jax
        numerics = numerical_policy(matmul_precision)
        self._matmul_precision = matmul_precision
        import jax.numpy as jnp
        from alphafold3.model import model as af_model
        from alphafold3.model import params
        from absl import flags
        # Tokamax lazily parses process argv; user CLI flags belong to the caller.
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["hallucination-lab"], known_only=True)
        from ._af3.hybrid import HybridDesignTrunk, HybridFullGate
        if not isinstance(bucket, int) or bucket < 1:
            raise ValueError("bucket must be positive")
        if not isinstance(recycles, int) or recycles < 0:
            raise ValueError("recycles must be a nonnegative integer")
        self.input_path = Path(input_json).resolve()
        self.input_data = json.loads(self.input_path.read_text())
        self.chain = design_chain
        proteins = [entry["protein"] for entry in self.input_data["sequences"] if "protein" in entry]
        matches = [p for p in proteins if p["id"] == design_chain or p["id"] == [design_chain]]
        if len(matches) != 1:
            raise ValueError("design_chain must identify exactly one protein (expand repeated IDs)")
        protein = matches[0]
        if protein.get("modifications"):
            raise ValueError("Modified residues on the designed chain are not supported")
        self.sequence = protein["sequence"]
        if not self.sequence or any(a not in ALPHABET for a in self.sequence):
            raise ValueError("Designed sequence must contain standard amino acids")
        # Loading copied JSON elsewhere must not silently change relative file semantics.
        def reject_paths(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.endswith("Path"):
                        raise ValueError("Use inline AF3 inputs; external *Path references are not supported")
                    reject_paths(item)
            elif isinstance(value, list):
                for item in value:
                    reject_paths(item)
        reject_paths(self.input_data)
        positions = design_indices(design_positions, len(self.sequence))
        self.positions, self.design_length, self.bucket = positions, len(positions), bucket
        if soft:
            from importlib.metadata import version
            from alphafold3.model.network import atom_cross_attention
            from ..provenance import sha256_file
            from ._af3 import soft_atom
            if (recycles != 0 or attention != "xla" or matmul_precision != "highest"
                    or version("alphafold3") != "3.0.3"):
                raise ValueError("Soft Atom requires AF3 3.0.3, recycles=0, XLA and highest matmul precision")
            native_encoder_sha = sha256_file(atom_cross_attention.__file__)
            if native_encoder_sha != soft_atom.NATIVE_ENCODER_SHA256:
                raise ValueError("Soft Atom requires the unmodified AF3 v3.0.3 atom cross-attention source")
            chains = soft_atom.input_chains(self.input_data)
            sequences, combinations, allowed = soft_atom.candidate_layouts(
                self.sequence, positions, soft_atom_alphabets,
            )
            soft_atom.require_stable_entities(chains, self.chain, sequences)
        self._scratch = tempfile.TemporaryDirectory(prefix="hallulab-af3-")
        self._cached_sequence, self._cached_batch = None, None
        batch = self._batch(self.sequence)
        # Native featurisation can reorder chains. Equal chain lengths must not
        # silently turn a JSON-order guess into optimization of the wrong chain.
        from ._af3.mapping import protein_tokens
        chain_tokens = protein_tokens(self.token_chain_ids, self.token_residue_ids,
                                      design_chain, len(self.sequence))
        self.token_indices = chain_tokens[positions]
        design_mask = np.zeros(bucket, bool)
        design_mask[self.token_indices] = True
        self.token_mask = np.asarray(batch["seq_mask"], bool)
        self.asym_id = np.asarray(batch["asym_id"])
        self.residue_index = np.asarray(batch["residue_index"])
        config = af_model.Model.Config()
        config.num_recycles = 0
        config.return_distogram = True
        config.return_embeddings = False
        # Research default: mixed precision failed the local derivative audit.
        # An explicit configure callback may opt back into its approximate semantics.
        config.global_config.bfloat16 = "none"
        config.global_config.flash_attention_implementation = attention
        config.heads.diffusion.eval.num_samples = 1
        if configure is not None:
            configure(config)
        if config.heads.diffusion.eval.num_samples != 1:
            raise ValueError("full_prediction exports one sample; num_samples must remain 1")
        if config.num_recycles != 0:
            raise ValueError("Use recycles argument; official model recycles must remain zero")
        if soft:
            if (config.global_config.bfloat16 != "none"
                    or config.global_config.flash_attention_implementation != "xla"
                    or not config.return_distogram):
                raise ValueError("Soft Atom requires the F32 XLA distogram configuration")
            raw_reference = self._batch(self.sequence, raw=True)
            raw_batches = [raw_reference if seq == self.sequence else self._batch(seq, raw=True)
                           for seq in sequences]
            self._soft_atom_bank = jax.device_put(soft_atom.assemble_bank(
                raw_reference, raw_batches, [{**chains, self.chain: seq} for seq in sequences],
                self.token_indices,
            ))
            del raw_reference, raw_batches
            self._soft_reference = batch
            self._allowed_amino_acids = allowed

        def full_features(pseudo):
            full = jnp.zeros((bucket, 31), jnp.float32)
            return full.at[self.token_indices, :20].set(pseudo)

        def features(logits, controls):
            pseudo = (soft_atom.relax_atom_sequence(logits, controls, allowed)[0]
                      if soft else relax(logits, controls))
            return pseudo, full_features(pseudo)

        @hk.transform
        def forward(batch, full, key, atom_bank=None, atom_layout_logs=None):
            return HybridDesignTrunk(config, remat, jnp.asarray(design_mask),
                                     num_design_recycles=recycles, detach_recycles=True,
                                     template_gradient=template_gradient)(
                                         batch, full, full, key, atom_bank=atom_bank,
                                         atom_layout_logs=atom_layout_logs)

        parameters = params.get_model_haiku_params(model_dir=Path(model_dir))
        self.parameters = parameters

        def prediction(logits, batch, key, controls, atom_bank=None, weights=None):
            if soft:
                pseudo, log_p = soft_atom.relax_atom_sequence(logits, controls, allowed)
                layout_logs = soft_atom.joint_log_probabilities(log_p, combinations)
                native = forward.apply(weights, key, batch, full_features(pseudo), key,
                                       atom_bank, layout_logs)
            else:
                pseudo, full = features(logits, controls)
                native = forward.apply(parameters, key, batch, full, key)
            distogram = native["distogram"]
            return Prediction(distogram["distogram"], distogram["bin_edges"], batch["seq_mask"],
                              batch["residue_index"], batch["asym_id"], pseudo, native)

        def scored_prediction(logits, batch, key, controls, *soft_inputs):
            predicted = prediction(logits, batch, key, controls, *soft_inputs)
            loss, metrics = objective_result(objective(predicted, controls))
            return loss, (metrics, predicted)

        def score(logits, batch, key, controls, *soft_inputs):
            loss, (metrics, _) = scored_prediction(logits, batch, key, controls, *soft_inputs)
            return loss, metrics

        @hk.transform
        def full_forward(batch, full, key):
            return HybridFullGate(config, remat, jnp.asarray(design_mask),
                                  num_design_recycles=recycles, detach_recycles=True,
                                     template_gradient=template_gradient)(
                                      batch, full, full, key)

        self._forward = jax.jit(scored_prediction)
        self._gradient_full = jax.jit(jax.value_and_grad(scored_prediction, has_aux=True))
        self._score = jax.jit(score)
        self._gradient = jax.jit(jax.value_and_grad(score, has_aux=True))
        # Large closed-over constants prevented the native full executable from
        # serializing into JAX's persistent cache. Pass the same weights as device
        # operands; their values and precision are unchanged.
        self._full_apply = jax.jit(
            lambda weights, batch, full, key: full_forward.apply(weights, key, batch, full, key))
        self._full_parameters = None
        def full_model(batch, full, key):
            if self._full_parameters is None:
                self._full_parameters = jax.device_put(parameters)
            return self._full_apply(self._full_parameters, batch, full, key)
        self._full = full_model
        self._features = features
        from ..provenance import sha256_file, source_revision, source_tree_fingerprint
        from importlib.metadata import version
        from alphafold3.model.network import evoformer
        weight_path = Path(model_dir) / "af3.bin"
        self.metadata = {"numerics": numerics, "model_config": config.as_dict(),
                         "runtime_source": source_tree_fingerprint(Path(af_model.__file__).parents[1]),
                         "runtime_source_revision": source_revision(evoformer.__file__),
                         "alphafold3": version("alphafold3"),
                         "evoformer_source_sha256": sha256_file(evoformer.__file__),
                         "weight_sha256": sha256_file(weight_path) if weight_path.exists() else None,"backend": "af3", "jax": jax.__version__, "bucket": bucket,
                         "recycles": recycles, "recycle_gradient": "last", "remat": remat,
                         "input_sha256": hashlib.sha256(self.input_path.read_bytes()).hexdigest(),
                         "sequence": self.sequence, "design_chain": design_chain,
                         "design_positions": positions.tolist(), "design_tokens": self.token_indices.tolist(),
                         "token_chain_ids": self.token_chain_ids.tolist(),
                         "token_residue_ids": self.token_residue_ids.tolist(),
                         "native_gradient": ["distogram", "pair", "single"],
                         "forward_only": ["diffusion_atom_positions", "confidence"],
                         "full_parameter_transport": "explicit_jit_operands_device_resident",
                         "carrier": "argmax re-featurized; no gradient through atom chemistry",
                         "alphabet": ALPHABET, "attention": attention,
                         "template_gradient": template_gradient}
        if soft:
            self.metadata.update(atom_mode="soft", soft_atom_alphabets=list(soft_atom_alphabets),
                                 soft_atom_layouts=len(sequences),
                                 soft_atom_native_encoder_sha256=native_encoder_sha,
                                 carrier="fixed query-only trunk; shared probabilities for token and atom inputs",
                                 full_prediction_identity="locked allowed argmax")

    def _batch(self, sequence, *, raw=False):
        if not raw and sequence == self._cached_sequence:
            return self._cached_batch
        import jax
        import jax.numpy as jnp
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation
        from alphafold3.model.components import utils
        obj = json.loads(json.dumps(self.input_data))
        for entry in obj["sequences"]:
            protein = entry.get("protein", {})
            if protein.get("id") in (self.chain, [self.chain]):
                protein["sequence"] = sequence
                msa = protein.get("unpairedMsa", "")
                if msa:
                    lines = msa.splitlines()
                    boundary = next((i for i in range(1, len(lines)) if lines[i].startswith(">")), len(lines))
                    protein["unpairedMsa"] = "\n".join([lines[0], sequence, *lines[boundary:]]) + "\n"
                # A mutable paired query would require preserving species pairing.
                if protein.get("pairedMsa"):
                    raise ValueError("Designed chain pairedMsa must be empty")
        path = Path(self._scratch.name) / "carrier.json"
        path.write_text(json.dumps(obj))
        fold_input = next(iter(folding_input.load_fold_inputs_from_path(path)))
        if not hasattr(self, "_ccd"):
            self._ccd = chemical_components.Ccd()
        batch = featurisation.featurise_input(fold_input=fold_input, ccd=self._ccd,
                                              buckets=[self.bucket])[0]
        from ._af3.mapping import atom_names, token_identity
        chains, residues = token_identity(batch)
        names = atom_names(batch)
        if hasattr(self, "token_chain_ids"):
            if not (np.array_equal(chains, self.token_chain_ids)
                    and np.array_equal(residues, self.token_residue_ids)):
                raise ValueError("AF3 carrier re-featurisation changed token identity/order")
        else:
            self.token_chain_ids, self.token_residue_ids = chains, residues
        if len(batch["seq_mask"]) != self.bucket:
            raise ValueError("Input exceeds configured token bucket")
        if raw:
            return batch
        batch = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
        self._atom_names = names
        self._cached_sequence, self._cached_batch = sequence, batch
        return batch

    def _carrier(self, logits, controls):
        if self.atom_mode == "soft":
            return self._batch(self.sequence_from_logits(logits, controls))
        # relax can be raw logits; select the exact argmax used in the forward channel.
        pseudo = np.asarray(relax(logits, controls))
        sequence = list(self.sequence)
        for index, residue in zip(self.positions, pseudo.argmax(-1)):
            sequence[index] = ALPHABET[residue]
        return self._batch("".join(sequence))

    @with_precision
    def evaluate(self, logits, key, controls=None, *, gradient=True, prediction=False):
        values, key, controls = inputs(logits, key, controls, self.design_length)
        if controls["dropout"]:
            raise ValueError("AF3 dropout is not implemented; use configure for supported model options")
        if self.atom_mode == "soft":
            from ._af3.soft_atom import f32_controls, f32_logits
            if controls["soft"] != 1 or controls["hard"] not in (0, 1):
                raise ValueError("Soft Atom requires soft=1 and hard=0 for optimization, or hard=1 for scoring")
            if gradient and controls["hard"]:
                raise ValueError("Score a locked Soft Atom sequence with gradient=False")
            controls = f32_controls(controls)
            import jax
            if self._full_parameters is None:
                self._full_parameters = jax.device_put(self.parameters)
            args = (f32_logits(values), self._soft_reference, key, controls,
                    self._soft_atom_bank, self._full_parameters)
        else:
            batch = self._carrier(values, controls)
            args = (values, batch, key, controls)
        predicted = None
        if prediction:
            if gradient:
                (loss, (metrics, predicted)), grad = self._gradient_full(*args)
            else:
                loss, (metrics, predicted) = self._forward(*args)
                grad = None
        elif gradient:
            (loss, metrics), grad = self._gradient(*args)
        else:
            loss, metrics = self._score(*args)
            grad = None
        return evaluation(loss, metrics, grad, predicted)

    @with_precision
    def full_prediction(self, logits, key, controls=None, *, autotune=True):
        """Forward-only native atom positions/confidence; intentionally outside grad API.

        Autotune cache misses locally by default, because a generic Tokamax kernel
        may exceed a device's shared-memory limit. First-call tuning adds cost and
        may change floating-point rounding. The scope is restored after this call;
        evaluate() and its gradient path are unaffected. Set autotune=False only
        when the selected kernels are already qualified for the device and shape.
        """
        import jax
        values, key, controls = inputs(logits, key, controls, self.design_length)
        if controls["dropout"]:
            raise ValueError("AF3 dropout is not implemented; use configure for supported model options")
        if self.atom_mode == "soft":
            from ._af3.soft_atom import f32_controls, f32_logits
            values = f32_logits(values)
            controls = f32_controls({**controls, "hard": 1, "soft": 1})
        batch = self._carrier(values, controls)
        _, full = self._features(values, controls)
        from contextlib import nullcontext
        scope = nullcontext()
        if autotune:
            try:
                import tokamax
            except ImportError:
                pass
            else:
                # Official local configuration scope; no upstream/global monkey-patch.
                scope = tokamax.config.autotuning_cache_miss_fallback("autotune")
        with scope:
            result = jax.device_get(self._full(batch, full, key))
        # Dense coordinates include padded atom slots; masks travel with the output.
        return {**result, "atom_mask": np.asarray(batch["pred_dense_atom_mask"]),
                "atom_names": self._atom_names.copy(),
                "token_mask": np.asarray(batch["seq_mask"]),
                "token_chain_ids": self.token_chain_ids.copy(),
                "token_residue_ids": self.token_residue_ids.copy(),
                "asym_id": np.asarray(batch["asym_id"]),
                "residue_index": np.asarray(batch["residue_index"])}

    def sequence_from_logits(self, logits, controls=None):
        """Decode the same carrier identities as evaluate, retaining fixed positions."""
        from ..sequence import decode, validate_controls
        if self.atom_mode == "soft":
            from ._af3.soft_atom import f32_logits
            values = f32_logits(logits)
        else:
            values = np.asarray(logits)
        if values.shape != (self.design_length, 20) or not np.isfinite(values).all():
            raise ValueError("logits must be finite [design_length,20]")
        if self.atom_mode == "soft":
            validate_controls(controls)
            design = "".join(ALPHABET[i] for i in
                             np.argmax(np.where(self._allowed_amino_acids, values, -np.inf), axis=-1))
        else:
            design = decode(np.asarray(relax(values, validate_controls(controls))))
        if len(design) != self.design_length:
            raise ValueError("Design sequence length differs")
        sequence = list(self.sequence)
        for index, residue in zip(self.positions, design):
            sequence[index] = residue
        return "".join(sequence)

    def close(self):
        self._scratch.cleanup()
