"""AF3 trunk differentiation with hard-carrier chemistry and soft query features."""
from pathlib import Path
import hashlib
import json
import tempfile
import numpy as np
from .common import inputs, evaluation, design_indices, numerical_policy, with_precision
from ..sequence import ALPHABET, relax
from ..types import Prediction, objective_result


class AF3:
    """One designed protein chain in an otherwise arbitrary AF3 input.

    Native AF3 JSON must contain inline MSA/template data, already prepared for
    inference. Positions are zero-based within design_chain. No diffusion
    backpropagation: full_prediction() is explicitly forward-only.
    """
    def __init__(self, *, input_json, design_chain, objective, model_dir,
                 design_positions=None, bucket=256, recycles=0, remat=True,
                 configure=None, attention="xla", template_gradient=False, matmul_precision="highest"):
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

        def features(logits, controls):
            pseudo = relax(logits, controls)
            full = jnp.zeros((bucket, 31), jnp.float32)
            full = full.at[self.token_indices, :20].set(pseudo)
            return pseudo, full

        @hk.transform
        def forward(batch, full, key):
            return HybridDesignTrunk(config, remat, jnp.asarray(design_mask),
                                     num_design_recycles=recycles, detach_recycles=True,
                                     template_gradient=template_gradient)(
                                         batch, full, full, key)

        parameters = params.get_model_haiku_params(model_dir=Path(model_dir))
        self.parameters = parameters

        def prediction(logits, batch, key, controls):
            pseudo, full = features(logits, controls)
            native = forward.apply(parameters, key, batch, full, key)
            distogram = native["distogram"]
            return Prediction(distogram["distogram"], distogram["bin_edges"], batch["seq_mask"],
                              batch["residue_index"], batch["asym_id"], pseudo, native)

        def scored_prediction(logits, batch, key, controls):
            predicted = prediction(logits, batch, key, controls)
            loss, metrics = objective_result(objective(predicted, controls))
            return loss, (metrics, predicted)

        def score(logits, batch, key, controls):
            loss, (metrics, _) = scored_prediction(logits, batch, key, controls)
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

    def _batch(self, sequence):
        if sequence == self._cached_sequence:
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
        batch = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
        if len(batch["seq_mask"]) != self.bucket:
            raise ValueError("Input exceeds configured token bucket")
        self._atom_names = names
        self._cached_sequence, self._cached_batch = sequence, batch
        return batch

    def _carrier(self, logits, controls):
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
        batch = self._carrier(values, controls)
        predicted = None
        if prediction:
            if gradient:
                (loss, (metrics, predicted)), grad = self._gradient_full(values, batch, key, controls)
            else:
                loss, (metrics, predicted) = self._forward(values, batch, key, controls)
                grad = None
        elif gradient:
            (loss, metrics), grad = self._gradient(values, batch, key, controls)
        else:
            loss, metrics = self._score(values, batch, key, controls)
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
        values = np.asarray(logits)
        if values.shape != (self.design_length, 20) or not np.isfinite(values).all():
            raise ValueError("logits must be finite [design_length,20]")
        design = decode(np.asarray(relax(values, validate_controls(controls))))
        if len(design) != self.design_length:
            raise ValueError("Design sequence length differs")
        sequence = list(self.sequence)
        for index, residue in zip(self.positions, design):
            sequence[index] = residue
        return "".join(sequence)

    def close(self):
        self._scratch.cleanup()
