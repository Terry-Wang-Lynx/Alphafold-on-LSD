"""AF3 trunk differentiation with hard-carrier chemistry and soft query features."""
from datetime import date
from pathlib import Path
import hashlib
import json
import tempfile
import numpy as np
from .common import inputs, evaluation, design_indices, numerical_policy, with_precision
from ..sequence import ALPHABET, decode, initial_logits, relax, validate_controls
from ..types import Prediction, objective_result


_REFERENCE_CUTOFF = date(2021, 9, 30)


def _a3m_records(msa):
    # Native A3M starts at its first FASTA header. Strip file preambles so
    # comment/blank text cannot become an orphan sequence or obsolete query.
    records = []
    for line in msa.splitlines():
        clean = line.strip()
        if clean.startswith(">"):
            records.append((clean, []))
        elif records:
            records[-1][1].append(line)
        elif clean and not clean.startswith(("#", ";")):
            raise ValueError("Unexpected text before the first A3M query header")
    if msa.strip() and not records:
        raise ValueError("Non-empty A3M is missing its query header")
    if any(not "".join(body).strip() for _, body in records):
        raise ValueError("A3M contains an empty sequence record")
    return records


def _record_identity(record):
    name, lines = record
    return name, "".join(line.strip() for line in lines)


def _rewrite_query(msa, sequence, original_sequence):
    records = _a3m_records(msa)
    if not records:
        return ""
    query = _record_identity(records[0])
    if query[1] != original_sequence:
        from alphafold3.constants import mmcif_names
        from alphafold3.data import msa as native_msa
        if not native_msa.sequences_are_feature_equivalent(
                query[1], original_sequence, mmcif_names.PROTEIN_CHAIN):
            raise ValueError("First A3M sequence differs from the original design chain")
    output = []
    for record in records:
        name, body = record
        output.append(name)
        if _record_identity(record) == query:
            output.append(sequence)
        else:
            # A distinct record that happens to match WT is still a supplied
            # homolog. Do not mutate it along with the designed protein.
            output.extend(body)
    return "\n".join(output) + "\n"


def _profile_mode(proteins, requested):
    if requested not in ("auto", "native", "query_only", "replace"):
        raise ValueError("profile_mode must be auto, native, query_only or replace")
    def no_homologs(protein):
        msa = protein.get("unpairedMsa")
        if msa is None or protein.get("pairedMsa"):
            return False
        records = _a3m_records(msa)
        if not records:
            return True
        query = _record_identity(records[0])
        return all(_record_identity(record) == query for record in records[1:])
    query_only = all(no_homologs(protein) for protein in proteins)
    if requested == "query_only" and not query_only:
        raise ValueError("query_only requires empty or single-query protein MSAs")
    if requested == "auto":
        return "query_only" if query_only else "native"
    return requested


def _check_carrier(batch, chain_tokens, sequence, profile_mode, *, allow_native_fallback=False):
    expected = np.array([ALPHABET.index(a) for a in sequence])
    if not np.array_equal(np.asarray(batch["aatype"])[chain_tokens], expected):
        raise ValueError("AF3 carrier token identities differ from the requested sequence")
    if profile_mode == "query_only":
        rows = np.asarray(batch["msa"])[:, chain_tokens]
        active = np.asarray(batch["msa_mask"], bool)[:, chain_tokens]
        profile = np.asarray(batch["profile"])[chain_tokens]
        expected_profile = np.eye(profile.shape[-1], dtype=profile.dtype)[expected]
        if (not active[0].all() or np.any(active & (rows != expected))
                or not np.array_equal(profile, expected_profile)):
            if allow_native_fallback:
                # Multichain assembly may create masked or gapped native rows,
                # even from individual query-only protein inputs.
                return "native"
            raise ValueError("query_only carrier contains unexpected MSA identities or profile")
    return profile_mode


def _weight_provenance(files):
    # Match the native loader's exact ordering for monolithic, compressed
    # and sharded archives. A one-file digest retains its original meaning.
    digest = hashlib.sha256()
    names = []
    for file in files:
        path = Path(file)
        names.append(path.name)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    if not names:
        raise ValueError("AF3 loader selected no weight files")
    return {"weight_sha256": digest.hexdigest(), "weight_files": names}


def _load_parameters(params, model_dir):
    def files_and_stats():
        selected, _ = params.select_model_files(model_dir)
        paths = [Path(file) for file in selected]
        signatures = []
        for path in paths:
            stat = path.stat()
            signatures.append((stat.st_dev, stat.st_ino, stat.st_size,
                               stat.st_mtime_ns, stat.st_ctime_ns))
        return paths, signatures

    paths, signatures = files_and_stats()
    parameters = params.get_model_haiku_params(model_dir=model_dir)
    provenance = _weight_provenance(paths)
    if files_and_stats() != (paths, signatures):
        raise RuntimeError("AF3 weight files changed while loading or fingerprinting")
    return parameters, provenance


class AF3:
    """One designed protein chain in an otherwise arbitrary AF3 input.

    Native AF3 JSON must contain inline MSA/template data, already prepared for
    inference. Positions are zero-based within design_chain. No diffusion
    backpropagation: full_prediction() is explicitly forward-only.
    """
    def __init__(self, *, input_json, design_chain, objective, model_dir,
                 design_positions=None, bucket=256, recycles=0, remat=True,
                 configure=None, attention="xla", template_gradient=False, matmul_precision="highest",
                 profile_mode="auto"):
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
        self._profile_mode = _profile_mode(proteins, profile_mode)
        self._auto_profile = profile_mode == "auto"
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
                                     template_gradient=template_gradient,
                                     profile_mode=self._profile_mode)(
                                         batch, full, full, key)

        parameters, weight_provenance = _load_parameters(params, Path(model_dir))
        self.parameters = parameters

        def prediction(weights, logits, batch, key, controls):
            pseudo, full = features(logits, controls)
            native = forward.apply(weights, key, batch, full, key)
            distogram = native["distogram"]
            return Prediction(distogram["distogram"], distogram["bin_edges"], batch["seq_mask"],
                              batch["residue_index"], batch["asym_id"], pseudo, native)

        def scored_prediction(weights, logits, batch, key, controls):
            predicted = prediction(weights, logits, batch, key, controls)
            loss, metrics = objective_result(objective(predicted, controls))
            return loss, (metrics, predicted)

        def score(weights, logits, batch, key, controls):
            loss, (metrics, _) = scored_prediction(weights, logits, batch, key, controls)
            return loss, metrics

        @hk.transform
        def full_forward(batch, full, key, diffusion_key, native_input):
            return HybridFullGate(config, remat, jnp.asarray(design_mask),
                                  num_design_recycles=recycles, detach_recycles=True,
                                  template_gradient=template_gradient,
                                  profile_mode=self._profile_mode,
                                  native_sequence_input=native_input)(
                                      batch, full, full, key, diffusion_key=diffusion_key)

        self._forward = jax.jit(scored_prediction)
        self._gradient_full = jax.jit(jax.value_and_grad(scored_prediction, argnums=1, has_aux=True))
        self._score = jax.jit(score)
        self._gradient = jax.jit(jax.value_and_grad(score, argnums=1, has_aux=True))
        # Use the same resident weights as explicit operands in every graph.
        self._full_parameters = jax.device_put(parameters)
        self._full_apply = jax.jit(full_forward.apply, static_argnums=(6,))

        def full_model(batch, full, key, diffusion_key, native_input):
            return self._full_apply(self._full_parameters, key, batch, full, key,
                                    diffusion_key, native_input)
        self._full = full_model
        self._features = features
        from ..provenance import sha256_file, source_revision, source_tree_fingerprint
        from importlib.metadata import version
        from alphafold3.model.network import evoformer
        self.metadata = {"numerics": numerics, "model_config": config.as_dict(),
                         "runtime_source": source_tree_fingerprint(Path(af_model.__file__).parents[1]),
                         "runtime_source_revision": source_revision(evoformer.__file__),
                         "alphafold3": version("alphafold3"),
                         "evoformer_source_sha256": sha256_file(evoformer.__file__),
                         **weight_provenance,
                         "profile_mode": self._profile_mode,
                         "ref_max_modified_date": _REFERENCE_CUTOFF.isoformat(),
                         "backend": "af3", "jax": jax.__version__, "bucket": bucket,
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
                original_sequence = protein["sequence"]
                protein["sequence"] = sequence
                msa = protein.get("unpairedMsa", "")
                if msa:
                    protein["unpairedMsa"] = _rewrite_query(msa, sequence, original_sequence)
                # A mutable paired query would require preserving species pairing.
                if protein.get("pairedMsa"):
                    raise ValueError("Designed chain pairedMsa must be empty")
        path = Path(self._scratch.name) / "carrier.json"
        path.write_text(json.dumps(obj))
        fold_input = next(iter(folding_input.load_fold_inputs_from_path(path)))
        if not hasattr(self, "_ccd"):
            self._ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
        batch = featurisation.featurise_input(
            fold_input=fold_input, ccd=self._ccd, buckets=[self.bucket],
            ref_max_modified_date=_REFERENCE_CUTOFF)[0]
        from ._af3.mapping import atom_names, protein_tokens, token_identity
        chains, residues = token_identity(batch)
        if hasattr(self, "token_chain_ids"):
            if not (np.array_equal(chains, self.token_chain_ids)
                    and np.array_equal(residues, self.token_residue_ids)):
                raise ValueError("AF3 carrier re-featurisation changed token identity/order")
        resolved_profile = _check_carrier(
            batch, protein_tokens(chains, residues, self.chain, len(sequence)),
            sequence, self._profile_mode,
            allow_native_fallback=self._auto_profile and self._cached_sequence is None)
        names = atom_names(batch)
        if not np.array_equal(names != "", np.asarray(batch["pred_dense_atom_mask"], bool)):
            raise ValueError("AF3 carrier atom names disagree with its dense atom mask")
        batch = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
        if len(batch["seq_mask"]) != self.bucket:
            raise ValueError("Input exceeds configured token bucket")
        self.token_chain_ids, self.token_residue_ids = chains, residues
        self._profile_mode, self._atom_names = resolved_profile, names
        self._cached_sequence, self._cached_batch = sequence, batch
        return batch

    def _carrier(self, logits, controls):
        return self._batch(self.sequence_from_logits(logits, controls))

    @with_precision
    def evaluate(self, logits, key, controls=None, *, gradient=True, prediction=False):
        values, key, controls = inputs(logits, key, controls, self.design_length)
        if controls["dropout"]:
            raise ValueError("AF3 dropout is not implemented; use configure for supported model options")
        batch = self._carrier(values, controls)
        predicted = None
        if prediction:
            if gradient:
                (loss, (metrics, predicted)), grad = self._gradient_full(self._full_parameters, values, batch, key, controls)
            else:
                loss, (metrics, predicted) = self._forward(self._full_parameters, values, batch, key, controls)
                grad = None
        elif gradient:
            (loss, metrics), grad = self._gradient(self._full_parameters, values, batch, key, controls)
        else:
            loss, metrics = self._score(self._full_parameters, values, batch, key, controls)
            grad = None
        return evaluation(loss, metrics, grad, predicted)

    @with_precision
    def full_prediction(self, logits, key, controls=None, *, autotune=True, diffusion_key=None):
        """Forward-only structure prediction from the selected hybrid controls.

        With default controls, even one-hot logits represent a soft query. Use
        predict_hard_sequence() for validation of a locked protein sequence.

        Autotune cache misses locally by default, because a generic Tokamax kernel
        may exceed a device's shared-memory limit. First-call tuning adds cost and
        may change floating-point rounding. The scope is restored after this call;
        evaluate() and its gradient path are unaffected. Set autotune=False only
        when the selected kernels are already qualified for the device and shape.
        """
        values, key, controls = inputs(logits, key, controls, self.design_length)
        if controls["dropout"]:
            raise ValueError("AF3 dropout is not implemented; use configure for supported model options")
        batch = self._carrier(values, controls)
        _, full = self._features(values, controls)
        return self._export_prediction(batch, full, key, diffusion_key, autotune,
                                       native_input=False)

    @with_precision
    def predict_hard_sequence(self, sequence, key, *, autotune=True, diffusion_key=None):
        """Validate a complete locked design-chain sequence using native input features.

        The input string includes fixed positions. Re-featurisation supplies
        its hard atoms, token identities, MSA rows and profile to the same
        adapter and configured recycle/precision policy used in this instance.
        """
        if not isinstance(sequence, str) or len(sequence) != len(self.sequence):
            raise ValueError("sequence must be the full-length designed chain")
        mutable = set(self.positions.tolist())
        if any(a not in ALPHABET or (i not in mutable and a != self.sequence[i])
               for i, a in enumerate(sequence)):
            raise ValueError("sequence contains an invalid residue or changed a fixed position")
        onehot = initial_logits(sequence="".join(sequence[i] for i in self.positions))
        _, key, _ = inputs(onehot, key, None, self.design_length)
        batch = self._batch(sequence)
        # The native-input graph bypasses all replacement of aatype, profile
        # and MSA rows. No relaxation or padded differentiable input is needed.
        result = self._export_prediction(batch, None, key, diffusion_key, autotune,
                                         native_input=True)
        result["sequence"] = sequence
        return result

    def _export_prediction(self, batch, full, key, diffusion_key, autotune, *, native_input):
        import jax
        if diffusion_key is None:
            diffusion_key = np.asarray(jax.random.fold_in(key, 1))
            source = "model_key_fold_in_1"
        else:
            diffusion_key = np.asarray(diffusion_key)
            if diffusion_key.dtype != np.uint32 or diffusion_key.shape != (2,):
                raise ValueError("diffusion_key must be a uint32 array of shape (2,)")
            source = "explicit"
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
            result = jax.device_get(self._full(batch, full, key, diffusion_key, native_input))
        # Dense coordinates include padded atom slots; masks travel with the output.
        return {**result,
                "sequence_representation": "native_hard" if native_input else "hybrid_controls",
                "rng_provenance": {"model_key": key.tolist(), "diffusion_key": diffusion_key.tolist(),
                                   "diffusion_key_source": source},
                "atom_mask": np.asarray(batch["pred_dense_atom_mask"]),
                "atom_names": self._atom_names.copy(),
                "token_mask": np.asarray(batch["seq_mask"]),
                "token_chain_ids": self.token_chain_ids.copy(),
                "token_residue_ids": self.token_residue_ids.copy(),
                "asym_id": np.asarray(batch["asym_id"]),
                "residue_index": np.asarray(batch["residue_index"])}

    def sequence_from_logits(self, logits, controls=None):
        """Decode the same carrier identities as evaluate, retaining fixed positions."""
        validate_controls(controls)
        values = np.asarray(logits, np.float32)
        if values.shape != (self.design_length, 20) or not np.isfinite(values).all():
            raise ValueError("logits must be finite [design_length,20]")
        # A positive scale and convex raw/soft/hard mixing preserve order.
        # Decode before rounding probabilities can create a spurious tie.
        design = decode(values)
        sequence = list(self.sequence)
        for index, residue in zip(self.positions, design):
            sequence[index] = residue
        return "".join(sequence)

    def close(self):
        self._scratch.cleanup()
