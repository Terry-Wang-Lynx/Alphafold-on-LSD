"""AF2 adapter over an unmodified, separately installed ColabDesign runtime."""
from copy import deepcopy
from importlib.metadata import version
import numpy as np
from .common import inputs, evaluation, design_indices, numerical_policy, with_precision
from ..sequence import ALPHABET
from ..types import Prediction, objective_result


def _validate_recycle_options(options):
    unsupported = [key for key in ("initial_guess", "use_initial_guess", "use_initial_atom_pos")
                   if options.get(key, False)]
    if unsupported:
        raise ValueError(f"AF2 independent evaluations do not support {unsupported}")
    if not options.get("clear_prev", True):
        raise ValueError("AF2 independent evaluations require clear_prev=True")


class AF2:
    """Protocol preparation is delegated to ColabDesign; objectives are explicit.

    sequence contains the mutable protein's starting sequence (binder sequence
    for protocol='binder'). design_positions are zero-based within that sequence.
    prep_options accepts ColabDesign prep_inputs options (e.g. pdb_filename,
    target_chain, binder_len). model_options accepts constructor options.
    The upstream model object is available for advanced preparation via prepare.
    Optional model_names loads a parameter bank. controls['model_index'] selects
    one model per evaluation (default 0); controls['recycles'] overrides recycles.
    Parameter values are dynamic JIT arguments. Recycles run on the host using
    native compiled single-pass functions; changing counts reuses those graphs. Prediction.native['losses'] and ['loss_weights'] expose the
    prepared native components, but only the explicit objective is optimized.
    Each evaluation starts from zero previous state; initial-guess/initial-atom
    modes and clear_prev=False are explicitly unsupported.
    score_mode='consistent' uses the gradient graph even for score-only calls,
    discarding its gradient. This avoids graph-dependent score/gradient values
    at the cost of a backward pass. 'forward' opts into native forward scoring.
    probe_forward() exposes that independent graph for numerical audits.
    numerical_guard=True additionally compares the final gradient graph's scalar
    with a native forward call at the same previous state and key. A mismatch
    raises NumericalAuditError; this is not a gradient correctness certificate.
    Forward-only scoring requires explicitly setting numerical_guard=False.
    """
    def __init__(self, *, sequence, objective, data_dir, protocol="hallucination",
                 design_positions=None, model_name=None, model_names=None, recycles=0,
                 model_options=None, prep_options=None, prepare=None, matmul_precision="highest",
                 score_mode="consistent", numerical_guard=True):
        import jax
        if matmul_precision not in ('default','high','highest'):
            raise ValueError('matmul_precision must be default, high, or highest')
        self._matmul_precision = matmul_precision
        if score_mode not in ("consistent", "forward"):
            raise ValueError("score_mode must be consistent or forward")
        if not isinstance(numerical_guard, bool):
            raise ValueError("numerical_guard must be boolean")
        if score_mode == "forward" and numerical_guard:
            raise ValueError("score_mode='forward' requires numerical_guard=False")
        self._score_mode = score_mode
        self._numerical_guard = numerical_guard
        import jax.numpy as jnp
        # ColabDesign sets its compiler flags during import. Import it before
        # numerical_policy initializes devices and snapshots the environment.
        import colabdesign
        numerics = numerical_policy(matmul_precision)
        from colabdesign import mk_afdesign_model
        if not sequence or any(a not in ALPHABET for a in sequence):
            raise ValueError("sequence must contain standard uppercase amino acids")
        if not isinstance(recycles, int) or recycles < 0:
            raise ValueError("recycles must be a nonnegative integer")
        positions = design_indices(design_positions, len(sequence))
        self.design_length = len(positions)
        self.positions = positions
        self.sequence = sequence
        options = dict(model_options or {})
        options.setdefault("use_bfloat16", False)
        _validate_recycle_options(options)
        multimer = bool(options.get("use_multimer", False))
        if model_names is not None and model_name is not None:
            raise ValueError("Use model_name or model_names, not both")
        if isinstance(model_names, str):
            raise ValueError("model_names must be a sequence, not a string")
        model_name = model_name or ("model_1_multimer_v3" if multimer else "model_1_ptm")
        names = [model_name] if model_names is None else list(model_names)
        if (not names or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("model_names must be a nonempty sequence of unique names")
        if any(("multimer" in name) != multimer for name in names):
            raise ValueError("model names and model_options['use_multimer'] must agree")
        forbidden = {"post_callback", "loss_callback", "model_names", "protocol", "data_dir",
                     "num_seq", "recycle_mode", "num_recycles", "copies",
                     "alpha", "soft", "hard", "temp", "dropout", "learning_rate",
                     "optimizer", "norm_seq_grad", "num_models", "sample_models"}
        if forbidden & options.keys():
            raise ValueError(f"Use adapter arguments, step functions or controls for {sorted(forbidden & options.keys())}")

        def post(inputs, outputs, aux, seq):
            native = dict(outputs)
            native.update({"plddt": aux["plddt"], "pae": aux["pae"],
                           "ptm": aux["ptm"], "iptm": aux["i_ptm"],
                           "losses": dict(aux["losses"]),
                           "sequence": {k: aux["seq"][k] for k in ("logits", "pssm", "hard")},
                           "loss_weights": {k: v for k, v in inputs["opt"]["weights"].items()
                                            if k != "lab_objective"}})
            prediction = Prediction(outputs["distogram"]["logits"],
                                    outputs["distogram"]["bin_edges"], inputs["seq_mask"],
                                    inputs["residue_index"], inputs["asym_id"],
                                    aux["seq"]["pseudo"][0, positions], native)
            total, metrics = objective_result(objective(prediction, inputs["lab_controls"]))
            # Replace rather than add: no hidden upstream loss affects the gradient.
            aux["losses"] = {"lab_objective": total}
            aux["lab_metrics"] = metrics
            aux["lab_prediction"] = prediction

        model = mk_afdesign_model(protocol=protocol, data_dir=str(data_dir),
                                  model_names=names, post_callback=post,
                                  recycle_mode="last", num_recycles=0, num_seq=1, **options)
        if list(model._model_names) != names:
            raise FileNotFoundError(f"AF2 parameters missing: {sorted(set(names) - set(model._model_names))}")
        prep = dict(prep_options or {})
        if prep.get("copies", 1) != 1:
            raise ValueError("This adapter supports one designed sequence; copies must be 1")
        if protocol == "hallucination":
            prep.setdefault("length", len(sequence))
        elif protocol == "binder":
            prep.setdefault("binder_len", len(sequence))
        model.prep_inputs(**prep)
        model.set_seq(sequence)
        if model._params["seq"].shape != (1, len(sequence), 20):
            raise ValueError("Prepared protocol sequence length differs from supplied sequence")
        if prepare is not None:
            prepare(model)
        _validate_recycle_options(model._args)
        fixed = np.setdiff1d(np.arange(len(sequence)), positions)
        # ColabDesign's pre-feature _fix_pos applies exact one-hot identities.
        if len(fixed):
            if protocol == "partial":
                raise ValueError("Partial protocol + design_positions requires a custom prepare adapter")
            model._wt_aatype = np.asarray([ALPHABET.index(a) for a in sequence], np.int32)
            model.opt["fix_pos"] = fixed
        # Keep preparation weights visible to native loss callbacks, but replace
        # their entire loss dictionary in post(), so none enter the total implicitly.
        model.opt["weights"] = {**model.opt["weights"], "lab_objective": 1.0}
        self.model = model
        base = deepcopy(model._inputs)
        options_base = deepcopy(model.opt)
        self._default_pssm_hard = bool(options_base.get("pssm_hard", False))
        # Snapshot prepared bias just like the compiled forward's immutable inputs.
        self._sequence_bias = np.broadcast_to(np.asarray(base.get("bias", 0.)),
                                              (1, len(sequence), 20))[0].copy()
        base_seq = jnp.asarray(model._params["seq"])
        length = len(base["residue_index"])
        previous = {"prev_msa_first_row": jnp.zeros((length, 256)),
                    "prev_pair": jnp.zeros((length, length, 128)),
                    "prev_pos": jnp.zeros((length, 37, 3))}
        if model._args["use_dgram"]:
            previous["prev_dgram"] = jnp.zeros((length, length, 64))
        self._parameters = model._model_params
        self._default_recycles = recycles

        def execute(logits, key, controls, parameters, recycle_count, gradient):
            params = {"seq": base_seq.at[0, positions].set(logits)}
            batch = {**base, "opt": {**options_base,
                     "soft": controls["soft"], "hard": controls["hard"],
                     "temp": controls["temperature"], "alpha": controls["alpha"],
                     "dropout": controls["dropout"],
                     "pssm_hard": controls["pssm_hard"]}, "lab_controls": controls,
                     "prev": previous}
            for _ in range(recycle_count):
                key, subkey = jax.random.split(key)
                _, aux = model._model["fn"](params, parameters, batch, subkey)
                batch = {**batch, "prev": aux["prev"]}
            # Previous state is a separate argument, so differentiating only params
            # detaches all earlier passes, matching native recycle_mode='last'.
            if gradient:
                (loss, aux), grads = model._model["grad_fn"](params, parameters, batch, key)
                grad = grads["seq"][0, positions]
            else:
                loss, aux = model._model["fn"](params, parameters, batch, key)
                grad = None
            probe_loss = None
            if gradient and self._numerical_guard:
                probe_loss, _ = model._model["fn"](params, parameters, batch, key)
            return loss, aux["lab_metrics"], grad, aux["lab_prediction"], probe_loss

        self._execute = execute
        from pathlib import Path
        from inspect import getfile
        from ..provenance import sha256_file, source_tree_fingerprint
        weights = {name: Path(data_dir) / f"params_{name}.npz" for name in names}
        weight_hashes = {name: sha256_file(path) if path.exists() else None
                         for name, path in weights.items()}
        self.metadata = {"numerics": numerics, "model_config": model._cfg.to_dict(),
                         "runtime_source": source_tree_fingerprint(Path(colabdesign.__file__).parent),
                         "runtime_model_source_sha256": sha256_file(getfile(type(model))),
                         "weight_sha256": weight_hashes[names[0]] if len(names) == 1 else None,
                         "model_names": names, "weight_sha256_by_model": weight_hashes,"backend": "af2", "colabdesign": version("colabdesign"),
                         "jax": jax.__version__, "model": names[0] if len(names) == 1 else None, "protocol": protocol,
                         "sequence": sequence, "design_positions": positions.tolist(),
                         "recycles": recycles, "recycle_gradient": "last",
                         "execution": "native_single_pass_host_recycles",
                         "score_mode": score_mode,
                         "numerical_guard": numerical_guard,
                         "numerical_guard_scope": "same_state_final_scalar_only",
                         "numerical_guard_tolerances": {"rtol": 1e-4, "atol": 1e-4},
                         "recycle_initialization": "zero_each_evaluation",
                         "pssm_hard_default": self._default_pssm_hard,
                         "native_gradient": ["distogram", "predicted_lddt", "predicted_aligned_error",
                                             "structure_module", "plddt", "pae", "ptm", "iptm", "losses"],
                         "alphabet": ALPHABET}

    @with_precision
    def evaluate(self, logits, key, controls=None, *, gradient=True, prediction=False):
        return self._evaluate(logits, key, controls, gradient=gradient, prediction=prediction,
                              gradient_graph=gradient or self._score_mode == "consistent")

    @with_precision
    def probe_forward(self, logits, key, controls=None):
        """Independent native forward graph for audits, not an AD correctness test.

        A mismatch with evaluate may expose graph-dependent numerical sensitivity.
        Ordinary finite differences of hard/STE outputs do not validate the STE.
        """
        return self._evaluate(logits, key, controls, gradient=False, prediction=False,
                              gradient_graph=False)

    def _evaluate(self, logits, key, controls, *, gradient, prediction, gradient_graph):
        values, key, controls = inputs(logits, key, controls, self.design_length)
        controls.setdefault("pssm_hard", self._default_pssm_hard)
        if not isinstance(controls["pssm_hard"], (bool, np.bool_)):
            raise ValueError("pssm_hard must be boolean")
        model_index = controls.pop("model_index", 0)
        recycle_count = controls.pop("recycles", self._default_recycles)
        for name, value in (("model_index", model_index), ("recycles", recycle_count)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if model_index >= len(self._parameters):
            raise ValueError("model_index exceeds the configured model bank")
        loss, metrics, grad, predicted, probe_loss = self._execute(
            values, key, controls, self._parameters[model_index], int(recycle_count), gradient_graph)
        if not gradient:
            grad = None
        if not prediction:
            predicted = None
        metrics = {**metrics, "af2_model_index": model_index, "af2_recycles": recycle_count,
                   "af2_pssm_hard": controls["pssm_hard"]}
        result = evaluation(loss, metrics, grad, predicted)
        if probe_loss is not None:
            from ..numerics import compare_evaluations, NumericalAuditError
            from ..types import Evaluation
            check = compare_evaluations(result, Evaluation(float(probe_loss), {}), gradients=False)
            if not check["passed"]:
                import hashlib
                report = {"passed": False, "scope": "Same-state final scalar consistency only",
                          "checks": {"final_forward_vs_gradient": check},
                          "controls": {**controls, "model_index": model_index,
                                       "recycles": recycle_count}, "key": key.tolist(),
                          "input_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                          "backend_metadata": self.metadata}
                raise NumericalAuditError(report, evaluation=result, input_logits=values)
        return result

    def sequence_from_logits(self, logits, controls=None):
        """Decode native hard identities (alpha * logits + prepared bias).

        alpha defaults to 1, matching evaluate's default controls. This is the
        native hard sequence, not the argmax of a raw/soft mixed pseudosequence.
        Fixed positions always retain the supplied starting identities.
        """
        from ..sequence import decode, validate_controls
        controls = validate_controls(controls)
        values = np.asarray(logits, np.float32)
        if values.shape != (self.design_length, 20) or not np.isfinite(values).all():
            raise ValueError("logits must be finite [design_length,20]")
        design = decode(controls["alpha"] * values + self._sequence_bias[self.positions])
        if len(design) != self.design_length:
            raise ValueError("Design sequence length differs")
        sequence = list(self.sequence)
        for index, residue in zip(self.positions, design):
            sequence[index] = residue
        return "".join(sequence)
