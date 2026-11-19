"""A small host-side loop; every expensive evaluation belongs to a backend."""
from dataclasses import dataclass, field, asdict
from copy import deepcopy
from pathlib import Path
from typing import Callable
import hashlib
import json
import time
import numpy as np
from .sequence import decode
from ._version import __version__


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    learning_rate: float = 0.1
    controls: dict = field(default_factory=dict)
    end: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or type(self.steps) is not int or self.steps < 1:
            raise ValueError("A stage needs a name and positive integer steps")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for mapping in (self.controls, self.end):
            if "learning_rate" in mapping:
                rate = mapping["learning_rate"]
                if not np.isscalar(rate) or not np.isfinite(rate) or rate <= 0:
                    raise ValueError("learning_rate endpoints must be finite and positive")
        if not self.end.keys() <= self.controls.keys() | {"learning_rate"}:
            raise ValueError("Every ramp endpoint needs a starting control value")

    def at(self, index):
        if not 0 <= index < self.steps:
            raise IndexError(index)
        fraction = index / max(self.steps - 1, 1)
        controls = {"learning_rate": self.learning_rate, **self.controls}
        for key, value in self.end.items():
            if fraction == 1:
                controls[key] = value
            elif fraction > 0:
                controls[key] = (1 - fraction) * controls[key] + fraction * value
        return controls


@dataclass
class State:
    logits: np.ndarray
    step: int = 0
    optimizer: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    best_loss: float | None = None
    best_logits: np.ndarray | None = None
    last_controls: dict | None = None
    best_controls: dict | None = None


@dataclass
class StepResult:
    """loss and metrics MUST describe evaluated_logits, not the next state."""
    logits: np.ndarray
    loss: float
    metrics: dict
    evaluated_logits: np.ndarray
    optimizer: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    effective_controls: dict | None = None


def gradient_step(state, backend, controls, key, *, optimizer="adam", clip_norm=None,
                  transform: Callable | None = None):
    """Public replaceable step; transforms receive (gradient, state, controls).

    Backend gradients and design variables are float32. Host norms and optimizer
    moments use float64 so squaring a finite float32 gradient cannot overflow.
    The returned design is explicitly rounded to float32 after a finite check.
    """
    learning_rate = controls["learning_rate"]
    if not np.isscalar(learning_rate) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    result = backend.evaluate(state.logits, key, controls, gradient=True)
    gradient = np.array(result.gradient, dtype=np.float32, copy=True)
    if gradient.shape != state.logits.shape or not np.isfinite(gradient).all():
        raise FloatingPointError("Invalid sequence gradient; state was not advanced")
    if transform is not None:
        gradient = np.asarray(transform(gradient, state, controls), np.float32)
        if gradient.shape != state.logits.shape or not np.isfinite(gradient).all():
            raise FloatingPointError("Gradient transform produced an invalid gradient")
    gradient = gradient.astype(np.float64)
    norm = float(np.linalg.norm(gradient))
    if clip_norm is not None:
        if not np.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        gradient *= min(1.0, clip_norm / max(norm, 1e-12))
    saved = state.optimizer
    if saved and saved.get("kind") != optimizer:
        raise ValueError("Changing optimizer requires explicitly resetting its state")
    if optimizer == "sgd":
        direction, saved = gradient, {"kind": "sgd"}
    elif optimizer == "adam":
        count = saved.get("count", 0)
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Adam count must be a nonnegative integer")
        if count and not {"first", "second"} <= saved.keys():
            raise ValueError("Resuming Adam requires both moments")
        first = np.asarray(saved.get("first", np.zeros_like(gradient)), np.float64)
        second = np.asarray(saved.get("second", np.zeros_like(gradient)), np.float64)
        if (first.shape != gradient.shape or second.shape != gradient.shape
                or not np.isfinite(first).all() or not np.isfinite(second).all()
                or (second < 0).any()):
            raise ValueError("Adam moments must have the gradient shape, be finite, and second >= 0")
        count = int(count) + 1
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            direction = (first / (1 - 0.9**count)) / (np.sqrt(second / (1 - 0.999**count)) + 1e-8)
        saved = {"kind": "adam", "count": count, "first": first, "second": second}
    else:
        raise ValueError("optimizer must be adam or sgd; custom optimizers use a step callable")
    if not np.isfinite(direction).all():
        raise FloatingPointError("Nonfinite optimizer direction; state was not advanced")
    with np.errstate(over="raise", invalid="raise"):
        updated = (state.logits.astype(np.float64) - learning_rate * direction).astype(np.float32)
    if not np.isfinite(updated).all():
        raise FloatingPointError("Optimizer update is not representable as finite float32")
    return StepResult(updated, result.loss, {**result.metrics, "gradient_norm": norm},
                      state.logits.copy(), saved, dict(state.extra))


def _step_identity(function):
    """Identify the implementation and serializable bound arguments at run creation."""
    from functools import partial
    import inspect

    def argument(value):
        if isinstance(value, np.ndarray) and not value.dtype.hasobject:
            return {"dtype": str(value.dtype), "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            # Arbitrary external state still belongs in the caller's problem_id.
            return {"caller_owned_type": type(value).__qualname__}
        return value

    if isinstance(function, partial):
        def bound(value):
            return _step_identity(value) if callable(value) else argument(value)
        return {"partial": _step_identity(function.func),
                "args": [bound(x) for x in function.args],
                "kwargs": {k: bound(v) for k, v in function.keywords.items()}}
    impl = function if inspect.isroutine(function) else type(function).__call__
    try:
        source = inspect.getsource(impl)
    except (OSError, TypeError):
        source = None
    cells = getattr(impl, "__closure__", None) or ()
    return {"module": getattr(impl, "__module__", None), "name": impl.__qualname__,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
            "defaults": [argument(x) for x in (getattr(impl, "__defaults__", None) or ())],
            "kwdefaults": {k: argument(v) for k, v in (getattr(impl, "__kwdefaults__", None) or {}).items()},
            "closure": [argument(cell.cell_contents) for cell in cells]}


class Run:
    """Deterministic seed-per-step execution, atomic checkpoints, caller hooks.

    A problem_id must change if the objective, input, parameters, or algorithm
    changes. Backend metadata and the complete schedule are additionally checked
    on resume, including initial logits and the step implementation. Capture
    custom closure / object state and external dependencies in problem_id. Hooks run after a committed step; returning False stops cleanly.
    """
    def __init__(self, backend, logits, stages, *, problem_id, seed=0,
                 step_fn=gradient_step, observers=(), output=None):
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("Provide a nonempty problem_id for checkpoint compatibility")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        logits = np.asarray(logits, np.float32)
        if logits.shape != (backend.design_length, 20) or not np.isfinite(logits).all():
            raise ValueError("Initial logits must be finite [design_length,20]")
        self.backend, self.stages = backend, deepcopy(tuple(stages))
        if not self.stages:
            raise ValueError("At least one stage is required")
        self.state = State(logits.copy())
        if isinstance(step_fn, dict):
            missing = {stage.name for stage in self.stages} - step_fn.keys()
            if missing or not all(callable(value) for value in step_fn.values()):
                raise ValueError(f"Provide a step callable for every stage; missing {sorted(missing)}")
        elif not callable(step_fn):
            raise TypeError("step_fn must be a callable or a stage-name-to-callable dictionary")
        self.seed, self.step_fn, self.observers = seed, step_fn, tuple(observers)
        self.output = Path(output) if output else None
        self._restored = False
        from .provenance import source_tree_fingerprint
        self.manifest = {"format": 3, "tool_version": __version__,
                         "tool_source": source_tree_fingerprint(Path(__file__).parent),
                         "host_numerics": {"numpy": np.__version__,
                                           "gradient_step": "float32_design_float64_moments"},
                         "problem_id": problem_id, "seed": seed,
                         "initial_logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                         "step": ({k: _step_identity(v) for k, v in step_fn.items()}
                                  if isinstance(step_fn, dict) else _step_identity(step_fn)),
                         "backend": backend.metadata, "stages": [asdict(s) for s in self.stages]}
        encoded = json.dumps(self.manifest, sort_keys=True, allow_nan=False).encode()
        self.manifest = json.loads(encoded)  # Normalize tuples identically to persisted JSON.
        self.fingerprint = hashlib.sha256(encoded).hexdigest()
        if self.output:
            manifest_path = self.output / "manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != self.manifest:
                raise ValueError("Output directory belongs to a different run")

    def _ensure_output_manifest(self):
        """Commit output metadata only after any requested restore has succeeded."""
        if not self.output:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != self.manifest:
                raise ValueError("Output directory belongs to a different run")
            return
        from .checkpoint import atomic_text
        atomic_text(manifest_path, json.dumps(self.manifest, indent=2) + "\n")

    @property
    def total_steps(self):
        return sum(s.steps for s in self.stages)

    def controls_at(self, global_step):
        remaining = global_step
        for stage in self.stages:
            if remaining < stage.steps:
                return stage.name, stage.at(remaining)
            remaining -= stage.steps
        raise IndexError(global_step)

    def _sequence_fields(self, prefix, logits, controls):
        """Keep raw variable identities distinct from backend-decoded identities."""
        raw = decode(logits) if logits is not None else None
        decoder = getattr(self.backend, "sequence_from_logits", None)
        if logits is None:
            sequence, semantics = None, "not_available"
        elif decoder is None:
            sequence, semantics = raw, "raw_design_argmax"
        elif controls is None:
            sequence, semantics = None, "unknown_effective_controls"
        else:
            sequence = decoder(logits, deepcopy(controls))
            semantics = "backend_hard_identity"
        return {f"{prefix}_sequence": sequence,
                f"{prefix}_raw_design_sequence": raw,
                f"{prefix}_sequence_semantics": semantics,
                f"{prefix}_sequence_controls": deepcopy(controls)}

    def _record_numerical_failure(self, error, key, controls):
        """Persist a rejected evaluation without committing it as a run step."""
        if self.output is None:
            return
        import tempfile
        root = self.output / "failures"
        root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f"step-{self.state.step:06d}-", dir=root))
        self.save(folder / "state.npz")
        arrays = {"state_logits": self.state.logits, "step_key": key}
        rejected_logits = getattr(error, "input_logits", None)
        if rejected_logits is not None:
            arrays["rejected_logits"] = np.asarray(rejected_logits)
            if "key" in error.report:
                arrays["rejected_key"] = np.asarray(error.report["key"], np.uint32)
        rejected = getattr(error, "evaluation", None)
        if rejected is not None and rejected.gradient is not None:
            arrays["rejected_gradient"] = np.asarray(rejected.gradient)
        np.savez_compressed(folder / "rejected.npz", **arrays)

        def serialize(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"Cannot serialize diagnostic value {type(value).__name__}")

        report = {"status": "numerical_failure", "step": self.state.step,
                  "key": key.tolist(), "controls": controls, "fingerprint": self.fingerprint,
                  "error": str(error), "audit": error.report}
        from .checkpoint import atomic_text
        atomic_text(folder / "report.json", json.dumps(report, indent=2, allow_nan=False,
                                                       default=serialize) + "\n")
        error.evidence_directory = str(folder)

    def run(self, *, max_steps=None, max_seconds=None, checkpoint_every=1):
        """Time bound checked between evaluations, never interrupts a device kernel."""
        if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
            raise ValueError("max_steps must be a nonnegative integer")
        if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds < 0):
            raise ValueError("max_seconds must be nonnegative and finite")
        if type(checkpoint_every) is not int or checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.output and self.state.step == 0 and not self._restored:
            trajectory = self.output / "trajectory.jsonl"
            if trajectory.exists() and trajectory.stat().st_size:
                raise FileExistsError("Output has a trajectory; restore its checkpoint or choose a new directory")
        self._ensure_output_manifest()
        started, completed = time.monotonic(), 0
        while self.state.step < self.total_steps:
            if max_steps is not None and completed >= max_steps:
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                break
            stage, controls = self.controls_at(self.state.step)
            key = np.array([self.seed, self.state.step], dtype=np.uint32)
            before = time.monotonic()
            function = self.step_fn[stage] if isinstance(self.step_fn, dict) else self.step_fn
            from .numerics import NumericalAuditError
            try:
                result = function(deepcopy(self.state), self.backend, deepcopy(controls), key)
            except NumericalAuditError as error:
                self._record_numerical_failure(error, key, controls)
                raise
            with np.errstate(over="ignore", invalid="ignore"):
                arrays = [np.asarray(result.logits, np.float32),
                          np.asarray(result.evaluated_logits, np.float32)]
            if any(x.shape != self.state.logits.shape or not np.isfinite(x).all() for x in arrays):
                raise FloatingPointError("Step produced invalid logits; state was not advanced")
            metrics = {k: float(v) for k, v in result.metrics.items()}
            if not np.isfinite(result.loss) or not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Step produced nonfinite loss/metrics; state was not advanced")
            best_loss, best_logits = self.state.best_loss, self.state.best_logits
            effective_controls = deepcopy(controls if result.effective_controls is None
                                          else result.effective_controls)
            best_controls = self.state.best_controls
            if best_loss is None or result.loss < best_loss:
                best_loss, best_logits = float(result.loss), arrays[1].copy()
                best_controls = deepcopy(effective_controls)
            row = {"step": self.state.step, "stage": stage, "loss": float(result.loss),
                   "metrics": metrics,
                   "controls": effective_controls,
                   "requested_controls": controls,
                   **self._sequence_fields("evaluated", arrays[1], effective_controls),
                   "seconds": time.monotonic() - before}
            next_state = State(arrays[0].copy(), self.state.step + 1,
                               result.optimizer, result.extra, best_loss, best_logits,
                               effective_controls, best_controls)
            # Validate serialization before committing user-provided state or controls.
            from .checkpoint import encode
            encode(asdict(next_state))
            encoded_row = json.dumps(row, allow_nan=False) + "\n"
            self.state = next_state
            completed += 1
            if self.output:
                with (self.output / "trajectory.jsonl").open("a") as handle:
                    handle.write(encoded_row)
                if self.state.step % checkpoint_every == 0:
                    self.save(self.output / "checkpoint.npz")
            stop = any(value is False for value in [hook(self, row) for hook in self.observers])
            if stop:
                break
        if self.output:
            self.save(self.output / "checkpoint.npz")
            summary = {"completed_steps": self.state.step, "total_steps": self.total_steps,
                       **self._sequence_fields("next", self.state.logits,
                                               self.state.last_controls),
                       "next_sequence_context": "Unevaluated next state decoded under last effective controls",
                       "best_evaluated_loss": self.state.best_loss,
                       **self._sequence_fields("best_evaluated", self.state.best_logits,
                                               self.state.best_controls),
                       "best_comparison": "Lowest evaluated objective under its recorded controls. "
                                          "A decoded hard sequence needs its own hard evaluation."}
            from .checkpoint import atomic_text
            atomic_text(self.output / "summary.json", json.dumps(summary, indent=2) + "\n")
        return self.state

    def save(self, path):
        from .checkpoint import save
        save(path, {"fingerprint": self.fingerprint, "state": asdict(self.state)})

    def restore(self, path):
        from .checkpoint import load
        saved = load(path)
        if not isinstance(saved, dict) or not {"fingerprint", "state"} <= saved.keys():
            raise ValueError("Invalid run checkpoint: missing fingerprint or state")
        if saved["fingerprint"] != self.fingerprint:
            raise ValueError("Checkpoint fingerprint differs: code/runtime/problem/backend/schedule/seed changed")
        if not isinstance(saved["state"], dict):
            raise ValueError("Invalid run checkpoint: state must be a mapping")
        state = State(**saved["state"])
        if not isinstance(state.logits, np.ndarray) or state.logits.shape != (self.backend.design_length, 20):
            raise ValueError("Checkpoint sequence dimensions differ")
        if type(state.step) is not int or not 0 <= state.step <= self.total_steps:
            raise ValueError("Checkpoint step outside schedule")
        if ((state.best_loss is None) != (state.best_logits is None)
                or state.best_loss is not None and (
                    type(state.best_loss) not in (float, int) or not np.isfinite(state.best_loss)
                    or not isinstance(state.best_logits, np.ndarray)
                    or state.best_logits.shape != state.logits.shape
                    or state.best_logits.dtype != np.float32 or not np.isfinite(state.best_logits).all())):
            raise ValueError("Invalid checkpoint best score or logits")
        if state.logits.dtype != np.float32 or not np.isfinite(state.logits).all():
            raise ValueError("Checkpoint contains nonfinite logits")
        # A trajectory may contain steps beyond the last atomic checkpoint.
        if self.output and (self.output / "trajectory.jsonl").exists():
            path = self.output / "trajectory.jsonl"
            rows = []
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise ValueError("Malformed trajectory before its final interrupted record")

            # Restoring into a new directory may legitimately begin a log at a
            # nonzero step. Once begun, the log must be ordered and contiguous.
            previous = None
            for row in rows:
                if (not isinstance(row, dict) or type(row.get("step")) is not int
                        or row["step"] < 0 or previous is not None and row["step"] != previous + 1):
                    raise ValueError("Invalid trajectory step record")
                previous = row["step"]
            if previous is not None and previous < state.step - 1:
                raise ValueError("Trajectory ends before the checkpoint")
            from .checkpoint import atomic_text
            atomic_text(path, "".join(json.dumps(r) + "\n" for r in rows
                                      if r["step"] < state.step))
        self.state = state
        self._restored = True
        return self
