[English](README.md) | [简体中文](README.zh-CN.md)

Offline HTML documentation (`docs/`, open locally in a browser): [Overview](docs/index.html) · [Install](docs/install.html) · [Use](docs/guide.html) · [API](docs/api.html) · [Numerics](docs/numerics.html) · [Migration](docs/migration.html) · [Attribution](docs/provenance.html) · [Contribute](docs/contributing.html)

Hallucination Lab · `0.3.0rc3` · Release candidate  
Distribution: `hallucination-lab` · Python package: `hallucination_lab` · CLI: `hallulab`

# Protein hallucination, open to customization.

Customize a differentiable objective, choose AF2 or AF3 (or compose multiple conditions), and control sequence optimization from ordinary Python.

The framework provides import-light model adapters, explicit distogram objectives, a replaceable optimization step with float64 host-side optimizer moments, staged control ramps, atomic pickle-free checkpoints, fixed-input numerical audit gates, and explicit evaluation records. Model runtimes and weights are installed separately.

## Start with a numerical example

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[jax]'
.venv/bin/hallulab doctor
.venv/bin/hallulab run hallucination_lab.toy:build --max-steps 3
.venv/bin/hallulab run hallucination_lab.toy:build --resume runs/toy/checkpoint.npz
```

Run these commands from the source directory. The toy uses an artificial quadratic objective, not a protein predictor. For a wheel install, pass the wheel path to `pip` and install JAX in the same environment.

## What you can customize

| Layer | Public interface |
|----|----|
| Model | `AF2` (ColabDesign adapter), `AF3` (adapted trunk/distogram + forward-only structure sampling), `Functional` (custom differentiable backends), and `Composite` (chain-rule composition across conditions with explicit residue alignment). |
| Objective | `distribution` (masked cross-entropy), `interval` (distance-event probability or exclusion), `contact` (all-pair or top-k partner contact), `weighted` combinations with control-overridable weights, or any custom JAX callable returning a scalar or `(scalar, metrics)`. |
| Optimization | `Run`, `Stage` control/LR ramps, `gradient_step` (Adam/SGD in FP64 host moments, gradient clipping, custom transforms), `mutation_step` (gradient-free single-site proposals), per-stage step callables, and stopping observers. |
| Verification & Records | `audit_backend` / `compare_evaluations` numerical gates, `paired_evaluate` fixed-condition endpoint comparison, `preflight` (`hallulab doctor`), source/runtime fingerprints, deterministic per-step PRNG keys, atomic pickle-free checkpoints, and explicit exports (`save_prediction`, `write_fasta`). |

## Composable objectives and workflows

Explicit objectives replace hidden default model losses; include every desired term yourself:

```python
from hallucination_lab.losses import distribution, interval, weighted

objective = weighted({
    "shape": (1.0, distribution(target, pair_mask, edges=edges)),
    "range": (0.2, interval(other_mask, edges, lower=4.0, upper=12.0)),
})
```

See [examples/af2.py](examples/af2.py), [examples/af3.py](examples/af3.py), [examples/stages.py](examples/stages.py), and [examples/customize.py](examples/customize.py) for complete runnable factory scripts and custom step/objective patterns, and [docs/install.html](docs/install.html) for pinned Linux/CUDA runtime setup for AF2 and AF3.

## Repository layout and CPU verification

- `src/hallucination_lab/`: core optimization engine, sequence relaxation, distogram losses, numerical gates, paired assessment, provenance/diagnostics, and lazy-imported `AF2`, `AF3`, `Composite`, and `Functional` backends.
- `examples/`: standalone CLI factory scripts (`toy.py`, `af2.py`, `af3.py`, `stages.py`, `customize.py`, `make_af3_input.py`) and JSON config templates.
- `environments/`: pinned Linux/CUDA reference profiles and source verification (`verify_af2_source.py`).
- `tests/`: CPU test suite covering analytic float64 loss/gradient oracles, optimizer precision against 80-digit `Decimal` references, AF2/AF3 adapter contracts, numerical guards, checkpoint/trajectory recovery, and release allowlist hygiene.
- `docs/`: self-contained offline HTML documentation (installation, usage guide, API reference, numerical contracts, migration, attribution, contributing).
- `scripts/audit_release.py` and `release-files.json`: explicit public file allowlist and source-archive integrity audit.

Run the full CPU test, lint, and release-hygiene suite without GPU or model weights:

```sh
.venv/bin/python -m pip install '.[dev,jax]'
.venv/bin/python -m ruff check src tests examples scripts
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_release.py
```

## Computational scope

AF2 uses a separately installed ColabDesign runtime and differentiates through the final recycle with an optional same-state forward scalar guard. AF3 differentiates through its adapted trunk and distogram paths using a re-featurized hard atom carrier and soft query features (preserving native MSA profiles when homologs are present under `profile_mode="auto"`). AF3 diffusion coordinates and confidence remain forward-only; use `AF3.predict_hard_sequence()` to validate a locked sequence with native hard inputs. Hard-sequence optimization steps use a declared straight-through surrogate.

The software exposes mathematical objectives. Lower objective values do not by themselves establish folding, affinity, or biological function. Review the numerical contracts in [docs/numerics.html](docs/numerics.html) before changing runtimes or gradient semantics.

## Package contents and licensing

The wheel contains the importable library (`hallucination_lab`) and CLI (`hallulab`). The source distribution additionally contains the offline HTML documentation, examples, runtime profiles, software tests, and release audit checks. No model weights, databases, or project-specific design datasets are bundled.

Independent research software. Project code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE) for upstream attributions. External runtimes, model parameters, and model outputs retain their own terms.
