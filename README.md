[English](README.md) | [简体中文](README.zh-CN.md)

[Overview](docs/index.html) · [Install](docs/install.html) · [Use](docs/guide.html) · [API](docs/api.html) · [Numerics](docs/numerics.html) · [Migration](docs/migration.html) · [Attribution](docs/provenance.html) · [Contribute](docs/contributing.html)

Hallucination Lab · 0.3.0rc3 · Release candidate

# Protein hallucination, open to customization.

Customize an objective, choose AF2 or AF3, and control sequence optimization from ordinary Python.

The framework provides model adapters, mathematical objectives, a replaceable optimization step, staged controls, checkpoints, and explicit evaluation records. Model runtimes and weights are separate installations.

## Start with a numerical example

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[jax]'
.venv/bin/hallulab doctor
.venv/bin/hallulab run hallucination_lab.toy:build --max-steps 3
.venv/bin/hallulab run hallucination_lab.toy:build --resume runs/toy/checkpoint.npz
```

Run these commands from the unpacked source directory. The toy is an artificial quadratic objective, not a protein predictor. For a wheel install, pass the wheel path to pip and install JAX in the same environment.

## What you can customize

| Layer | Public interface |
|----|----|
| Model | AF2, AF3, Functional and Composite backends. |
| Objective | Distribution cross-entropy, distance events, exclusions, contacts, weighted combinations, or a custom JAX callable. |
| Optimization | Stage controls and ramps, Adam or SGD, gradient transforms, caller-defined steps and observers. |
| Records | Source/runtime identity, random keys, controls, evaluated sequence identity, atomic checkpoints and explicit exports. |

## Computational scope

AF2 uses a separately installed ColabDesign runtime. AF3 differentiates through its adapted trunk and distogram paths. AF3 diffusion coordinates and confidence are forward-only. Hard-sequence steps use a declared straight-through surrogate.

The software exposes mathematical objectives. Lower objective values do not by themselves establish folding, affinity or biological function. See the numerical contract before changing runtimes or gradient semantics.

## Package contents

The wheel contains the importable library and CLI. The source package additionally contains HTML documentation, examples, runtime profiles, software tests and build checks. No model weights, databases or project-specific design datasets are bundled.

Independent research software. Project code: Apache-2.0. External runtimes, model parameters and outputs retain their own terms.
