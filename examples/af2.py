"""A short AF2 research script. Supply a JSON config with data_dir and output.

    PYTHONPATH=examples hallulab run af2:build --config my_af2.json
"""
import numpy as np
from hallucination_lab import Run, initial_logits
import stages as stage_module
from hallucination_lab.backends.af2 import AF2
from hallucination_lab.losses import contact, weighted
from hallucination_lab.provenance import experiment_id


def build(config):
    missing = [key for key in ("data_dir",) if key not in config]
    if missing:
        raise ValueError(f"AF2 example config requires: {', '.join(missing)}")
    length = config.get("length", 32)
    mask = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :]) >= 8
    objective = weighted({"contact": (1.0, contact(mask, np.linspace(2.3125, 21.6875, 63),
                                                 cutoff=8.0, partners=1))})
    backend = AF2(sequence="A" * length, objective=objective, data_dir=config["data_dir"],
                  model_options={"use_bfloat16": config.get("bfloat16", False)},
                  recycles=config.get("recycles", 0))
    stages = stage_module.make_stages(config)
    return Run(backend, initial_logits(length, seed=7), stages,
               problem_id=experiment_id(files=[__file__, stage_module.__file__], settings={k: v for k, v in config.items()
                                                  if k not in {"output", "data_dir", "model_dir", "input_json", "pair_mask"}}),
               output=config.get("output", "runs/af2"), seed=7)
