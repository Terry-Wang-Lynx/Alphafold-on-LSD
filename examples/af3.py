"""AF3 distogram research. Supply prepared input JSON, model_dir, and a pair mask.

Pair mask .npy is indexed by AF3 tokens, before bucket padding.
    PYTHONPATH=examples hallulab run af3:build --config my_af3.json
"""
import numpy as np
from hallucination_lab import Run, initial_logits
import stages as stage_module
from hallucination_lab.backends.af3 import AF3
from hallucination_lab.losses import interval, weighted
from hallucination_lab.provenance import experiment_id


def build(config):
    required = ("input_json", "pair_mask", "model_dir", "design_chain")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"AF3 example config requires: {', '.join(missing)}")
    mask = np.load(config["pair_mask"], allow_pickle=False)
    edges = np.linspace(2.3125, 21.6875, 63)
    objective = weighted({"range": (1., interval(mask, edges, lower=config.get("lower", 0),
                                               upper=config.get("upper", 8)))})
    backend = AF3(input_json=config["input_json"], design_chain=config["design_chain"],
                  objective=objective, model_dir=config["model_dir"],
                  design_positions=config.get("design_positions"), bucket=config.get("bucket", 256),
                  recycles=config.get("recycles", 0))
    return Run(backend, initial_logits(backend.design_length, seed=7),
               stage_module.make_stages(config),
               problem_id=experiment_id(files=[__file__, stage_module.__file__, config["input_json"], config["pair_mask"]],
                                        settings={k: v for k, v in config.items()
                                                  if k not in {"output", "data_dir", "model_dir", "input_json", "pair_mask"}}),
               output=config.get("output", "runs/af3"), seed=7)
