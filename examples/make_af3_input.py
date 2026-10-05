"""Create a small artificial single-chain input and mask; no model is run."""
import argparse
import json
from pathlib import Path
import numpy as np


def create(directory, length=32):
    if type(length) is not int or length < 9:
        raise ValueError("length must be an integer of at least 9")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = {"name": "interface_example", "modelSeeds": [0], "sequences": [
        {"protein": {"id": "A", "sequence": "A" * length,
                     "unpairedMsa": "", "pairedMsa": "", "templates": []}}],
        "dialect": "alphafold3", "version": 3}
    (directory / "input.json").write_text(json.dumps(data, indent=2) + "\n")
    i = np.arange(length)
    mask = (np.abs(i[:, None] - i[None, :]) >= 8).astype(np.float32)
    np.save(directory / "pair_mask.npy", mask, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example-input"))
    parser.add_argument("--length", type=int, default=32)
    args = parser.parse_args()
    create(args.output, args.length)
