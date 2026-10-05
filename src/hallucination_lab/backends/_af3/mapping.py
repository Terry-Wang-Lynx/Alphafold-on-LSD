"""Resolve chain identities from AF3's actual token layout, never JSON order."""
import numpy as np


def _layout(batch):
    if "token_atoms_layout" not in batch:
        raise ValueError("AF3 token_atoms_layout is required to resolve chain identity")
    layout = batch["token_atoms_layout"]
    if isinstance(layout, np.ndarray):
        layout = layout.item()
    return layout


def atom_names(batch):
    """Copy exact native dense-slot names, including empty padding slots."""
    return np.asarray(_layout(batch).atom_name, dtype=str).copy()


def token_identity(batch):
    """Return native chain and residue IDs before object-valued features are removed."""
    layout = _layout(batch)
    names = np.asarray(layout.atom_name)
    chains = np.asarray(layout.chain_id)
    residues = np.asarray(layout.res_id)
    valid = np.asarray(batch["seq_mask"], bool)
    if names.ndim != 2 or names.shape[0] != len(valid) or chains.shape != names.shape or residues.shape != names.shape:
        raise ValueError("AF3 token atom layout dimensions disagree with token mask")
    token_chains, token_residues = [], []
    for active, atom_names, chain_ids, residue_ids in zip(valid, names, chains, residues):
        if not active:
            token_chains.append("")
            token_residues.append(0)
            continue
        occupied = atom_names != ""
        chain_ids, residue_ids = np.unique(chain_ids[occupied]), np.unique(residue_ids[occupied])
        if len(chain_ids) != 1 or len(residue_ids) != 1:
            raise ValueError("AF3 token must map to one chain and residue identity")
        token_chains.append(str(chain_ids[0]))
        token_residues.append(int(residue_ids[0]))
    return np.asarray(token_chains), np.asarray(token_residues)


def protein_tokens(chain_ids, residue_ids, chain, length):
    """Return chain-local sequence order using native one-based residue IDs."""
    indices = np.flatnonzero(np.asarray(chain_ids) == chain)
    order = np.argsort(np.asarray(residue_ids)[indices])
    indices = indices[order]
    if len(indices) != length or not np.array_equal(np.asarray(residue_ids)[indices], np.arange(1, length + 1)):
        raise ValueError("Designed chain must map once to every standard protein residue")
    return indices
