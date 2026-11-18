"""Host-side admissibility checks for the explicit query-only relaxation."""

import hashlib
import numpy as np
from ...sequence import ALPHABET


def require_query_only_input(data):
    proteins = [entity["protein"] for entity in data["sequences"] if "protein" in entity]
    if not proteins:
        raise ValueError("query_only relaxation requires protein input")
    for protein in proteins:
        if (
            protein.get("unpairedMsa") != ""
            or protein.get("pairedMsa") != ""
            or protein.get("templates") != []
            or any(key.endswith("Path") for key in protein)
        ):
            raise ValueError(
                "query_only requires explicit empty paired/unpaired MSA and templates; no paths"
            )


def audit_query_only_batch(batch, protein_ids, sequence):
    """Check real native features BEFORE converting/caching a new hard carrier.

    Multiple identical full-query rows are valid. Partial/gapped or homolog
    rows are not. Padding rows are retained and never made differentiable.
    """
    ids = np.asarray(protein_ids)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(ids) != len(sequence)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("query_only needs the complete canonical protein token map")
    if not sequence or any(aa not in ALPHABET for aa in sequence):
        raise ValueError("query_only requires the complete standard-amino-acid hard sequence")
    aatype = np.asarray(batch["aatype"])[ids]
    if not np.array_equal(aatype, [ALPHABET.index(aa) for aa in sequence]):
        raise ValueError(
            "query_only carrier native aatype does not match the complete hard sequence"
        )
    rows = np.asarray(batch["msa"])
    mask = np.asarray(batch["msa_mask"], bool)
    profile = np.asarray(batch["profile"])[ids]
    active = np.flatnonzero(mask[:, ids].any(axis=1))
    if (
        not len(active)
        or active[0] != 0
        or not mask[np.ix_(active, ids)].all()
        or not np.all(rows[np.ix_(active, ids)] == aatype[None, :])
    ):
        raise ValueError("query_only carrier has a missing, partial or non-query effective MSA row")
    expected = np.eye(profile.shape[-1], dtype=profile.dtype)[aatype]
    if not np.array_equal(profile, expected):
        raise ValueError("query_only carrier native profile must equal hard query one-hot exactly")
    deletion = np.asarray(batch["deletion_matrix"])[:, ids]
    if (
        np.any(deletion[mask[:, ids]] != 0)
        or np.any(np.asarray(batch["deletion_mean"])[ids] != 0)
        or np.asarray(batch["template_atom_mask"], bool).any()
    ):
        raise ValueError(
            "query_only carrier must have zero deletion features and no template atoms"
        )
    return {
        "schema": "af3_query_only_carrier_audit/1",
        "passed": True,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "protein_token_count": len(ids),
        "effective_query_row_indices": active.tolist(),
        "effective_query_copy_count": len(active),
        "unique_effective_query_count": 1,
        "native_profile_exact_query_onehot": True,
        "deletion_features_zero": True,
        "no_effective_template_atoms": True,
        "padded_rows_retained": True,
    }
