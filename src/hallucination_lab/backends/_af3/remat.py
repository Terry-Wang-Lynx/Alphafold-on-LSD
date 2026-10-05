# Copyright 2026 AlphaFold3 Hallucination Contributors
# Modifications Copyright 2026 Hallucination Lab contributors
# SPDX-License-Identifier: Apache-2.0
"""Local opt-in block rematerialization, no global monkey-patching."""
import haiku as hk

def maybe_remat(block_fn, enabled):
    return hk.remat(block_fn) if enabled else block_fn
