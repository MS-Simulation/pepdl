"""Vendored primitives for pepdl — pure-Python helpers copied from imspy-core / imspy-simulation so pepdl
does not depend on them. Behaviour is a faithful copy; the ccs/mz helpers delegate to `mscorepy`'s Rust
chemistry (the same routine imspy-core called via imspy_connector).

Provenance:
  ccs_to_one_over_k0, one_over_k0_to_ccs, calculate_mz : imspy_core.chemistry.{mobility,utility}
  linear_map                                           : imspy_core.utility.utilities
  remove_unimod_annotation, tokenize_unimod_sequence   : imspy_core.utility.sequence
  flatten_prosit_array                                 : imspy_simulation.utility
Regression vectors live in tests/test_vendored.py (pin these against the originals).
"""
from __future__ import annotations
import re
from typing import List

import numpy as np
import mscorepy

_chem = mscorepy.py_chemistry


def ccs_to_one_over_k0(ccs, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """Convert CCS to reduced ion mobility (1/k0). Delegates to mscorepy (was imspy_connector)."""
    return _chem.ccs_to_one_over_reduced_mobility(ccs, mz, charge, mass_gas, temp, t_diff)


def one_over_k0_to_ccs(one_over_k0, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """Convert reduced ion mobility (1/k0) to CCS. Delegates to mscorepy."""
    return _chem.one_over_reduced_mobility_to_ccs(one_over_k0, mz, charge, mass_gas, temp, t_diff)


def calculate_mz(mass: float, charge: int) -> float:
    """m/z from neutral mass + charge. Delegates to mscorepy."""
    return _chem.calculate_mz(mass, charge)


def linear_map(value, old_min, old_max, new_min=0.0, new_max=60.0):
    """Linear mapping from one domain to another (pure copy)."""
    scale = (new_max - new_min) / (old_max - old_min)
    offset = new_min - old_min * scale
    return value * scale + offset


def remove_unimod_annotation(sequence: str) -> str:
    """Remove UNIMOD annotations from a peptide sequence (pure copy)."""
    return re.sub(r'\[UNIMOD:\d+\]', '', sequence)


def tokenize_unimod_sequence(unimod_sequence: str) -> List[str]:
    """Tokenize a UNIMOD-annotated sequence (pure copy)."""
    token_pattern = r'[A-Z](?:\[UNIMOD:\d+\])?'
    if unimod_sequence.startswith("[UNIMOD:1]"):
        special_token = "<START>[UNIMOD:1]"
        rest = unimod_sequence[len("[UNIMOD:1]"):]
        return [special_token] + re.findall(token_pattern, rest) + ['<END>']
    return ['<START>'] + re.findall(token_pattern, unimod_sequence) + ['<END>']


def flatten_prosit_array(array):
    """Flatten a Prosit (29,2,3) intensity array to the flat 174 layout (pure copy)."""
    if array.ndim == 1:
        return array
    array_return = np.zeros(174)
    ptr = 0
    for c in range(3):
        array_return[ptr:ptr + 29] = array[:, 0, c]; ptr += 29
        array_return[ptr:ptr + 29] = array[:, 1, c]; ptr += 29
    return array_return
