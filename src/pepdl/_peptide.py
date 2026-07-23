"""Peptide primitives for pepdl, backed by `mscorepy` (mscore + ms-chem) — NO imspy-core / imspy_connector.

Thin wrappers matching the imspy_core.data.peptide API the predictors rely on, so predictor code repoints
by import only. Behaviour is a faithful port of imspy_core.data.peptide (the wrapper there only validated
`fragment_type` and delegated to the same Rust `associate_with_predicted_intensities`).
Provenance: ported from imspy-core src/imspy_core/data/peptide.py (associate_* wrapper).
"""
from __future__ import annotations
from typing import List, Optional

import mscorepy

_pep = mscorepy.py_peptide
_KNOWN_FRAGMENT_TYPES = {"a", "b", "c", "x", "y", "z"}


class PeptideProductIonSeriesCollection:
    """Wraps mscorepy's raw PyPeptideProductIonSeriesCollection (same API the predictor's callers used)."""
    def __init__(self, py_ptr):
        self._ptr = py_ptr

    @classmethod
    def from_py_ptr(cls, obj) -> "PeptideProductIonSeriesCollection":
        return cls(obj)

    def get_py_ptr(self):
        return self._ptr

    @property
    def series(self):
        return self._ptr.series

    def find_series(self, charge: int):
        return self._ptr.find_ion_series(charge)

    def to_json(self) -> str:
        return self._ptr.to_json()


class PeptideSequence:
    def __init__(self, sequence: str, peptide_id: Optional[int] = None):
        self._ptr = _pep.PyPeptideSequence(sequence, peptide_id)

    @classmethod
    def from_py_ptr(cls, obj) -> "PeptideSequence":
        inst = cls.__new__(cls)
        inst._ptr = obj
        return inst

    def get_py_ptr(self):
        return self._ptr

    @property
    def sequence(self) -> str:
        return self._ptr.sequence

    @property
    def mono_isotopic_mass(self) -> float:
        return self._ptr.mono_isotopic_mass

    def associate_fragment_ion_series_with_prosit_intensities(
        self, flat_intensities: List[float], charge: int,
        fragment_type: str = "b", normalize: bool = True, half_charge_one: bool = True,
    ) -> PeptideProductIonSeriesCollection:
        fragment_type = fragment_type.lower()
        assert fragment_type in _KNOWN_FRAGMENT_TYPES, (
            f"Invalid fragment type: {fragment_type}, must be one of {_KNOWN_FRAGMENT_TYPES}")
        result = self._ptr.associate_with_predicted_intensities(
            flat_intensities, charge, fragment_type, normalize, half_charge_one)
        return PeptideProductIonSeriesCollection.from_py_ptr(result)
