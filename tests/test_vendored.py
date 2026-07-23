"""Regression vectors for pepdl._vendored — pin the vendored primitives against known values so a future
mscorepy/vendor change can't silently shift them. (ccs/mz delegate to mscorepy; the rest are pure copies.)"""
import numpy as np
from pepdl import _vendored as V


def test_calculate_mz():
    assert abs(V.calculate_mz(1000.0, 2) - 501.007276) < 1e-5


def test_ccs_roundtrip_and_value():
    k0 = V.ccs_to_one_over_k0(350.0, 500.0, 2)
    assert abs(k0 - 0.861920) < 1e-4          # pinned (mscorepy ccs<->1/k0)
    ccs = V.one_over_k0_to_ccs(k0, 500.0, 2)
    assert abs(ccs - 350.0) < 1e-3            # round-trips


def test_linear_map():
    assert V.linear_map(5.0, 0.0, 10.0, 0.0, 100.0) == 50.0


def test_remove_unimod_annotation():
    assert V.remove_unimod_annotation("PEPT[UNIMOD:21]IDE") == "PEPTIDE"


def test_tokenize_unimod_sequence():
    assert V.tokenize_unimod_sequence("PEPT[UNIMOD:21]IDE") == \
        ['<START>', 'P', 'E', 'P', 'T[UNIMOD:21]', 'I', 'D', 'E', '<END>']


def test_flatten_prosit_array():
    out = V.flatten_prosit_array(np.ones((29, 2, 3)))
    assert out.shape == (174,) and out.sum() == 174.0
    flat = np.arange(174.0)
    assert np.array_equal(V.flatten_prosit_array(flat), flat)   # already-flat passthrough
