"""Contract test for the intensity repoint (imspy_core.data -> pepdl._peptide over mscorepy).
Values cross-checked against imspy_core: the fragment-series JSON is BYTE-IDENTICAL (same Rust); the peptide
mass matches within 1e-5 (the current-mscore compute-from-elements value; imspy_core@old differed at 1e-6)."""
import hashlib
from pepdl import PeptideSequence

# reference: verified equal to imspy_core.data.PeptideSequence (fragment JSON identical; mass within 1e-6)
CASES = {
    "PEPTIDER":            (955.461075,  "9cbeeac1e505d5fe"),
    "SAMPLERPEPTIDEK":     (1711.845086, "a47b679e2ca038cc"),
    "PEPT[UNIMOD:21]IDEK": (1007.421258, "e7d29ecf5d63c747"),
}


def test_mass_and_fragment_series_contract():
    for seq, (mass, jsha) in CASES.items():
        p = PeptideSequence(seq)
        assert abs(p.mono_isotopic_mass - mass) < 1e-5, seq
        coll = p.associate_fragment_ion_series_with_prosit_intensities([0.1] * 174, 2, "b")
        assert len(coll.series) == 2                       # b + y
        got = hashlib.sha256(coll.to_json().encode()).hexdigest()[:16]
        assert got == jsha, f"{seq}: fragment JSON changed ({got} != {jsha})"


def test_find_series_and_fragment_type_validation():
    p = PeptideSequence("PEPTIDER")
    coll = p.associate_fragment_ion_series_with_prosit_intensities([0.1] * 174, 2, "y")
    assert coll.find_series(2) is not None
    try:
        p.associate_fragment_ion_series_with_prosit_intensities([0.1] * 174, 2, "q")
        assert False, "invalid fragment type must raise"
    except AssertionError as e:
        assert "Invalid fragment type" in str(e)
