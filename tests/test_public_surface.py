"""Guard the externally-facing surface: every exported name must actually resolve, and the
exported functions must not reference undefined globals. Both defects shipped once -- a deprecated
`get_collision_energy_calibration_factor` delegating to a `calibrate_nce` that lives in pepdl-train,
and `token_list_from_sequence` using `re` without importing it.

Scope note: these modules are torch-free by design, but nothing in THIS file proves that. Every test
here runs in the shared pytest process, where another test (or pytest itself) may already have
imported torch, so a `"torch" not in sys.modules` assertion here would be meaningless. The real
independence gate is `test_import_gate.py`: it imports the inference subpackages in a *fresh
subprocess* and asserts torch/imspy/koina/numba stayed unimported. Keep that proof there; this file
only checks that the exported names resolve."""
import importlib
import pytest

# Subpackages whose exports must resolve. (That they import without torch/koina/sagepy is proven in
# test_import_gate.py, in a subprocess — see the scope note above.)
NAME_RESOLUTION_MODULES = [
    "pepdl",
    "pepdl.ccs",
    "pepdl.ccs.utility",
    "pepdl.intensity",
    "pepdl.intensity.utility",
    "pepdl.utilities",
    "pepdl.koina_models",
]


@pytest.mark.parametrize("name", NAME_RESOLUTION_MODULES)
def test_every_exported_name_resolves(name):
    mod = importlib.import_module(name)
    missing = [s for s in getattr(mod, "__all__", []) if not hasattr(mod, s)]
    assert missing == [], f"{name}.__all__ exports unresolvable names: {missing}"


def test_token_list_from_sequence_runs():
    """Regression: this raised NameError ('re' was never imported)."""
    from pepdl.ccs import token_list_from_sequence

    assert token_list_from_sequence("AC[UNIMOD:4]DEK") == [
        "<SOS>", "A", "C", "[UNIMOD:4]", "D", "E", "K", "<EOS>",
    ]


def test_collision_energy_calibration_shim_is_gone():
    """calibrate_nce is rescoring/training-only and lives in pepdl_train.rescoring; the pepdl shim
    that called it always raised NameError, so it must stay deleted rather than come back."""
    import pepdl.intensity as intensity
    import pepdl.intensity.predictors as predictors

    for mod in (intensity, predictors):
        assert not hasattr(mod, "get_collision_energy_calibration_factor")
        assert not hasattr(mod, "calibrate_nce")
    assert "get_collision_energy_calibration_factor" not in intensity.__all__
