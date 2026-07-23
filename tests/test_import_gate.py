"""The independence gate: importing pepdl inference must load NO imspy/torch/koina (mscorepy is not imspy)."""
import sys

FORBIDDEN = ("imspy_core", "imspy_connector", "imspy_simulation", "imspy_search",
             "imspy_predictors", "ms_io", "torch", "koinapy", "numba")


def test_inference_imports_are_imspy_and_torch_free():
    from pepdl.ccs.predictors import DeepPeptideIonMobilityApex          # noqa: F401
    from pepdl.rt.predictors import predict_retention_time_with_koina    # noqa: F401
    from pepdl.intensity.predictors import Prosit2023TimsTofWrapper, DeepPeptideIntensityPredictor  # noqa
    from pepdl.utilities import ProformaTokenizer                        # noqa: F401
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)
    assert loaded == [], f"forbidden modules loaded on inference import: {loaded}"
