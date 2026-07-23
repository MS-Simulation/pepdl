"""Live Koina predict smoke-test (the torch-free remote inference path). Skips if koinapy or the Koina
server is unavailable — so the suite stays green offline, but exercises a real remote predict when it can."""
import pytest
import pandas as pd

koinapy = pytest.importorskip("koinapy")


def _koina_up():
    try:
        import urllib.request
        urllib.request.urlopen("https://koina.wilhelmlab.org/", timeout=8)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _koina_up(), reason="Koina server unreachable")
def test_koina_rt_predict():
    from pepdl.rt.predictors import predict_retention_time_with_koina
    df = pd.DataFrame({"sequence": ["PEPTIDEK", "SAMPLERPEPTIDER", "ELVISLIVESK"]})
    out = predict_retention_time_with_koina("Deeplc_hela_hf", df)
    # a prediction column per input peptide, finite values
    assert len(out) == 3
    import numpy as np
    pred_cols = [c for c in out.columns if "rt" in c.lower() or "predict" in c.lower() or "irt" in c.lower()]
    assert pred_cols, f"no RT prediction column in {list(out.columns)}"
    assert np.isfinite(out[pred_cols[0]].to_numpy(dtype=float)).all()


@pytest.mark.skipif(not _koina_up(), reason="Koina server unreachable")
def test_koina_intensity_predict():
    """The fragments-job intensity path: ModelFromKoina.predict with Koina's canonical column names."""
    import pandas as pd
    from pepdl.koina_models.access_models import ModelFromKoina
    for seqs in (["PEPTIDEK"], ["PEPTIDEK", "SAMPLERPEPTIDER", "ELVISLIVESK"]):
        df = pd.DataFrame({"peptide_sequences": [str(s) for s in seqs],
                           "precursor_charges": [2] * len(seqs),
                           "collision_energies": [30.0] * len(seqs)})
        out = ModelFromKoina(model_name="Prosit_2020_intensity_HCD").predict(df)
        assert {"annotation", "intensities"} <= set(out.columns)
        assert len(out) > 0
