"""`DeepPeptideIntensityPredictor(instrument=...)` reaches the model's instrument embedding.

The shipped intensity model was fine-tuned on timsTOF spectra labelled "timstof"; without an
instrument the model falls back to "unknown". On measured timsTOF spectra "timstof" is better
(fixture median spectral angle 0.854 vs 0.826; 76% of spectra improve).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mscorepy")

from test_padding_mask_models import (  # noqa: E402
    SEQS, _bare, _intensity_predictor, _measured_ms2, _spectral_angle)


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.instruments = []
        self.max_seq_len = 50

    def forward(self, tokens, charge=None, collision_energy=None, padding_mask=None, instrument=None):
        self.instruments.append(instrument)
        return {"intensity": torch.zeros(tokens.shape[0], 174)}


def _recording(instrument_id):
    apex = _bare("pepdl.intensity.predictors", "DeepPeptideIntensityPredictor")
    apex._torch = torch
    apex.instrument_id = instrument_id
    apex.model = rec = _Recorder()
    apex._predict_batch(SEQS, [2] * len(SEQS), [0.3] * len(SEQS), batch_size=3)
    return rec.instruments


def test_no_instrument_passes_none_as_before():
    assert _recording(None) == [None, None]


def test_an_instrument_reaches_every_batch():
    from pepdl.models.heads import INSTRUMENT_TO_ID

    got = _recording(INSTRUMENT_TO_ID["timstof"])
    assert len(got) == 2 and all(t is not None for t in got)
    assert torch.cat(got).tolist() == [INSTRUMENT_TO_ID["timstof"]] * len(SEQS)


def test_an_unknown_instrument_name_is_refused_before_loading_weights(monkeypatch):
    import pepdl.intensity.predictors as ip

    def boom():
        raise AssertionError("weights were loaded before the name was checked")

    monkeypatch.setattr(ip, "load_deep_intensity_predictor", boom)
    with pytest.raises(ValueError, match="unknown instrument 'timsTOF'"):
        ip.DeepPeptideIntensityPredictor(verbose=False, instrument="timsTOF")


def test_timstof_improves_measured_timstof_spectra():
    args, y = _measured_ms2()
    n = len(y)
    base = _spectral_angle(y, _intensity_predictor()._predict_batch(*args).reshape(n, -1))
    tof = _spectral_angle(
        y, _intensity_predictor(instrument="timstof")._predict_batch(*args).reshape(n, -1))
    assert np.median(tof) >= 0.81, np.median(tof)
    assert np.mean(tof - base) > 0.005, np.mean(tof - base)
