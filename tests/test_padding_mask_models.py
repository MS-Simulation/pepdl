"""The padding mask reaches the model, and the CCS it produces is right.

Two halves:

* **Wiring** (no weights): a stand-in model records the ``padding_mask`` every predictor passes.
  Before the fix, CCS / RT / intensity passed none.
* **Behaviour** (real weights, skipped when they cannot be loaded): a prediction must not depend on
  what else is in the batch (before the fix, batching a peptide with a longer one changed the
  padding it saw and so its prediction), and the CCS of a committed ``meier_test`` sample must be
  within the gates below. Unmasked, 2+ was off by +35 Å².

Fixture: ``tests/data/ccs_meier_test_fixture.tsv`` is 100 ions per charge (2, 3, 4) drawn with
``random_state=7`` from the ``meier_test`` origin of the ionmob test split
(huggingface ``theGreatHerrLebert/ionmob``), which the shipped checkpoint was not trained on.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mscorepy")

FIXTURE = Path(__file__).parent / "data" / "ccs_meier_test_fixture.tsv"
SEQS = ["PEPTIDEK", "K", "ACDEFGHIKLMNPQRSTVWY", "PEPTIDEK-[UNIMOD:258]"]


def _bare(module, name):
    import importlib

    from pepdl.utilities.tokenizers import ProformaTokenizer

    cls = getattr(importlib.import_module(module), name)
    obj = cls.__new__(cls)
    obj.tokenizer = ProformaTokenizer.with_defaults()
    obj._device = torch.device("cpu")
    obj.verbose = False
    return obj


def _expected_mask(seqs, pad_len=50):
    from pepdl.utilities.padding import token_ids_and_mask
    from pepdl.utilities.tokenizers import ProformaTokenizer

    return torch.as_tensor(token_ids_and_mask(ProformaTokenizer.with_defaults(), seqs, pad_len)[1])


class _Recorder(torch.nn.Module):
    """Stands in for UnifiedPeptideModel and records each call's padding_mask."""

    def __init__(self):
        super().__init__()
        self.masks = []
        self.max_seq_len = 50

    def predict_ccs(self, tokens, mz, charge, padding_mask=None, instrument=None):
        self.masks.append(padding_mask)
        n = tokens.shape[0]
        return torch.full((n, 1), 400.0), torch.ones(n, 1)

    def predict_rt(self, tokens, padding_mask=None, instrument=None):
        self.masks.append(padding_mask)
        return torch.zeros(tokens.shape[0], 1)

    def forward(self, tokens, charge=None, collision_energy=None, padding_mask=None, **kw):
        self.masks.append(padding_mask)
        return {"intensity": torch.zeros(tokens.shape[0], 174)}


def _assert_masks(rec, seqs, batch_size):
    want = _expected_mask(seqs)
    assert rec.masks and all(m is not None for m in rec.masks), "a forward call got no padding_mask"
    got = torch.cat([m.cpu() for m in rec.masks])
    assert torch.equal(got, want)
    assert len(rec.masks) == -(-len(seqs) // batch_size)


def test_ccs_passes_the_mask_on_every_batch():
    apex = _bare("pepdl.ccs.predictors", "DeepPeptideIonMobilityApex")
    apex.model = rec = _Recorder()
    apex.simulate_ion_mobilities(SEQS, [2] * len(SEQS), [500.0] * len(SEQS), batch_size=3)
    _assert_masks(rec, SEQS, 3)


def test_rt_passes_the_mask_on_every_batch():
    apex = _bare("pepdl.rt.predictors", "DeepChromatographyApex")
    apex.model = rec = _Recorder()
    apex.simulate_separation_times(SEQS, batch_size=3)
    _assert_masks(rec, SEQS, 3)


def test_intensity_passes_the_mask_on_every_batch():
    apex = _bare("pepdl.intensity.predictors", "DeepPeptideIntensityPredictor")
    apex._torch = torch
    apex.model = rec = _Recorder()
    apex._predict_batch(SEQS, [2] * len(SEQS), [0.3] * len(SEQS), batch_size=3)
    _assert_masks(rec, SEQS, 3)


# ── behaviour, with the shipped weights ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ccs_apex():
    from pepdl.ccs import DeepPeptideIonMobilityApex

    try:
        return DeepPeptideIonMobilityApex(verbose=False)
    except Exception as e:  # weights absent and not downloadable here
        pytest.skip(f"CCS weights unavailable: {e}")


def test_ccs_does_not_depend_on_the_rest_of_the_batch(ccs_apex):
    """Alone, the peptide is padded from its own length; next to a 40-mer, the PAD region is
    different. Masked, the model sees neither, so the prediction must not move."""
    seq, z, mz = "LVNELTEFAK", 2, 582.3190
    alone = ccs_apex.simulate_ion_mobilities([seq], [z], [mz], return_ccs=True)[0]
    batched = ccs_apex.simulate_ion_mobilities([seq, "A" * 40 + "K"], [z, 3], [mz, 900.0],
                                               return_ccs=True)[0]
    assert abs(alone - batched) < 1e-3, (alone, batched)


@pytest.mark.parametrize("model_attr", ["intensity", "rt"])
def test_other_heads_do_not_depend_on_the_rest_of_the_batch(model_attr):
    if model_attr == "intensity":
        from pepdl.intensity.predictors import DeepPeptideIntensityPredictor as P
    else:
        from pepdl.rt.predictors import DeepChromatographyApex as P
    try:
        apex = P()
    except Exception as e:
        pytest.skip(f"{model_attr} weights unavailable: {e}")
    seq = "LVNELTEFAK"
    if model_attr == "intensity":
        alone = apex._predict_batch([seq], [2], [0.3])[0]
        batched = apex._predict_batch([seq, "A" * 40 + "K"], [2, 3], [0.3, 0.3])[0]
    else:
        alone = apex.simulate_separation_times([seq])[0]
        batched = apex.simulate_separation_times([seq, "A" * 40 + "K"])[0]
    np.testing.assert_allclose(alone, batched, atol=1e-4)


def test_ccs_matches_held_out_measurements_per_charge(ccs_apex):
    """Gates (Å²), from PLAN-pepdl-padding-mask.md: 2+ |median| <= 6 and MAE <= 9; 3+ and 4+
    MAE <= 20; MAE weighted to the test split's charge mix (2+ 67%, 3+ 27%, 4+ 5%) <= 10.
    Shipped without the mask: 2+ median +35, MAE about 35."""
    import pandas as pd

    d = pd.read_csv(FIXTURE, sep="\t")
    pred = ccs_apex.simulate_ion_mobilities(d.sequence_modified.tolist(), d.charge.tolist(),
                                            d.mz.tolist(), return_ccs=True)
    err = np.asarray(pred) - d.ccs.to_numpy()
    by = {z: err[d.charge.to_numpy() == z] for z in (2, 3, 4)}
    mae = {z: float(np.mean(np.abs(e))) for z, e in by.items()}
    report = {z: (round(float(np.median(e)), 1), round(mae[z], 1)) for z, e in by.items()}
    assert abs(np.median(by[2])) <= 6 and mae[2] <= 9, report
    assert mae[3] <= 20 and mae[4] <= 20, report
    weighted = (0.67 * mae[2] + 0.27 * mae[3] + 0.05 * mae[4]) / 0.99
    assert weighted <= 10, (weighted, report)
