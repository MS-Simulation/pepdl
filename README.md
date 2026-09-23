# pepdl

Peptide deep-learning **property predictors** for mass spectrometry — **inference only**.

`pepdl` predicts four properties from a peptide sequence (ProForma / UNIMOD bracket notation):

| Property | Module | Local class | Koina helper |
| --- | --- | --- | --- |
| **CCS / ion mobility** | `pepdl.ccs` | `DeepPeptideIonMobilityApex` | `predict_inverse_ion_mobility_with_koina` |
| **Retention time** | `pepdl.rt` | `DeepChromatographyApex`, `Chronologer` | `predict_retention_time_with_koina` |
| **MS2 fragment intensity** | `pepdl.intensity` | `DeepPeptideIntensityPredictor` | `Prosit2023TimsTofWrapper`, `predict_fragment_intensities_with_koina` |
| **Charge state / flyability** | `pepdl.ionization` | `DeepChargeStateDistribution`, `BinomialChargeStateDistributionModel` | `predict_peptide_flyability_with_koina` |

Training code is **not** here — it lives in the sibling repo
[`pepdl-train`](https://github.com/MS-Simulation/pepdl-train). This package only loads
pretrained weights and runs forward passes.

## Two backends

* **Local** — PyTorch checkpoints run on your own CPU/GPU. Requires the `[local]` extra.
* **Koina** — remote inference against the public [Koina](https://koina.wilhelmlab.org) server
  (default host `koina.wilhelmlab.org:443`). This path is **fully torch-free**, so `pepdl[koina]`
  works on a machine with no CUDA, no PyTorch, and no compiler.

Torch is *optional*: importing `pepdl` or any predictor module never pulls it in. Local model
classes call `pepdl.utility.require_torch()` at construction time, which raises an `ImportError`
with an actionable `pip install 'pepdl[local]'` hint if torch is missing.

## Independence

`pepdl` is **imspy-free** and **timsTOF-free**. Chemistry, peptide, and fragment-series primitives
come from [`mscorepy`](https://github.com/MS-Simulation/mscore) — a lean Rust (PyO3) wheel
wrapping `mscore` + `ms-chem`. This is enforced by `tests/test_import_gate.py`, which imports the
inference surface in a fresh subprocess and asserts that none of `imspy_*`, `ms_io`, `torch`,
`koinapy`, or `numba` ended up in `sys.modules`.

## Install

```bash
pip install pepdl                # core: numpy/pandas/scipy/tqdm + mscorepy. Torch-free.
pip install 'pepdl[koina]'       # + koinapy, for remote prediction. Still torch-free.
pip install 'pepdl[local]'       # + torch and the searlelab Chronologer (on-device stack)
pip install 'pepdl[rt]'          # just the default RT backend (Chronologer + torch)
pip install 'pepdl[dev]'         # + pytest
```

Python >= 3.11.

## Model weights

Weights are **not bundled** in the wheel. They are downloaded on first use from GitHub Releases
(tag `models-v0.5.0`) and verified against a pinned **SHA-256** before being moved into place
(see `pepdl/pretrained/hub.py`). The cache directory is:

```
$IMSPY_CACHE_DIR             # if set, used verbatim
~/.cache/imspy/models/v0.5.0/   # default
```

`IMSPY_CACHE_DIR` / `~/.cache/imspy` is a **legacy name** kept for compatibility with existing
imspy installs — but the variable and path are spelled exactly as above and are what `pepdl`
actually reads. A checkpoint found next to `pepdl/pretrained/` (editable/source installs) wins
over the cache.

The **Chronologer** base weights (default RT backend) are fetched directly from upstream
[searlelab/chronologer](https://github.com/searlelab/chronologer) and deliberately **not
re-hosted**, so upstream fixes propagate automatically (Apache-2.0; attribution preserved in
`pepdl.rt.chronologer`).

## Usage

```python
# Remote (Koina) — no torch required
from pepdl.intensity import Prosit2023TimsTofWrapper
from pepdl.rt import predict_retention_time_with_koina
from pepdl.ccs import predict_inverse_ion_mobility_with_koina

# Local (needs pepdl[local])
from pepdl.ccs import DeepPeptideIonMobilityApex
from pepdl.intensity import DeepPeptideIntensityPredictor

predictor = DeepPeptideIonMobilityApex()
inv_mob = predictor.simulate_ion_mobilities(
    sequences=["PEPTIDE", "SEQUENCE"],
    charges=[2, 3],
    mz=[400.0, 350.0],
)

intensities = DeepPeptideIntensityPredictor().predict_intensities(
    sequences=["PEPTIDE", "SEQUENCE"],
    charges=[2, 3],
    collision_energies=[30.0, 30.0],
)  # list of (29, 2, 3) Prosit-layout arrays
```

## Also included

* `pepdl.mixture` — `GaussianMixtureModel`, a torch-backed GMM for clustering spectral data.
* `pepdl.hashing` — random-projection (SimHash) locality-sensitive hashing for fast approximate
  cosine nearest-neighbour search over spectra: `CosimHasher`, `TimsHasher`, `SpectralHasher`.

Both are torch-gated in the same optional way as the local predictors.

Run the test suite with `python -m pytest tests -q`.
