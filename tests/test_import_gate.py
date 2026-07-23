"""The independence gate — importing pepdl inference must load NO imspy/torch/koina. Run in a fresh
subprocess so other tests (e.g. the koina importorskip) can't pollute sys.modules."""
import subprocess, sys

_CODE = r"""
import sys
from pepdl.ccs.predictors import DeepPeptideIonMobilityApex
from pepdl.rt.predictors import predict_retention_time_with_koina
from pepdl.intensity.predictors import Prosit2023TimsTofWrapper, DeepPeptideIntensityPredictor
from pepdl.utilities import ProformaTokenizer
FORBIDDEN = ('imspy_core','imspy_connector','imspy_simulation','imspy_search','imspy_predictors',
             'ms_io','torch','koinapy','numba')
bad = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN)
assert bad == [], f'forbidden modules loaded: {bad}'
print('CLEAN')
"""


def test_inference_imports_are_imspy_and_torch_free():
    r = subprocess.run([sys.executable, "-c", _CODE], capture_output=True, text=True)
    assert r.returncode == 0 and "CLEAN" in r.stdout, r.stderr or r.stdout
