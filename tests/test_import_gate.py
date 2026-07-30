"""The independence gate — importing pepdl inference must load NO imspy/koina/ms_io. Run in a fresh
subprocess so other tests (e.g. the koina importorskip) can't pollute sys.modules.

`torch` is deliberately NOT on this list, though it once was. The contract pepdl actually makes is
that torch is an *optional extra* (`pepdl[local]`): importing the package must work without it, and
a local model must then fail through `require_torch` with an install hint. That is a statement about
torch being ABSENT, and it is pinned in tests/test_torch_optional.py, which blocks the import to
check it honestly. Listing torch here instead asserted something different and much stronger — that
torch is never even LOADED when it *is* installed, i.e. that every use is lazy. pepdl does not do
that, has no reason to (these are the deep-learning predictor modules; a caller importing
`ccs.predictors` intends to predict), and the module-scope `try: import torch` has always loaded it
eagerly. So the entry made this gate red from the commit that introduced it while the separation it
was actually written to protect — imspy-freedom — passed the whole time.
"""
import subprocess, sys

_CODE = r"""
import sys
from pepdl.ccs.predictors import DeepPeptideIonMobilityApex
from pepdl.rt.predictors import predict_retention_time_with_koina
from pepdl.intensity.predictors import Prosit2023TimsTofWrapper, DeepPeptideIntensityPredictor
from pepdl.utilities import ProformaTokenizer
FORBIDDEN = ('imspy_core','imspy_connector','imspy_simulation','imspy_search','imspy_predictors',
             'ms_io','koinapy','numba')
bad = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN)
assert bad == [], f'forbidden modules loaded: {bad}'
print('CLEAN')
"""


def test_inference_imports_are_imspy_free():
    r = subprocess.run([sys.executable, "-c", _CODE], capture_output=True, text=True)
    assert r.returncode == 0 and "CLEAN" in r.stdout, r.stderr or r.stdout
