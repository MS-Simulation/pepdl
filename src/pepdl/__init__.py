"""pepdl — peptide deep-learning property predictors (inference): CCS, RT, MS2 fragment intensity, charge.

timsTOF-free and imspy-free: backed by the `mscorepy` primitives wheel (mscore + ms-chem). PyTorch is an
optional `[local]` extra; the Koina path is torch-free. Training lives in the separate `pepdl-train`.
"""
from pepdl._peptide import PeptideSequence, PeptideProductIonSeriesCollection

__all__ = ["PeptideSequence", "PeptideProductIonSeriesCollection"]

# Predictors are imported from their submodules to keep optional deps (torch/koina) lazy, e.g.:
#   from pepdl.ccs.predictors import DeepPeptideIonMobilityApex
#   from pepdl.rt.predictors import predict_retention_time_with_koina
#   from pepdl.intensity.predictors import Prosit2023TimsTofWrapper, DeepPeptideIntensityPredictor
