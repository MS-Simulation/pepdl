"""pepdl — peptide deep-learning property predictors (inference). Backed by mscorepy; timsTOF-free."""
from ._peptide import PeptideSequence, PeptideProductIonSeriesCollection

__all__ = ["PeptideSequence", "PeptideProductIonSeriesCollection"]
