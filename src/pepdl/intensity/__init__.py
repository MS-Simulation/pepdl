"""Fragment intensity prediction module."""

from pepdl.intensity.predictors import (
    IonIntensityPredictor,
    Prosit2023TimsTofWrapper,
    DeepPeptideIntensityPredictor,
    load_deep_intensity_predictor,
    remove_unimod_annotation,
    predict_fragment_intensities_with_koina,
)

from pepdl.intensity.utility import (
    post_process_predicted_fragment_spectra,
    reshape_dims,
    reshape_flat,
    seq_to_index,
    to_prosit_tensor,
    get_prosit_intensity_flat_labels,
    normalize_base_peak,
    mask_outofrange,
    mask_outofcharge,
)

__all__ = [
    # Predictors
    'IonIntensityPredictor',
    'Prosit2023TimsTofWrapper',
    'DeepPeptideIntensityPredictor',
    # Loaders
    'load_deep_intensity_predictor',
    # Utilities
    'remove_unimod_annotation',
    'post_process_predicted_fragment_spectra',
    'reshape_dims',
    'reshape_flat',
    'seq_to_index',
    'to_prosit_tensor',
    'get_prosit_intensity_flat_labels',
    'normalize_base_peak',
    'mask_outofrange',
    'mask_outofcharge',
    # Koina
    'predict_fragment_intensities_with_koina',
]
