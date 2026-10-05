"""MLACS — Multi-Lag AutoCorrelation Shift detector.

Онлайн-обнаружение скрытых сдвигов автокорреляционной структуры временного
ряда (среднее и дисперсия ряда при этом могут не меняться).
"""
from .generators import (ar_psi, ar_variance, ar_acf, simulate_piecewise_ar,
                         Segment)
from .features import block_acf, block_spectrum, block_mean, block_logvar
from .detectors import (BlockCusum, MLACS, ACF1Detector, SpectralDetector,
                        MeanDetector, VarianceDetector, ARResidualDetector,
                        fit_ar_aic)
from .offline import offline_single_split
from .calibration import calibrate_threshold, calibrate_threshold_bootstrap
from .online import detect_sequential

__all__ = [
    "ar_psi", "ar_variance", "ar_acf", "simulate_piecewise_ar", "Segment",
    "block_acf", "block_spectrum", "block_mean", "block_logvar",
    "BlockCusum", "MLACS", "ACF1Detector", "SpectralDetector", "MeanDetector",
    "VarianceDetector", "ARResidualDetector", "fit_ar_aic",
    "offline_single_split", "calibrate_threshold",
    "calibrate_threshold_bootstrap", "detect_sequential",
]
__version__ = "1.0.0"
