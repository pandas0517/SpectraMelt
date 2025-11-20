from scipy.fft import fft
import numpy as np


def normalize_and_flatten(signals):
    """
    signals: complex np array, shape (N, L)
    """
    scales = np.max(np.abs(signals), axis=1)
    scales[scales == 0] = 1.0

    signals_norm = signals / scales[:, None]

    re = signals_norm.real
    im = signals_norm.imag

    return np.concatenate([re, im], axis=1).astype(np.float32), scales.astype(np.float32)


def unflatten_and_denormalize(pred_flat, scales):
    """
    pred_flat: (N, 2L) model output
    scales: (N,) scale factors
    """
    L = pred_flat.shape[1] // 2
    
    real = pred_flat[:, :L]
    imag = pred_flat[:, L:]
    
    pred_complex = real + 1j * imag
    pred_complex = pred_complex * scales[:, None]
    
    return pred_complex


def normalize_real(signals):
    """
    signals: real np array, shape (N, L)
    returns:
        signals_norm: normalized real signals
        scales: per-signal scale factors
    """
    scales = np.max(np.abs(signals), axis=1)
    scales[scales == 0] = 1.0

    signals_norm = signals / scales[:, None]

    return signals_norm.astype(np.float32), scales.astype(np.float32)


def denormalize_real(pred_norm, scales):
    """
    pred_norm: normalized real signals, shape (N, L)
    scales: (N,) scale factors
    """
    return pred_norm * scales[:, None]


def sparse_fft(signal, threshold_frac=0.05, auto_threshold=False, sparsify=True):
    """
    Compute sparse magnitude and phase of a time-domain signal using time vector.

    Parameters
    ----------
    signal : array_like
        Input time-domain signal (real or complex)
    threshold_frac : float, optional
        Keep bins with magnitude > threshold_frac * max(magnitude)
        Default = 0.05 (5%). Ignored if auto_threshold=True
    auto_threshold : bool, optional
        If True, automatically determine threshold using median + 2*std method
    sparsify : bool, optional
        If False, return unsparsified signals

    Returns
    -------
    magnitude_sparse : ndarray
        Magnitude spectrum with zeroed values below threshold
    phase_sparse : ndarray
        Phase spectrum with zeroed values below threshold
    mask : ndarray
        Boolean mask of where significant tones are kept
    """                      
    if signal is None:
        raise ValueError("Signal can not be none")

    fft_vals = fft(signal)

    # Magnitude + Phase
    magnitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)

    # If no sparsification, return everything untouched
    if not sparsify:
        mask = np.ones_like(magnitude, dtype=bool)
        return magnitude, phase, mask

    # Otherwise determine threshold
    if auto_threshold:
        threshold = np.median(magnitude) + 2*np.std(magnitude)
    else:
        threshold = np.max(magnitude) * threshold_frac

    # Binary mask
    mask = magnitude > threshold

    # Zero out non-significant bins
    magnitude_sparse = magnitude * mask
    phase_sparse = phase * mask

    return magnitude_sparse, phase_sparse, mask