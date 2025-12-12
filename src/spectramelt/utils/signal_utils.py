import numpy as np
from scipy.fft import (
    fft,
    fftshift,
    ifft,
    ifftshift
)


def fft_encode_signals(
    signals,
    mode="mag_ang_sincos",
    apply_fftshift=False,
    apply_fft=True,
    normalize=True
):
    """
    Encode signals for network input with mode-specific constraints.
    
    Parameters
    ----------
    signals : np.ndarray
        Complex array, shape (L,) or (N, L)
    mode : str
        "complex", "real", "imag", "real_imag",
        "mag", "ang", "mag_ang", "mag_ang_sincos"
    apply_fftshift : bool
        If True, apply fftshift before processing (restricted by mode)
    apply_fft : bool
        If True, apply FFT along last axis
    normalize : bool
        If True, divide each signal by its length (restricted by mode)

    Returns
    -------
    arr : np.ndarray
        Encoded array
    """
    # Enforce mode-specific restrictions
    if apply_fftshift:
        if mode in ["real_imag",
                    "mag_ang",
                    "mag_ang_sincos"]:
            apply_fftshift = False
    if normalize:
        if mode in ["real_imag",
                    "mag_ang",
                    "mag_ang_sincos",
                    "ang"]:
            normalize = False

    # Ensure signals are at least 2D (N, L)
    signals = np.atleast_2d(signals)
    N, L = signals.shape

    # FFT
    freq_signals = fft(signals, axis=1) if apply_fft else signals

    # FFT shift
    if apply_fftshift:
        freq_signals = fftshift(freq_signals, axes=1)

    # Optional normalization by signal length
    if normalize:
        freq_signals = freq_signals / L

    # ===== COMPLEX MODE =====
    if mode == "complex":
        arr = freq_signals.astype(np.complex64)

    # ===== REAL MODE =====
    elif mode == "real":
        arr = freq_signals.real.astype(np.float32)

    # ===== IMAG MODE =====
    elif mode == "imag":
        arr = freq_signals.imag.astype(np.float32)

    # ===== REAL + IMAG =====
    elif mode == "real_imag":
        arr = np.concatenate([freq_signals.real, freq_signals.imag], axis=1).astype(np.float32)

    # ===== MAGNITUDE =====
    elif mode == "mag":
        arr = np.abs(freq_signals).astype(np.float32)

    # ===== PHASE =====
    elif mode == "ang":
        arr = np.angle(freq_signals).astype(np.float32)

    # ===== MAG + PHASE =====
    elif mode == "mag_ang":
        mag = np.abs(freq_signals)
        phase = np.angle(freq_signals)
        arr = np.concatenate([mag, phase], axis=1).astype(np.float32)

    # ===== MAG + COS(PHASE) + SIN(PHASE) =====
    elif mode == "mag_ang_sincos":
        mag = np.abs(freq_signals)
        phase = np.angle(freq_signals)
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        arr = np.concatenate([mag, cos_phase, sin_phase], axis=1).astype(np.float32)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # If input was 1D, return a 1D array
    if signals.shape[0] == 1:
        arr = arr[0]

    return arr


def fft_decode_signals(
    encoded,
    scales=None,
    mode="mag_ang_sincos",
    apply_ifftshift=False,
    to_time_domain=False
):
    """
    encoded: np.array, shape (N, ?), output from the network
    scales: optional scaling factors returned by the encoder
    mode:
        "complex"
        "real_imag"
        "mag_ang"
        "mag_ang_sincos"
    apply_ifftshift: undo fftshift if you applied it before encoding
    to_time_domain: if True, return inverse FFT back to time domain

    returns:
        complex freq_signals OR complex time_signals
    """

    N = encoded.shape[0]

    # ===== COMPLEX MODE =====
    if mode == "complex":
        freq_signals = encoded.astype(np.complex64)

        if scales is not None:
            freq_signals *= scales[:, None]

    # ===== REAL + IMAG MODE =====
    elif mode == "real_imag":
        L = encoded.shape[1] // 2

        real = encoded[:, :L]
        imag = encoded[:, L:]
        freq_signals = real + 1j * imag

        if scales is not None:
            freq_signals *= scales[:, None]

    # ===== MAG + ANG MODE =====
    elif mode == "mag_ang":
        L = encoded.shape[1] // 2

        mag = encoded[:, :L]
        phase = encoded[:, L:]

        freq_signals = mag * np.exp(1j * phase)

        if scales is not None:
            freq_signals *= scales[:, None]

    # ===== MAG + COS(PHASE) + SIN(PHASE) =====
    elif mode == "mag_ang_sincos":
        L = encoded.shape[1] // 3

        mag = encoded[:, :L]
        cos_phase = encoded[:, L:2 * L]
        sin_phase = encoded[:, 2 * L:]

        # Recover phase
        phase = np.arctan2(sin_phase, cos_phase)

        freq_signals = mag * np.exp(1j * phase)

        if scales is not None:
            freq_signals *= scales[:, None]

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # ===== UNDO FFT SHIFT =====
    if apply_ifftshift:
        freq_signals = ifftshift(freq_signals, axes=1)

    # ===== OPTIONAL IFFT =====
    if to_time_domain:
        return ifft(freq_signals, axis=1)

    return freq_signals


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