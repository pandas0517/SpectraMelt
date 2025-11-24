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
    normalize=False
):
    """
    signals: complex np.array, shape (N, L)
    mode:
        "complex"
        "real"              <-- default
        "imag"
        "real_imag"
        "mag"
        "ang"
        "mag_ang"
        "mag_ang_sincos"
    apply_fftshift: if True, shifts before processing
    normalize: if True, normalize by per-signal scale
    returns:
        arr: network input
        scales: (N,) scale factors or None
    """
    if apply_fft:
        freq_signals = fft(signals, axis=1)
    else:
        freq_signals = signals

    if apply_fftshift:
        freq_signals = fftshift(freq_signals, axes=1)

    # ===== COMPLEX MODE =====
    if mode == "complex":
        if normalize:
            scales = np.max(np.abs(freq_signals), axis=1)
            scales[scales == 0] = 1.0
            freq_signals = freq_signals / scales[:, None]
        else:
            scales = None

        return freq_signals.astype(np.complex64), scales

    # ===== REAL MODE =====
    elif mode == "real":
        data = freq_signals.real
        if normalize:
            scales = np.max(np.abs(data), axis=1)
            scales[scales == 0] = 1.0
            data = data / scales[:, None]
        else:
            scales = None

        return data.astype(np.float32), scales

    # ===== IMAG MODE =====
    elif mode == "imag":
        data = freq_signals.imag
        if normalize:
            scales = np.max(np.abs(data), axis=1)
            scales[scales == 0] = 1.0
            data = data / scales[:, None]
        else:
            scales = None

        return data.astype(np.float32), scales

    # ===== REAL + IMAG =====
    elif mode == "real_imag":
        if normalize:
            scales = np.max(np.abs(freq_signals), axis=1)
            scales[scales == 0] = 1.0
            sig_norm = freq_signals / scales[:, None]
        else:
            scales = None
            sig_norm = freq_signals

        arr = np.concatenate([sig_norm.real, sig_norm.imag], axis=1)
        return arr.astype(np.float32), scales

    # ===== MAGNITUDE =====
    elif mode == "mag":
        mag = np.abs(freq_signals) / freq_signals.shape[1]

        if normalize:
            scales = np.max(mag, axis=1)
            scales[scales == 0] = 1.0
            mag = mag / scales[:, None]
        else:
            scales = None

        return mag.astype(np.float32), scales

    # ===== PHASE =====
    elif mode == "ang":
        phase = np.angle(freq_signals)
        return phase.astype(np.float32), None

    # ===== MAG + ANG =====
    elif mode == "mag_ang":
        mag = np.abs(freq_signals) / freq_signals.shape[1]
        phase = np.angle(freq_signals)

        if normalize:
            scales = np.max(mag, axis=1)
            scales[scales == 0] = 1.0
            mag = mag / scales[:, None]
        else:
            scales = None

        arr = np.concatenate([mag, phase], axis=1)
        return arr.astype(np.float32), scales

    # ===== MAG + COS(PHASE) + SIN(PHASE) =====
    elif mode == "mag_ang_sincos":
        mag = np.abs(freq_signals)
        phase = np.angle(freq_signals)

        if normalize:
            scales = np.max(mag, axis=1)
            scales[scales == 0] = 1.0
            mag = mag / scales[:, None]
        else:
            scales = None

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        arr = np.concatenate([mag, cos_phase, sin_phase], axis=1)
        return arr.astype(np.float32), scales

    else:
        raise ValueError(f"Unsupported mode: {mode}")


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