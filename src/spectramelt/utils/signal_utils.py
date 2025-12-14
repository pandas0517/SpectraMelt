import numpy as np
from scipy.fft import (
    fft,
    fftshift,
    ifft,
    ifftshift
)
from scipy.interpolate import interp1d

def fft_encode_signals(
    signals,
    mode="mag_ang_sincos",
    apply_fftshift=False,
    apply_fft=True,
    normalize=True,
    zero_pad=None,
    freq_axis=None
):
    """
    Encode signals for network input with mode-specific constraints,
    optionally zero-padded and resampled to a given frequency axis.
    
    Parameters
    ----------
    signals : np.ndarray
        Complex array, shape (L,) or (N, L)
    mode : str
        "complex", "real", "imag", "real_imag",
        "mag", "ang", "mag_ang", "mag_ang_sincos"
    apply_fftshift : bool
        If True, apply fftshift to frequency domain
    apply_fft : bool
        If True, compute FFT along last axis
    normalize : bool
        If True, divide by original signal length
    zero_pad : int or None
        If provided, FFT length (>= signal length)
    freq_axis : np.ndarray or None
        If provided, resample FFT to match this frequency array
    
    Returns
    -------
    arr : np.ndarray
        Encoded array, resampled to freq_axis if given
    """

    # -------------------------------
    # Mode-specific restrictions
    # -------------------------------
    if apply_fftshift and mode in {"real_imag", "mag_ang", "mag_ang_sincos"}:
        apply_fftshift = False
    if normalize and mode in {"real_imag", "mag_ang", "mag_ang_sincos", "ang"}:
        normalize = False

    # Ensure 2D: (N, L)
    signals = np.atleast_2d(signals)
    N_signals, L = signals.shape

    # Validate zero padding
    fft_len = L
    if zero_pad is not None:
        if not apply_fft:
            raise ValueError("zero_pad requires apply_fft=True")
        if zero_pad < L:
            raise ValueError("zero_pad must be >= signal length")
        fft_len = zero_pad

    # FFT
    freq_signals = fft(signals, n=fft_len, axis=1) if apply_fft else signals

    # Frequency axis corresponding to FFT
    df = 1.0 / fft_len
    fft_freqs = np.fft.fftfreq(fft_len, d=1.0)
    if apply_fftshift:
        freq_signals = fftshift(freq_signals, axes=1)
        fft_freqs = fftshift(fft_freqs)

    # Normalize
    if normalize:
        freq_signals = freq_signals / L

    # Resample to freq_axis if provided
    if freq_axis is not None:
        # Automatically match shift state
        if apply_fftshift:
            freq_axis_interp = freq_axis.copy()  # assume freq_axis is shifted
        else:
            freq_axis_interp = ifftshift(freq_axis)  # make unshifted to match unshifted FFT
        # Interpolate each signal
        arr_resampled = np.empty((N_signals, len(freq_axis)), dtype=freq_signals.dtype)
        for i in range(N_signals):
            interp_func = interp1d(fft_freqs, freq_signals[i], kind='linear', 
                                   bounds_error=False, fill_value=0.0)
            arr_resampled[i] = interp_func(freq_axis_interp)
        freq_signals = arr_resampled

    # -------------------------------
    # Encoding modes
    # -------------------------------
    if mode == "complex":
        arr = freq_signals.astype(np.complex64)
    elif mode == "real":
        arr = freq_signals.real.astype(np.float32)
    elif mode == "imag":
        arr = freq_signals.imag.astype(np.float32)
    elif mode == "real_imag":
        arr = np.concatenate([freq_signals.real, freq_signals.imag], axis=1).astype(np.float32)
    elif mode == "mag":
        arr = np.abs(freq_signals).astype(np.float32)
    elif mode == "ang":
        arr = np.angle(freq_signals).astype(np.float32)
    elif mode == "mag_ang":
        mag = np.abs(freq_signals)
        phase = np.angle(freq_signals)
        arr = np.concatenate([mag, phase], axis=1).astype(np.float32)
    elif mode == "mag_ang_sincos":
        mag = np.abs(freq_signals)
        phase = np.angle(freq_signals)
        arr = np.concatenate([mag, np.cos(phase), np.sin(phase)], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Restore 1D if needed
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