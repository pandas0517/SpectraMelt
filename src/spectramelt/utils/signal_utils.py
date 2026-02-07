import numpy as np
from scipy.fft import (
    fft,
    fftshift,
    ifft,
    ifftshift
)
from scipy.interpolate import interp1d
import re


VALID_SAVED_FREQ_MODES = {
    "complex", "real", "imag",
    "real_imag", "mag", "ang",
    "mag_ang", "mag_ang_sincos"
}


def enforce_hermitian(X: np.ndarray) -> np.ndarray:
    """
    Enforce Hermitian symmetry so that IFFT yields a real-valued signal.
    Assumes standard FFT ordering:
        X[0]        -> DC
        X[1:N//2]   -> positive frequencies
        X[N//2]     -> Nyquist (if N even)
        X[N//2+1:]  -> negative frequencies
    """
    X = X.copy()
    N = len(X)

    if N < 2:
        return X

    # Positive frequencies (exclude DC and Nyquist)
    pos = slice(1, N // 2)
    neg = slice(N // 2 + 1, None)

    X[neg] = np.conj(X[pos][::-1])

    # Enforce real DC and Nyquist
    X[0] = np.real(X[0])
    if N % 2 == 0:
        X[N // 2] = np.real(X[N // 2])

    return X


def snr_db(x_ref, x_rec, eps=1e-12):
    signal_power = np.mean(x_ref**2)
    noise_power  = np.mean((x_ref - x_rec)**2)
    return 10 * np.log10((signal_power + eps) / (noise_power + eps))


def enob_from_snr(snr_db):
    return (snr_db - 1.76) / 6.02


def filter_valid_names(names, valid_set=None):
    if valid_set is None:
        valid_set = VALID_SAVED_FREQ_MODES

    # Allow a single string or a list
    if isinstance(names, str):
        names = [names]

    valid = []
    removed = []

    for n in names:
        if n.lower() in valid_set:
            valid.append(n)
        else:
            removed.append(n)

    return valid, removed


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
    """
    # -------------------------------
    # Mode-specific restrictions
    # -------------------------------
    if apply_fftshift and mode in {"real_imag", "mag_ang", "mag_ang_sincos"}:
        apply_fftshift = False

    # -------------------------------------------------
    # Ensure 2D: (N_signals, L)
    # -------------------------------------------------
    signals = np.atleast_2d(signals)
    N_signals, L = signals.shape

    # -------------------------------------------------
    # Zero padding
    # -------------------------------------------------
    fft_len = L
    if zero_pad is not None:
        if not apply_fft:
            raise ValueError("zero_pad requires apply_fft=True")
        if zero_pad < L:
            raise ValueError("zero_pad must be >= signal length")
        fft_len = zero_pad

    # -------------------------------------------------
    # FFT
    # -------------------------------------------------
    freq_signals = fft(signals, n=fft_len, axis=1) if apply_fft else signals

    fft_freqs = np.fft.fftfreq(fft_len, d=1.0)
    if apply_fftshift:
        freq_signals = fftshift(freq_signals, axes=1)
        fft_freqs = fftshift(fft_freqs)

    # -------------------------------------------------
    # Resample to freq_axis if provided
    # -------------------------------------------------
    if freq_axis is not None:
        if apply_fftshift:
            freq_axis_interp = freq_axis
        else:
            freq_axis_interp = ifftshift(freq_axis)

        resampled = np.empty((N_signals, len(freq_axis)), dtype=freq_signals.dtype)
        for i in range(N_signals):
            interp = interp1d(
                fft_freqs,
                freq_signals[i],
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )
            resampled[i] = interp(freq_axis_interp)

        freq_signals = resampled
        fft_len = len(freq_axis)

    # -------------------------------------------------
    # Encoding modes (NORMALIZATION APPLIED LOCALLY)
    # -------------------------------------------------
    if mode == "complex":
        arr = freq_signals.astype(np.complex64)
        if normalize:
            arr /= L

    elif mode == "real":
        arr = freq_signals.real.astype(np.float32)
        if normalize:
            arr /= L

    elif mode == "imag":
        arr = freq_signals.imag.astype(np.float32)
        if normalize:
            arr /= L

    elif mode == "real_imag":
        real = freq_signals.real
        imag = freq_signals.imag
        if normalize:
            real = real / L
            imag = imag / L
        arr = np.concatenate([real, imag], axis=1).astype(np.float32)

    elif mode == "mag":
        mag = np.abs(freq_signals)
        if normalize:
            mag = mag / L
        arr = mag.astype(np.float32)

    elif mode == "ang":
        # NEVER normalize phase
        arr = np.angle(freq_signals).astype(np.float32)

    elif mode == "mag_ang":
        mag = np.abs(freq_signals)
        if normalize:
            mag = mag / L
        phase = np.angle(freq_signals)
        arr = np.concatenate([mag, phase], axis=1).astype(np.float32)

    elif mode == "mag_ang_sincos":
        mag = np.abs(freq_signals)
        if normalize:
            mag = mag / L
        phase = np.angle(freq_signals)
        arr = np.concatenate(
            [mag, np.sin(phase), np.cos(phase)],
            axis=1
        ).astype(np.float32)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # -------------------------------------------------
    # Restore 1D if single signal
    # -------------------------------------------------
    if arr.shape[0] == 1:
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


def numeric_key(s: str) -> int:
    m = re.search(r'\d+', s)
    return int(m.group()) if m else float('inf')


def get_prefix_before_recovery(filename: str) -> str:
    idx = filename.lower().find("recovery")
    return filename[:idx] if idx != -1 else filename


def compute_recovery_stats(rec_vals, spur_vals, ref_vals, min_threshold=-1):
    """
    Compute recovered and spur statistics using a reference array for deviation.
    Always uses absolute values for rec/spur stats. Enforces min_threshold if provided.

    Parameters
    ----------
    rec_vals : np.ndarray
        Recovered values (magnitudes or real/imag concatenated).
    spur_vals : np.ndarray
        Spurious values (magnitudes or real/imag concatenated).
    ref_vals : np.ndarray
        Reference array to compute average deviation (same length as rec_vals for recovered tones).
    min_threshold : float
        Minimum allowed value for min_rec and min_spur.

    Returns
    -------
    tuple
        rec_size, spur_size, ave_rec_err, ave_rec, max_rec, min_rec,
        ave_spur, max_spur, min_spur
    """
    rec_vals = np.abs(rec_vals)
    spur_vals = np.abs(spur_vals)
    ref_vals = np.abs(ref_vals)

    rec_size = len(rec_vals)
    spur_size = len(spur_vals)

    # --- Recovered stats ---
    if rec_size > 0:
        ave_rec_err = np.mean(ref_vals) - np.mean(rec_vals)
        ave_rec = np.mean(rec_vals)
        max_rec = np.max(rec_vals)
        min_rec = np.min(rec_vals)
        if min_threshold > 0:
            min_rec = max(min_rec, min_threshold)
    else:
        ave_rec_err = ave_rec = max_rec = min_rec = -1

    # --- Spur stats ---
    if spur_size > 0:
        ave_spur = np.mean(spur_vals)
        max_spur = np.max(spur_vals)
        min_spur = np.min(spur_vals)
        if min_threshold > 0:
            min_spur = max(min_spur, min_threshold)
    else:
        ave_spur = max_spur = min_spur = -1

    return (
        rec_size, spur_size, abs(ave_rec_err),
        ave_rec, max_rec, min_rec,
        ave_spur, max_spur, min_spur
    )


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flattens a nested dictionary.

    Args:
        d: Nested dictionary to flatten.
        parent_key: Used internally to build full keys during recursion.
        sep: Separator between nested keys.

    Returns:
        A flat dictionary where keys are the full path in the nested dict.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, parent_key=new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def create_meta_data_dictionary(idx):
    meta_data = {
        'rms_util': {
            'col_name': "rms_util_" + str(idx),
            'value': 0            
        },
        'snr_db': {
            'col_name': "snr_db_" + str(idx),
            'value': 0            
        },
        'enob': {
            'col_name': "enob_" + str(idx),
            'value': 0            
        },
        'num_rec_freq': {
            'col_name': "num_rec_freq_" + str(idx),
            'value': 0
        },
        'num_spur_freq': {
            'col_name': "num_spur_freq_" + str(idx),
            'value': 0
        },
        'ave_rec_mag_err': {
            'col_name': "ave_rec_mag_err_" + str(idx),
            'value': 0
        },
        'rec_tone_thresh': {
            'col_name': "rec_tone_thresh_" + str(idx),
            'value': 0
        },
        'ave_rec_mag': {
            'col_name': "ave_rec_mag_" + str(idx),
            'value': 0
        },
        'max_rec_mag': {
            'col_name': "max_rec_mag_" + str(idx),
            'value': 0
        },
        'min_rec_mag': {
            'col_name': "min_rec_mag_" + str(idx),
            'value': 0
        },
        'ave_spur_mag': {
            'col_name': "ave_spur_mag_" + str(idx),
            'value': 0
        },
        'max_spur_mag': {
            'col_name': "max_spur_mag_" + str(idx),
            'value': 0
        },
        'min_spur_mag': {
            'col_name': "min_spur_mag_" + str(idx),
            'value': 0
        }
    }
    return meta_data


def filter_valid(vals):
    """Return numpy array with -1 values removed."""
    arr = np.asarray(vals, dtype=float)
    return arr[arr != -1]


def safe_mean(vals):
    vals = np.array(vals, dtype=float)
    vals = vals[vals != -1]
    return vals.mean() if vals.size > 0 else -1


def safe_max(vals):
    vals = np.array(vals, dtype=float)
    vals = vals[vals != -1]
    return vals.max() if vals.size > 0 else -1


def safe_min(vals):
    vals = np.array(vals, dtype=float)
    vals = vals[vals != -1]
    return vals.min() if vals.size > 0 else -1