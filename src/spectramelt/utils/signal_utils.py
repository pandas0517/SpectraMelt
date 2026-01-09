import numpy as np
from scipy.fft import (
    fft,
    fftshift,
    ifft,
    ifftshift
)
from scipy.interpolate import interp1d
import re
import pickle
from pathlib import Path

VALID_SAVED_FREQ_MODES = {
    "complex", "real", "imag",
    "real_imag", "mag", "ang",
    "mag_ang", "mag_ang_sincos"
}


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


def compute_recovery_stats(rec_mag_final, spur_mag_final, amps_combined):
    """Compute sizes, ave/max/min, signed deviation."""
    if rec_mag_final.size == 0:
        return (-1,) * 10
    rec_size, spur_size = len(rec_mag_final), len(spur_mag_final)
    ave_rec_err = np.mean(amps_combined) - np.mean(rec_mag_final)
    rec_abs, spur_abs = np.abs(rec_mag_final), np.abs(spur_mag_final)
    return (rec_size, spur_size, ave_rec_err,
            np.mean(rec_abs), np.max(rec_abs), np.min(rec_abs),
            np.mean(spur_abs), np.max(spur_abs), np.min(spur_abs))


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
        'total_input_tones': {
            'col_name': "total_input_tones_" + str(idx),
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


def process_signal_file(
    recovery_file: Path,
    wbf_wave_file: Path,
    input_file: Path,
    freq_modes: list[str],
    recovery_mag_threshold: float,
    num_recovery_sigs: int,
    inputset_config_name: str,
    DUT_config_name: str,
    recovery_config_name: str,
    wbf_freq: np.ndarray
) -> list[dict]:

    # --- Load recovery ---
    with np.load(recovery_file) as recovery_npz:
        available = set(recovery_npz.files)
        valid = [m for m in freq_modes if m in available]
        missing = set(freq_modes) - set(valid)
        if missing:
            print(f"Warning: Missing recovery modes in {recovery_file}: {missing}")
        recovery = {m: recovery_npz[m] for m in valid}

    # Handle real_imag
    if "real_imag" in recovery:
        arr = recovery["real_imag"]
        recovery["real_imag"] = {k: arr[..., i] for i, k in enumerate(["real", "imag"])}

    flat_recovery = flatten_dict(recovery)

    # Sanity check
    offending = [k for k, arr in flat_recovery.items() if arr.shape[0] != num_recovery_sigs]
    if offending:
        raise ValueError(f"Recovered signal size mismatch: {offending}")

    # Load WBF waves
    with open(wbf_wave_file, "rb") as f:
        wbf_waves = pickle.load(f)

    rows = []

    for idx_sig, wbf_wave in enumerate(wbf_waves):
        # Build combined arrays
        amps  = np.array([w["amp"] for w in wbf_wave])
        freqs = np.array([w["freq"] for w in wbf_wave])
        reals = np.array([w["real"] for w in wbf_wave])
        imags = np.array([w["imag"] for w in wbf_wave])

        pos_idx = np.array([np.argmin(np.abs(wbf_freq - f)) for f in freqs])
        neg_idx = np.array([np.argmin(np.abs(wbf_freq + f)) for f in freqs])
        wbf_unsorted = np.concatenate([neg_idx, pos_idx])

        freq_combined  = np.concatenate([-freqs, freqs])
        reals_combined = np.concatenate([reals, reals])
        imags_combined = np.concatenate([-imags, imags])
        amps_combined  = np.concatenate([amps, amps])

        meta_data = create_meta_data_dictionary(idx_sig)

        for mode in freq_modes:
            new_row = {
                "input_file_name": input_file,
                "wbf_file_name": wbf_wave_file,
                "recovery_file_name": recovery_file,
                "input_config_name": inputset_config_name,
                "DUT_config_name": DUT_config_name,
                "recovery_config_name": recovery_config_name,
                "Frequency_mode": mode,
            }

            # --- Determine recovered tones ---
            if mode == "mag":
                rec_mag = flat_recovery["mag"][idx_sig]
                mask = np.abs(rec_mag[wbf_unsorted]) > recovery_mag_threshold
                rec_sig_tones = wbf_unsorted[mask]
                rec_mag_final  = rec_mag[rec_sig_tones]
                spur_mag_final = rec_mag[np.setdiff1d(wbf_unsorted, rec_sig_tones)]
            elif mode == "real_imag":
                real_arr = flat_recovery["real_imag.real"][idx_sig]
                imag_arr = flat_recovery["real_imag.imag"][idx_sig]
                real_vals = real_arr[wbf_unsorted]
                imag_vals = imag_arr[wbf_unsorted]
                mask = (np.abs(real_vals) > recovery_mag_threshold) | (np.abs(imag_vals) > recovery_mag_threshold)
                rec_sig_tones = wbf_unsorted[mask]
                rec_mag_final  = np.concatenate([real_arr[rec_sig_tones], imag_arr[rec_sig_tones]])
                spur_mag_final = np.concatenate([real_arr[np.setdiff1d(wbf_unsorted, rec_sig_tones)],
                                                 imag_arr[np.setdiff1d(wbf_unsorted, rec_sig_tones)]])
            else:
                raise ValueError(f"Unsupported frequency mode: {mode}")

            # --- Compute stats ---
            rec_size, spur_size, ave_rec_err, ave_rec, max_rec, min_rec, ave_spur, max_spur, min_spur = \
                compute_recovery_stats(rec_mag_final, spur_mag_final, amps_combined)

            # --- Update meta_data ---
            meta_data.update({
                k: {'col_name': v['col_name'], 'value': val}
                for k, v, val in [
                    ('num_rec_freq', meta_data['num_rec_freq'], rec_size),
                    ('num_spur_freq', meta_data['num_spur_freq'], spur_size),
                    ('total_input_tones', meta_data['total_input_tones'], len(wbf_wave)),
                    ('rec_tone_thresh', meta_data['rec_tone_thresh'], recovery_mag_threshold),
                    ('ave_rec_mag_err', meta_data['ave_rec_mag_err'], ave_rec_err),
                    ('ave_rec_mag', meta_data['ave_rec_mag'], ave_rec),
                    ('max_rec_mag', meta_data['max_rec_mag'], max_rec),
                    ('min_rec_mag', meta_data['min_rec_mag'], min_rec),
                    ('ave_spur_mag', meta_data['ave_spur_mag'], ave_spur),
                    ('max_spur_mag', meta_data['max_spur_mag'], max_spur),
                    ('min_spur_mag', meta_data['min_spur_mag'], min_spur)
                ]
            })

            new_row.update({v['col_name']: v['value'] for v in meta_data.values()})
            rows.append(new_row)

    return rows