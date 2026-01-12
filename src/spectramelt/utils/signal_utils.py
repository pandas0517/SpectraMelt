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


def process_signal_file(
    recovery_file: Path,
    wbf_wave_file: Path,
    input_file: Path,
    freq_modes: list[str],
    recovery_mag_threshold: float,
    num_recovery_sigs: int,
    dataset_config_name: str,
    inputset_config_name: str,
    DUT_config_name: str,
    recovery_config_name: str,
    wbf_freq: np.ndarray
) -> list[dict]:

    # ---------- Load recovery ----------
    with np.load(recovery_file) as recovery_npz:
        available = set(recovery_npz.files)
        valid = [m for m in freq_modes if m in available]
        missing = set(freq_modes) - set(valid)
        if missing:
            print(f"Warning: Missing recovery modes in {recovery_file}: {missing}")
        recovery = {m: recovery_npz[m] for m in valid}

    # Split real_imag if present
    if "real_imag" in recovery:
        arr = recovery["real_imag"]
        real, imag = np.array_split(arr, 2, axis=1)
        recovery["real_imag"] = {"real": real, "imag": imag}

    # FFT shift
    for mode, data in recovery.items():
        if isinstance(data, dict):
            for k in data:
                data[k] = np.fft.fftshift(data[k], axes=-1)
        else:
            recovery[mode] = np.fft.fftshift(data, axes=-1)

    flat_recovery = flatten_dict(recovery)

    # ---------- Sanity check ----------
    for k, arr in flat_recovery.items():
        if arr.shape[0] != num_recovery_sigs:
            raise ValueError(f"Recovered signal size mismatch: {k}")

    # ---------- Load WBF waves ----------
    with open(wbf_wave_file, "rb") as f:
        wbf_waves = pickle.load(f)

    rows = []

    # ================================================================
    # ONE ROW PER FREQUENCY MODE
    # ================================================================
    for mode in freq_modes:

        row = {
            "input_file_name": input_file,
            "wbf_file_name": wbf_wave_file,
            "recovery_file_name": recovery_file,
            "dataset_config_name": dataset_config_name,
            "input_config_name": inputset_config_name,
            "DUT_config_name": DUT_config_name,
            "recovery_config_name": recovery_config_name,
            "Frequency_mode": mode,
        }

        # ============================================================
        # PER-SIGNAL STATS
        # ============================================================
        for idx_sig, wbf_wave in enumerate(wbf_waves):

            if idx_sig == 0:
                row["total_input_tones"] = len(wbf_wave)
                row["rec_tone_thresh"] = recovery_mag_threshold

            # ---- reference data ----
            amps  = np.array([w["amp"]  for w in wbf_wave])
            freqs = np.array([w["freq"] for w in wbf_wave])
            reals = np.array([w["real"] for w in wbf_wave])
            imags = np.array([w["imag"] for w in wbf_wave])

            pos_idx = np.array([np.argmin(np.abs(wbf_freq - f)) for f in freqs])
            neg_idx = np.array([np.argmin(np.abs(wbf_freq + f)) for f in freqs])
            rec_bins = np.concatenate([neg_idx, pos_idx])

            all_bins = np.arange(wbf_freq.size)
            non_rec_bins = np.setdiff1d(all_bins, rec_bins)

            # ========================================================
            # MAG MODE
            # ========================================================
            if mode == "mag":
                mag = flat_recovery["mag"][idx_sig]
                mag_abs = np.abs(mag)

                rec_vals = mag_abs[rec_bins]
                rec_final = rec_vals[rec_vals > recovery_mag_threshold]

                spur_vals = mag_abs[non_rec_bins]
                spur_final = spur_vals[spur_vals > recovery_mag_threshold]

                ref_vals = np.concatenate([amps, amps])

            # ========================================================
            # REAL / IMAG MODE
            # ========================================================
            elif mode == "real_imag":
                real = flat_recovery["real_imag.real"][idx_sig]
                imag = flat_recovery["real_imag.imag"][idx_sig]

                real_abs = np.abs(real)
                imag_abs = np.abs(imag)

                # recovered (expected bins)
                rec_real = real_abs[rec_bins]
                rec_imag = imag_abs[rec_bins]

                rec_final = np.concatenate([
                    rec_real[rec_real > recovery_mag_threshold],
                    rec_imag[rec_imag > recovery_mag_threshold],
                ])

                # spurs (unexpected bins)
                spur_real = real_abs[non_rec_bins]
                spur_imag = imag_abs[non_rec_bins]

                spur_final = np.concatenate([
                    spur_real[spur_real > recovery_mag_threshold],
                    spur_imag[spur_imag > recovery_mag_threshold],
                ])

                ref_vals = np.concatenate([np.abs(reals), np.abs(imags),
                                           np.abs(reals), np.abs(imags)])

            else:
                raise ValueError(f"Unsupported frequency mode: {mode}")

            # ---- compute stats ----
            stats = compute_recovery_stats(
                rec_final,
                spur_final,
                ref_vals,
                min_threshold=recovery_mag_threshold
            )

            # ---- write per-signal columns ----
            meta = create_meta_data_dictionary(idx_sig)
            (
                meta["num_rec_freq"]["value"],
                meta["num_spur_freq"]["value"],
                meta["ave_rec_mag_err"]["value"],
                meta["ave_rec_mag"]["value"],
                meta["max_rec_mag"]["value"],
                meta["min_rec_mag"]["value"],
                meta["ave_spur_mag"]["value"],
                meta["max_spur_mag"]["value"],
                meta["min_spur_mag"]["value"],
            ) = stats

            row.update({v["col_name"]: v["value"] for v in meta.values()})

        # ============================================================
        # AGGREGATES
        # ============================================================
        num_rec_vals  = [row[f"num_rec_freq_{i}"] for i in range(num_recovery_sigs)]
        num_spur_vals = [row[f"num_spur_freq_{i}"] for i in range(num_recovery_sigs)]

        row["ave_num_rec"]  = safe_mean(num_rec_vals)
        row["ave_num_spur"] = safe_mean(num_spur_vals)

        denom = 2 if mode == "mag" else 4
        row["recovery_rate"] = (
            row["ave_num_rec"] / (denom * row["total_input_tones"])
            if row["ave_num_rec"] != -1 else -1
        )

        row["ave_rec_mag_err"] = safe_mean([row[f"ave_rec_mag_err_{i}"] for i in range(num_recovery_sigs)])
        row["ave_rec_mag"]     = safe_mean([row[f"ave_rec_mag_{i}"] for i in range(num_recovery_sigs)])
        row["max_rec_mag"]     = safe_max([row[f"max_rec_mag_{i}"] for i in range(num_recovery_sigs)])
        row["min_rec_mag"]     = safe_min([row[f"min_rec_mag_{i}"] for i in range(num_recovery_sigs)])
        row["ave_spur_mag"]    = safe_mean([row[f"ave_spur_mag_{i}"] for i in range(num_recovery_sigs)])
        row["max_spur_mag"]    = safe_max([row[f"max_spur_mag_{i}"] for i in range(num_recovery_sigs)])
        row["min_spur_mag"]    = safe_min([row[f"min_spur_mag_{i}"] for i in range(num_recovery_sigs)])

        rows.append(row)

    return rows