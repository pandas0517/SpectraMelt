import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass
from scipy.fft import fftshift
from .signal_utils import VALID_SAVED_FREQ_MODES

# Displayable frequency modes (exclude abstract "complex")
valid_display_freq_modes = VALID_SAVED_FREQ_MODES - {"complex"}

REQUIRED_AXIS_KEYS = {"time", "freq"}


# =========================
# Data container
# =========================

@dataclass
class PlotBlock:
    family: str              # "polar", "complex", or "time"
    label: str               # "mag", "ang", "real", "imag", "Time"
    decibel: bool = False
    data: np.ndarray
    freqs_pos: np.ndarray = None
    pos_vals: np.ndarray  = None
    freqs_neg: np.ndarray = None
    neg_vals: np.ndarray  = None
    source: str = None       # original freq mode name


# =========================
# Utilities
# =========================

def extract(arr, idx):
    return arr[idx] if arr is not None else None


def overlay_markers(ax, freqs_pos, pos_vals, freqs_neg, neg_vals):
    if pos_vals is not None:
        n = min(len(freqs_pos), len(pos_vals))
        if n:
            ax.scatter(freqs_pos[:n], pos_vals[:n], color='red', marker="x", s=100)

    if neg_vals is not None:
        n = min(len(freqs_neg), len(neg_vals))
        if n:
            ax.scatter(freqs_neg[:n], neg_vals[:n], color='red', marker="x", s=100)


# =========================
# Split helpers
# =========================

def _split_real_imag(arr, fft_shift_flag=False, normalize=False):
    """
    Split 1D or 2D 'real_imag' arrays into real and imag components.
    - 1D: [real..., imag...]
    - 2D: shape (N,2) -> [real, imag]

    Parameters
    ----------
    arr : np.ndarray
        Packed array.
    fft_shift_flag : bool
        If True, apply fftshift along last axis.
    normalize : bool
        If True, divide by N (length of each component).

    Returns
    -------
    real, imag : np.ndarray
    """
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"real_imag length must be even, got {arr.size}")
        N = arr.size // 2
        real = arr[:N]
        imag = arr[N:]
        if fft_shift_flag:
            real = fftshift(real)
            imag = fftshift(imag)
        if normalize:
            real = real / N
            imag = imag / N
        return real, imag

    if arr.ndim == 2 and arr.shape[1] == 2:
        real = arr[:, 0]
        imag = arr[:, 1]
        if fft_shift_flag:
            real = fftshift(real, axes=-1)
            imag = fftshift(imag, axes=-1)
        if normalize:
            N = real.shape[-1]
            real = real / N
            imag = imag / N
        return real, imag

    raise ValueError(f"Unsupported real_imag shape {arr.shape}")


def _split_mag_ang(arr, fft_shift_flag=False, normalize=False):
    """
    Split 1D or 2D 'mag_ang' arrays into magnitude and angle.
    - 1D: [mag..., ang...]
    - 2D: shape (N,2) -> [mag, ang]

    Parameters
    ----------
    arr : np.ndarray
    fft_shift_flag : bool
        Shift only the magnitude.
    normalize : bool
        Normalize only the magnitude (leave angle untouched).

    Returns
    -------
    mag, ang : np.ndarray
    """
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"mag_ang length must be even, got {arr.size}")
        N = arr.size // 2
        mag = arr[:N]
        ang = arr[N:]
        if fft_shift_flag:
            mag = fftshift(mag)
            ang = fftshift(ang)
        if normalize:
            mag = mag / N
        return mag, ang

    if arr.ndim == 2 and arr.shape[1] == 2:
        mag = arr[:, 0]
        ang = arr[:, 1]
        if fft_shift_flag:
            mag = fftshift(mag, axes=-1)
            ang = fftshift(ang, axes=-1)
        if normalize:
            N = mag.shape[-1]
            mag = mag / N
        return mag, ang

    raise ValueError(f"Unsupported mag_ang shape {arr.shape}")


def _split_mag_ang_sincos(arr, fft_shift_flag=False, normalize=False):
    """
    Split 1D or 2D 'mag_ang_sincos' arrays into magnitude and angle.
    - 1D: [mag..., sin..., cos...]
    - 2D: shape (N,3) -> [mag, sin, cos]
    - 2D: shape (N,2) -> [mag, ang]

    Parameters
    ----------
    arr : np.ndarray
    fft_shift_flag : bool
        Shift only the magnitude.
    normalize : bool
        Normalize only the magnitude.

    Returns
    -------
    mag, ang : np.ndarray
    """
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"mag_ang_sincos length must be divisible by 3, got {arr.size}")
        N = arr.size // 3
        mag = arr[:N]
        sin = arr[N:2*N]
        cos = arr[2*N:3*N]
        ang = np.arctan2(sin, cos)
        if fft_shift_flag:
            mag = fftshift(mag)
            sin = fftshift(sin)
            cos = fftshift(cos)
            ang = fftshift(ang)
        if normalize:
            mag = mag / N
        return mag, ang

    if arr.ndim == 2:
        if arr.shape[1] == 3:
            mag = arr[:, 0]
            sin = arr[:, 1]
            cos = arr[:, 2]
            ang = np.arctan2(sin, cos)
            if fft_shift_flag:
                mag = fftshift(mag, axes=-1)
                ang = fftshift(ang, axes=-1)
            if normalize:
                N = mag.shape[-1]
                mag = mag / N
            return mag, ang
        if arr.shape[1] == 2:
            mag = arr[:, 0]
            ang = arr[:, 1]
            if fft_shift_flag:
                mag = fftshift(mag, axes=-1)
                ang = fftshift(ang, axes=-1)
            if normalize:
                N = mag.shape[-1]
                mag = mag / N
            return mag, ang

    raise ValueError(f"Unsupported mag_ang_sincos shape {arr.shape}")


# =========================
# PlotBlock expansion
# =========================

def expand_freq_modes(freq_arrays, freq_modes, idx,
                      fft_shift_flag=False, normalize=False, wp=None, decibels=False):
    def to_db(arr):
        # Voltage dB conversion with zero protection
        return 20.0 * np.log10(np.maximum(np.abs(arr), 1e-12))
    
    blocks = []

    if wp is not None:
        amps       = np.array([w["amp"] for w in wp])
        phases     = np.array([w["phase"] for w in wp])
        freqs      = np.array([w["freq"]  for w in wp])
        reals      = np.array([w["real"] for w in wp])
        imags      = np.array([w["imag"] for w in wp])
        neg_freqs   = -freqs
        neg_phases = -phases
        imag_neg   = -imags
        
        # ---- convert to dB if requested ----
        if decibels:
            amps  = 20 * np.log10(np.maximum(amps, 1e-12))
            reals = 20 * np.log10(np.maximum(np.abs(reals), 1e-12))
            imags = 20 * np.log10(np.maximum(np.abs(imags), 1e-12))
            imag_neg = 20 * np.log10(np.maximum(np.abs(imag_neg), 1e-12))
    else:
        amps = phases = freqs = reals = imags = neg_freqs = neg_phases = imag_neg = np.array([])

    def add(fam, lbl, data, pos_f=None, pos=None,
            neg_f=None, neg=None, src=None, decibel=False):
        if data is not None:
            blocks.append(
                PlotBlock(fam, lbl, data, freqs_pos=pos_f, pos_vals=pos,
                          freqs_neg=neg_f, neg_vals=neg, source=src, decibel=decibel)
            )

    # ---- Direct modes with shift/normalize ----
    if "mag" in freq_modes and freq_arrays.get("mag") is not None:
        arr = extract(freq_arrays["mag"], idx)
        if arr is not None:
            if fft_shift_flag:
                arr = fftshift(arr)
            if normalize:
                arr = arr / arr.size
            if decibels:
                arr = to_db(arr)
            add("polar", "mag", arr, freqs, amps, neg_freqs, amps, "mag", decibels)

    if "ang" in freq_modes and freq_arrays.get("ang") is not None:
        arr = extract(freq_arrays["ang"], idx)
        if arr is not None:
            if fft_shift_flag:
                arr = fftshift(arr)  # angle can be shifted if desired
            # NOTE: do NOT normalize angles
            add("polar", "ang", arr, freqs, phases, neg_freqs, neg_phases, "ang")

    if "real" in freq_modes and freq_arrays.get("real") is not None:
        arr = extract(freq_arrays["real"], idx)
        if arr is not None:
            if fft_shift_flag:
                arr = fftshift(arr)
            if normalize:
                arr = arr / arr.size
            if decibels:
                arr = to_db(arr)
            add("complex", "real", arr, freqs, reals, neg_freqs, reals, "real", decibels)


    if "imag" in freq_modes and freq_arrays.get("imag") is not None:
        arr = extract(freq_arrays["imag"], idx)
        if arr is not None:
            if fft_shift_flag:
                arr = fftshift(arr)
            if normalize:
                arr = arr / arr.size
            if decibels:
                arr = to_db(arr)
            add("complex", "imag", arr, freqs, imags, neg_freqs, imag_neg, "imag", decibels)


    # ---- real_imag ----
    if "real_imag" in freq_modes and freq_arrays.get("real_imag") is not None:
        arr = extract(freq_arrays["real_imag"], idx)
        if arr is not None:
            real, imag = _split_real_imag(arr, fft_shift_flag=fft_shift_flag, normalize=normalize)
            if decibels:
                real = to_db(real)
                imag = to_db(imag)
            add("complex", "real", real, freqs, reals, neg_freqs, reals, "real_imag", decibels)
            add("complex", "imag", imag, freqs, imags, neg_freqs, imag_neg, "real_imag", decibels)

    # ---- mag_ang ----
    if "mag_ang" in freq_modes and freq_arrays.get("mag_ang") is not None:
        arr = extract(freq_arrays["mag_ang"], idx)
        if arr is not None:
            mag, ang = _split_mag_ang(arr, fft_shift_flag=fft_shift_flag, normalize=normalize)
            if decibels:
                mag = to_db(mag)
            add("polar", "mag", mag, freqs, amps, neg_freqs, amps, "mag_ang", decibels)
            add("polar", "ang", ang, freqs, phases, neg_freqs, neg_phases, "mag_ang")

    # ---- mag_ang_sincos ----
    if "mag_ang_sincos" in freq_modes and freq_arrays.get("mag_ang_sincos") is not None:
        arr = extract(freq_arrays["mag_ang_sincos"], idx)
        if arr is not None:
            mag, ang = _split_mag_ang_sincos(arr, fft_shift_flag=fft_shift_flag, normalize=normalize)
            if decibels:
                mag = to_db(mag)
            add("polar", "mag", mag, freqs, amps, neg_freqs, amps, "mag_ang_sincos", decibels)
            add("polar", "ang", ang, freqs, phases, neg_freqs, neg_phases, "mag_ang_sincos")

    return blocks


# =========================
# Loading
# =========================

def load_and_prepare_arrays(freq_file):
    freq_arrays = {}

    with np.load(freq_file) as f:
        files = set(f.files)

        # Polar precedence
        use_mag_ang = "mag_ang" in files
        use_mag_ang_sincos = "mag_ang_sincos" in files and not use_mag_ang

        for m in valid_display_freq_modes:
            if m == "mag_ang_sincos" and not use_mag_ang_sincos:
                freq_arrays[m] = None
                continue
            if m == "mag_ang" and not use_mag_ang:
                freq_arrays[m] = None
                continue

            freq_arrays[m] = f[m] if m in files else None

    return freq_arrays


# =========================
# Plot layout
# =========================

def assign_columns(blocks, time_signal=None):
    columns = []

    if time_signal is not None:
        columns.append([PlotBlock("time", "Time", time_signal)])

    polar   = [b for b in blocks if b.family == "polar"]
    complex = [b for b in blocks if b.family == "complex"]

    if polar:
        columns.append(polar)
        if complex:
            columns.append(complex)
    elif complex:
        columns.append(complex)

    return columns


def plot_column(axs, col_blocks, freq=None, time=None, freq_range=None):
    for r, block in enumerate(col_blocks):
        ax = axs[r]

        if block.family == "time":
            ax.plot(time, block.data)
            ax.set_xlim(time.min(), time.max())
        else:
            ax.plot(freq, block.data)
            ax.set_xlim(-freq_range[1], freq_range[1])
            overlay_markers(ax, block.freqs_pos, block.pos_vals, block.freqs_neg, block.neg_vals)
        
        if block.decibel:
            ax.set_ylabel("Magnitude (dB)")

        ax.set_title(block.label)


# =========================
# Public API
# =========================

def plot_dynamic_frequency_modes(
    freq_signal_file,
    time,
    freq,
    freq_modes,
    freq_range,
    signals_per_file,
    time_signal_file=None,
    wave_params_file=None,
    base_title=None,
    normalize=False,
    fft_shift_flag=False,
    decibels=False
):

    wave_params = None
    if wave_params_file and wave_params_file.exists():
        with open(wave_params_file, "rb") as f:
            wave_params = pickle.load(f)

    time_signals = None
    if time_signal_file and time_signal_file.exists():
        time_signals = np.load(time_signal_file)

    freq_arrays = load_and_prepare_arrays(freq_signal_file, decibels)

    for idx in range(signals_per_file):
        wp = wave_params[idx] if wave_params is not None else None
        time_signal = time_signals[idx] if time_signals is not None else None

        blocks = expand_freq_modes(freq_arrays, freq_modes, idx,
                                   fft_shift_flag, normalize, wp, decibels)
        columns = assign_columns(blocks, time_signal)

        ncols = len(columns)
        nrows = max(len(c) for c in columns)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
        )

        for c, col in enumerate(columns):
            plot_column(axes[:, c], col, freq=freq, time=time, freq_range=freq_range)

        if base_title:
            fig.suptitle(base_title)

        plt.tight_layout()
        plt.show()