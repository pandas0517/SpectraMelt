import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass
from scipy.fft import fftshift
from .signal_utils import VALID_SAVED_FREQ_MODES

valid_display_freq_modes = VALID_SAVED_FREQ_MODES - {"complex"}

mode_to_key = {m: m for m in valid_display_freq_modes}

REQUIRED_AXIS_KEYS = {"time", "freq"}

@dataclass
class PlotBlock:
    family: str        # "polar", "complex", or "time"
    label: str
    data: np.ndarray
    pos_vals: np.ndarray = None
    neg_vals: np.ndarray = None
    source: str = None


def extract(arr, idx):
    return arr[idx] if arr is not None else None


def overlay_markers(ax, freqs_pos, pos_vals, freqs_neg, neg_vals):
    if pos_vals is not None and len(pos_vals) > 0:
        ax.scatter(freqs_pos, pos_vals, marker='x', color='red', s=100)
        ax.scatter(freqs_neg, neg_vals, marker='x', color='red', s=100)


def expand_freq_modes(freq_arrays, freq_modes, idx, wp=None):
    """
    Returns a list of PlotBlocks for the given signal index.
    Handles real_imag, mag_ang, mag_ang_sincos.
    """
    blocks = []

    if wp is not None:
        amps    = np.array([w["amp"] for w in wp])
        phases  = np.array([w["phase"] for w in wp])
        reals   = np.array([w["real"] for w in wp])
        imags   = np.array([w["imag"] for w in wp])
        neg_phases = -phases
        imag_neg   = -imags
    else:
        amps = phases = reals = imags = neg_phases = imag_neg = np.array([])

    def add(fam, lbl, data, pos=None, neg=None, src=None):
        if data is not None:
            blocks.append(PlotBlock(fam, lbl, data, pos_vals=pos, neg_vals=neg, source=src))

    # Direct modes
    if "real" in freq_modes: add("complex","real", extract(freq_arrays.get("real"), idx), reals, reals, "real")
    if "imag" in freq_modes: add("complex","imag", extract(freq_arrays.get("imag"), idx), imags, imag_neg, "imag")
    if "mag" in freq_modes: add("polar","mag", extract(freq_arrays.get("mag"), idx), amps, amps, "mag")
    if "ang" in freq_modes: add("polar","ang", extract(freq_arrays.get("ang"), idx), phases, neg_phases, "ang")

    # Combined modes
    if "real_imag" in freq_modes and freq_arrays.get("real_imag") is not None:
        arr = extract(freq_arrays["real_imag"], idx)
        if arr is not None:
            add("complex","real", arr[:,0], reals, reals, "real_imag")
            add("complex","imag", arr[:,1], imags, imag_neg, "real_imag")

    if "mag_ang" in freq_modes and freq_arrays.get("mag_ang") is not None:
        arr = extract(freq_arrays["mag_ang"], idx)
        if arr is not None:
            add("polar","mag", arr[:,0], amps, amps, "mag_ang")
            add("polar","ang", arr[:,1], phases, neg_phases, "mag_ang")

    if "mag_ang_sincos" in freq_modes and freq_arrays.get("mag_ang_sincos") is not None:
        arr = extract(freq_arrays["mag_ang_sincos"], idx)
        if arr is not None:
            mag, sin, cos = arr[:,0], arr[:,1], arr[:,2]
            ang = np.arctan2(sin, cos)
            add("polar","mag", mag, amps, amps, "mag_ang_sincos")
            add("polar","ang", ang, phases, -phases, "mag_ang_sincos")

    return blocks


def load_and_prepare_arrays(freq_signals_filename, freq_modes, fft_shift_flag=False, normalize=False, N=None):
    loaded = {}
    with np.load(freq_signals_filename) as freq_signals:
        for mode in freq_modes:
            loaded[mode] = freq_signals[mode] if mode in freq_signals.files else None
        freq_arrays = {mode: loaded.get(mode) for mode in valid_display_freq_modes}

    # Apply fftshift / normalization
    for key, arrays in freq_arrays.items():
        if arrays is not None:
            if fft_shift_flag:
                arrays = fftshift(arrays, axes=-1)
            if normalize and key not in ("ang","real_imag","mag_ang","mag_ang_sincos") and N is not None:
                arrays = arrays / N
            freq_arrays[key] = arrays

    return freq_arrays


def assign_columns(blocks, time_signal=None):
    columns = []
    if time_signal is not None:
        columns.append([PlotBlock("time","Time",time_signal)])

    polar_blocks = [b for b in blocks if b.family=="polar"]
    complex_blocks = [b for b in blocks if b.family=="complex"]

    if polar_blocks:
        columns.append(polar_blocks)
        if complex_blocks:
            columns.append(complex_blocks)
    elif complex_blocks:
        columns.append(complex_blocks)

    return columns


def plot_column(axs, col_blocks, freq=None, time=None, freq_range=None):
    for r, block in enumerate(col_blocks):
        ax = axs[r]
        if block.family=="time":
            ax.plot(time, block.data)
            ax.set_xlim(time.min(), time.max())
        else:
            ax.plot(freq, block.data)
            ax.set_xlim(-freq_range[1], freq_range[1])
            overlay_markers(ax, freq, block.pos_vals, -freq, block.neg_vals)
        ax.set_title(block.label)


def plot_dynamic_frequency_modes(freq_signal_file, time, freq, freq_modes, freq_range,
                                signals_per_file, time_signal_file=None, wave_params_file=None,
                                base_title=None, normalize=False, fft_shift_flag=False):   
    
    wave_params = None
    if wave_params_file is not None and wave_params_file.exists():
        with open(wave_params_file, "rb") as f:
            wave_params = pickle.load(f)

    if not freq_signal_file.exists() or freq_signal_file is None:
        raise ValueError(f"Frequency file {freq_signal_file} does not exist")
    
    time_signals = None
    if time_signal_file is not None and time_signal_file.exists():
        time_signals = np.load(time_signal_file)
    
    N = len(freq)
    freq_arrays = load_and_prepare_arrays(freq_signal_file, freq_modes, fft_shift_flag, normalize, N)

    for idx in range(signals_per_file):
        time_signal = time_signals[idx] if time_signals is not None else None
        wp = wave_params[idx] if wave_params is not None else None
        num_tones = len(wp) if wp is not None else None

        sup = f"{num_tones}-Tone Signals" if num_tones else "Signal"
        if base_title: sup = base_title + sup
        sup += f"\n{freq_signal_file}"

        blocks = expand_freq_modes(freq_arrays, freq_modes, idx, wp=wp)
        columns = assign_columns(blocks, time_signal)

        ncols = len(columns)
        nrows = 1 + max(len(col)-1 for col in columns)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

        for c, col_blocks in enumerate(columns):
            plot_column(axes[:,c], col_blocks, freq=freq, time=time, freq_range=freq_range)

        fig.suptitle(sup)
        fig.tight_layout(rect=[0,0,1,0.95])
        plt.show()