import matplotlib.pyplot as plt
from scipy.fft import fftshift

# Map modes to keys inside .npz
mode_to_key = {
    "mag": "mag",
    "ang": "ang",
    "real": "real",
    "imag": "imag",
}


valid_display_freq_modes = {"real", "imag", "mag", "ang"}


# Helper: Map plot title → mode name
def mode_from_title(title):
    return {
        "Magnitude": "mag",
        "Phase":     "ang",
        "Real":      "real",
        "Imaginary": "imag",
    }[title]


# Helper: safe per-signal extraction
def extract(arr, idx):
    return arr[idx] if arr is not None else None


def build_columns(
    plot_info,
    time_signal=None
):
    """
    Build column definitions for plotting.

    Parameters
    ----------
    plot_info : dict
        Keys: "mag", "ang", "real", "imag"
        Each value: (data, pos_vals, neg_vals)
    time_signal : np.ndarray or None
        Optional time-domain signal

    Returns
    -------
    columns : list of dict
    """

    mag_exists  = plot_info["mag"][0]  is not None
    ang_exists  = plot_info["ang"][0]  is not None
    real_exists = plot_info["real"][0] is not None
    imag_exists = plot_info["imag"][0] is not None

    columns = []

    # -----------------------------
    # Optional Time Column
    # -----------------------------
    if time_signal is not None:
        columns.append({
            "title_top": "Time (Signal)",
            "data_top": time_signal,
            "time_column": True,
            "bottom": None
        })

    # -----------------------------
    # Frequency-domain columns
    # -----------------------------
    if mag_exists or ang_exists:
        # Column: Mag / Phase
        top_title, top_data = (
            ("Magnitude", plot_info["mag"][0]) if mag_exists
            else ("Phase", plot_info["ang"][0])
        )

        bottom = (
            ("Phase", plot_info["ang"][0])
            if mag_exists and ang_exists
            else None
        )

        columns.append({
            "title_top": top_title,
            "data_top": top_data,
            "bottom": bottom,
            "time_column": False
        })

        # Optional Real / Imag column
        if real_exists or imag_exists:
            top_title, top_data = (
                ("Real", plot_info["real"][0]) if real_exists
                else ("Imaginary", plot_info["imag"][0])
            )

            bottom = (
                ("Imaginary", plot_info["imag"][0])
                if real_exists and imag_exists
                else None
            )

            columns.append({
                "title_top": top_title,
                "data_top": top_data,
                "bottom": bottom,
                "time_column": False
            })

    else:
        # No Mag/Phase → Real/Imag only
        top_title, top_data = (
            ("Real", plot_info["real"][0]) if real_exists
            else ("Imaginary", plot_info["imag"][0])
        )

        bottom = (
            ("Imaginary", plot_info["imag"][0])
            if real_exists and imag_exists
            else None
        )

        columns.append({
            "title_top": top_title,
            "data_top": top_data,
            "bottom": bottom,
            "time_column": False
        })

    return columns


def plot_dynamic_frequency_modes(
    freq_signals,
    time_signals,
    time,
    freq,
    freq_modes,
    freq_range,
    signals_per_file,
    wave_params=None,
    base_title=None,
    signal_file=None,
    normalize=False,
    fft_shift=False
):

    # -------------------------------
    # Load requested frequency modes
    # -------------------------------
    requested = [m for m in freq_modes if m.lower() in valid_display_freq_modes]

    loaded = {}
    for mode in requested:
        key = mode_to_key[mode]
        loaded[mode] = freq_signals[key] if key in freq_signals.files else None

    freq_arrays = {
        "mag":  loaded.get("mag"),
        "ang":  loaded.get("ang"),
        "real": loaded.get("real"),
        "imag": loaded.get("imag"),
    }

    N = len(freq)

    if fft_shift or normalize:
        for key, arrays in freq_arrays.items():
            if arrays is not None:
                if fft_shift:
                    arrays = fftshift(arrays, axes=-1)
                if normalize and key != "ang":
                    arrays = arrays / N
                freq_arrays[key] = arrays

    # -------------------------------
    # Iteration setup
    # -------------------------------
    if time_signals is not None:
        signal_iter = enumerate(time_signals[:signals_per_file])
    else:
        signal_iter = range(signals_per_file)

    # -------------------------------
    # Loop per signal
    # -------------------------------
    for item in signal_iter:
        if time_signals is not None:
            idx, time_signal = item
        else:
            idx = item
            time_signal = None

        # -------------------------------
        # Wave parameters
        # -------------------------------
        wp = wave_params[idx] if wave_params is not None else None
        num_tones = len(wp) if wp is not None else None

        # -------------------------------
        # Suptitle
        # -------------------------------
        sup = f"{num_tones}-Tone Signals" if num_tones else "Signal"
        if base_title:
            sup = base_title + sup
        if signal_file:
            sup += f"\n{signal_file}"

        # -------------------------------
        # Marker data
        # -------------------------------
        if wp is not None:
            amps   = [w["amp"]   for w in wp]
            phases = [w["phase"] for w in wp]
            freqs_ = [w["freq"]  for w in wp]
            reals  = [w["real"]  for w in wp]
            imags  = [w["imag"]  for w in wp]

            neg_freqs  = [-f for f in freqs_]
            neg_phases = [-p for p in phases]
            imag_neg   = [-i for i in imags]
        else:
            amps = phases = freqs_ = reals = imags = []
            neg_freqs = neg_phases = imag_neg = []

        plot_info = {
            "mag":  (extract(freq_arrays["mag"], idx),  amps,   amps),
            "ang":  (extract(freq_arrays["ang"], idx),  phases, neg_phases),
            "real": (extract(freq_arrays["real"], idx), reals,  reals),
            "imag": (extract(freq_arrays["imag"], idx), imags,  imag_neg),
        }

        # -------------------------------
        # Build columns
        # -------------------------------
        columns = build_columns(plot_info, time_signal=time_signal)

        need_bottom = any(col["bottom"] for col in columns)
        nrows = 2 if need_bottom else 1
        ncols = len(columns)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols, 3 * nrows),
            squeeze=False
        )

        # -------------------------------
        # Plotting
        # -------------------------------
        for c, col in enumerate(columns):
            ax_top = axes[0, c]

            if col["time_column"]:
                ax_top.plot(time, col["data_top"])
                ax_top.set_xlim(-0.0002, 0.0002)
            else:
                ax_top.plot(freq, col["data_top"])
                ax_top.set_xlim(-freq_range[1], freq_range[1])

                if wp is not None:
                    mode = mode_from_title(col["title_top"])
                    _, pos_vals, neg_vals = plot_info[mode]
                    if pos_vals:
                        ax_top.scatter(freqs_, pos_vals, marker='x', color='red', s=100)
                        ax_top.scatter(neg_freqs, neg_vals, marker='x', color='red', s=100)

            ax_top.set_title(col["title_top"])

            if nrows == 2 and col["bottom"]:
                title, data = col["bottom"]
                ax_bot = axes[1, c]
                ax_bot.plot(freq, data)
                ax_bot.set_xlim(-freq_range[1], freq_range[1])
                ax_bot.set_title(title)

                if wp is not None:
                    mode = mode_from_title(title)
                    _, pos_vals, neg_vals = plot_info[mode]
                    if pos_vals:
                        ax_bot.scatter(freqs_, pos_vals, marker='x', color='red', s=100)
                        ax_bot.scatter(neg_freqs, neg_vals, marker='x', color='red', s=100)

        fig.suptitle(sup)
        fig.tight_layout()
        plt.show()