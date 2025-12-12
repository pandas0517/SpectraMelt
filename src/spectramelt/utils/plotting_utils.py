import matplotlib.pyplot as plt

# --- Title mapping ---
mode_to_title = {
    "mag": "Magnitude",
    "ang": "Phase",
    "real": "Real",
    "imag": "Imaginary",
}

# Map modes to keys inside .npz
mode_to_key = {
    "mag": "mag",
    "ang": "ang",
    "real": "real",
    "imag": "imag",
}

valid_display_freq_modes = {"real", "imag", "mag", "ang"}

def plot_dynamic_frequency_modes(
    wave_params,
    freq_signals,
    time_signals,
    time,
    freq,
    freq_modes,
    freq_range,
    signals_per_file,
    base_title=None,
    signal_file=None
):
    """
    Generate dynamic 1-row plots for time domain + selected freq-domain modes.
    """
    display_freq_modes = []
    for mode in freq_modes:
        if mode.lower() in valid_display_freq_modes:
            display_freq_modes.append(mode)
            
    # --- SAFE LOAD OF ONLY REQUESTED MODES ---
    loaded_freq_signals = {}

    for mode in display_freq_modes:
        key = mode_to_key[mode]
        if key in freq_signals.files:
            loaded_freq_signals[mode] = freq_signals[key]
        else:
            print(f"WARNING: Mode '{mode}' requested but key '{key}' not found in file.")
            loaded_freq_signals[mode] = None

    # Process each time-domain signal
    for idx, time_signal in enumerate(time_signals[:signals_per_file]):

        wave_param = wave_params[idx]
        num_tones = len(wave_param)
        sup_title = f"{num_tones}-Tone Signals"
        if base_title is not None:
            sup_title = base_title + sup_title
        if signal_file is not None:
            sup_title = sup_title + f"\n{signal_file}"

        # Extract parameters
        amps   = [w["amp"] / 2 for w in wave_param]
        phases = [w["phase"] for w in wave_param]
        freqs  = [w["freq"] for w in wave_param]
        neg_freqs   = [-f for f in freqs]
        neg_phases  = [-p for p in phases]
        real        = [w["real"] for w in wave_param]
        imag_pos    = [w["imag"] for w in wave_param]
        imag_neg    = [-i for i in imag_pos]

        # Easy-access arrays
        freq_mag_signal   = loaded_freq_signals.get("mag")
        freq_phase_signal = loaded_freq_signals.get("ang")
        freq_real_signal  = loaded_freq_signals.get("real")
        freq_imag_signal  = loaded_freq_signals.get("imag")

        mode_marker_data = {
            "mag":  (freq_mag_signal,   amps,      amps),
            "ang":  (freq_phase_signal, phases,    neg_phases),
            "real": (freq_real_signal,  real,      real),
            "imag": (freq_imag_signal,  imag_pos,  imag_neg),
        }

        valid_modes = [
            m for m in display_freq_modes
            if loaded_freq_signals.get(m) is not None
        ]

        # Number of columns: time-domain + freq modes
        num_cols = 1 + len(valid_modes)

        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

        if num_cols == 1:
            axes = [axes]

        # --- TIME DOMAIN ---
        axes[0].plot(time, time_signal)
        axes[0].set_title("Time (Signal)")
        axes[0].set_xlim(-0.0002, 0.0002)

        # --- FREQUENCY DOMAIN ---
        col = 1
        for mode in valid_modes:

            data, pos_vals, neg_vals = mode_marker_data[mode]
            if data is None:
                continue

            ax = axes[col]
            ax.plot(freq, data)
            ax.scatter(freqs, pos_vals, marker='x', color='red', s=100)
            ax.scatter(neg_freqs, neg_vals, marker='x', color='red', s=100)
            ax.set_title(mode_to_title[mode])
            ax.set_xlim(-freq_range[1], freq_range[1])

            col += 1

        fig.suptitle(sup_title)
        fig.tight_layout()
        plt.show()
