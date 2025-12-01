'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import load_config_from_json, get_logger
    from pathlib import Path
    from spectramelt.Recovery import Recovery
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift, ifftshift
    import logging
    import pickle
    from spectramelt.Recovery import VALID_SAVED_FREQ_MODES

    load_dotenv()
    
    create_set = False
    use_mlp = False

    decode_recovery_set = False
    decode_wbf_set = False

    create_recovery_dataframe = False
    set_recovery_dataframe = False

    display_recovered_signals = True
    DUT_type = "NYFR"
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr_config = load_config_from_json(Path(getenv('NYFR_CONF')))
    recovery = Recovery(config_file_path=Path(getenv('RECOVERY_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr_config.get('config_name'),
                      recovery_config_name=recovery.get_config_name(),
                      config_file_path=Path(getenv('DATASET_CONF')))
    directories = dataset.get_directories()
    output_dir = directories.get('outputs', "Outputs")
    recovery_dir = directories.get('recovery', "Recovery")
    saved_output_freq_modes = dataset.get_outputset_params().get("saved_freq_modes")
    # selected_freq_modes = saved_output_freq_modes
    selected_freq_modes = saved_output_freq_modes[1:2]

    if create_set:
        if use_mlp:
            from spectramelt.MLP import MLP
            mlp = MLP(config_file_path=Path(getenv('MLP_CONF')))
            dataset.create_recovery_set(recovery, mlp=mlp)
        else:
            dataset.create_recovery_set(recovery)

    if decode_recovery_set:
        dataset.decode_complex_sets(recovery_dir)

    if decode_wbf_set:
        dataset.decode_complex_sets(output_dir)

    if create_recovery_dataframe:
        dataset.create_recovery_dataframe()

    if set_recovery_dataframe:
        dataset.set_recovery_dataframe(selected_freq_modes)

    if display_recovered_signals:
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)

        flat_filenames = dataset.get_flat_filenames()
        FREQ_FILE_KEYS = DataSet.FREQ_FILE_KEYS

        # Load shared WBF metadata
        wbf_time = np.load(output_dir / flat_filenames.get('wbf_time', "wbf_time.npy"))
        wbf_freq = np.load(output_dir / flat_filenames.get('wbf_freq', "wbf_freq.npy"))

        input_wave_params_filename = flat_filenames.get('input.wave_params', "wave_params.pkl")

        signals_per_file = 3

        def shift_norm(arr):
            """FFT-shift and normalize."""
            return fftshift(arr) / len(wbf_freq)

        def plot_two_panel(title_left, data_left, title_right, data_right):
            fig, axes = plt.subplots(1, 2, figsize=(8,4))
            axes[0].plot(wbf_freq, data_left)
            axes[0].set_title(title_left)
            axes[1].plot(wbf_freq, data_right)
            axes[1].set_title(title_right)
            fig.suptitle("Actual vs Recovered Signals")
            fig.tight_layout()
            plt.show()

        def plot_four_panel(titles, datasets, wave_param=None):
            fig, axes = plt.subplots(2, 2, figsize=(8,4))
            axes_flat = axes.flat  # flatten for easy indexing

            for i, (ax, title, data) in enumerate(zip(axes_flat, titles, datasets)):
                ax.plot(wbf_freq, data)
                ax.set_title(title)

                if wave_param and i == 0:  # scatter only on first subplot
                    amps = np.array([w["amp"] / 2 for w in wave_param])
                    freqs = np.array([w["freq"] for w in wave_param])
                    logger.info(freqs)
                    neg_freqs = -freqs

                    pos_indices = [np.argmin(np.abs(wbf_freq - f)) for f in freqs]
                    neg_indices = [np.argmin(np.abs(wbf_freq - f)) for f in neg_freqs]

                    ax.scatter(wbf_freq[pos_indices], amps, marker='x', color='red', s=100)
                    ax.scatter(wbf_freq[neg_indices], amps, marker='x', color='red', s=100)

            fig.suptitle("Actual vs Recovered Signals")
            fig.tight_layout()
            plt.show()

        for mode in selected_freq_modes:
            if mode not in VALID_SAVED_FREQ_MODES:
                logger.warning(f"Skipping invalid freq mode: {mode}")
                continue

            key = FREQ_FILE_KEYS[mode]
            filename = flat_filenames.get(key)
            if not filename:
                logger.error(f"No filename configured for freq mode '{mode}' (key='{key}')")
                continue

            # npz modes must use .npz naming
            if mode in ("mag_ang_sincos", "mag_ang" ,"real_imag"):
                filename = str(Path(filename).with_suffix(".npz"))

            for file_path in recovery_dir.iterdir():

                # Find matching recovery files
                if not (file_path.is_file() and file_path.name.endswith(filename)):
                    continue

                stem = file_path.name
                key_part = stem.split(filename)[0]
                wbf_file = output_dir / f"{key_part}wbf_{filename}"
                wbf_wave_file = output_dir / f"{key_part}{input_wave_params_filename}"

                if not wbf_file.exists():
                    msg = f"{file_path} does not have a matching wbf file {wbf_file}"
                    logger.error(msg)
                    raise ValueError(msg)
                
                if not wbf_wave_file.exists():
                    msg = f"{wbf_wave_file} does not exist"
                    logger.error(msg)
                    raise ValueError(msg)
                
                with open(wbf_wave_file, "rb") as f:
                    wave_params = pickle.load(f)

                # Load recovery + WBF signals
                recovery_signals = np.load(file_path)
                wbf_signals = np.load(wbf_file)

                # -------------------------------
                # MODE: MAG + PHASE
                # -------------------------------
                if mode in ("mag_ang", "mag_ang_sincos"):
                    rec_mag = recovery_signals["complex_mag"]
                    rec_phase = recovery_signals["complex_phase"]
                    wbf_mag = wbf_signals["complex_mag"]
                    wbf_phase = wbf_signals["complex_phase"]

                    for idx in range(signals_per_file):
                        wave_param = wave_params[idx]
                        plot_four_panel(
                            ["WBF Magnitude", "Recovered Magnitude",
                            "WBF Phase",     "Recovered Phase"],
                            [
                                shift_norm(wbf_mag[idx]),
                                shift_norm(rec_mag[idx]) / 10,   # your original scaling
                                fftshift(wbf_phase[idx]),
                                fftshift(rec_phase[idx]),
                            ],
                            wave_param=wave_param
                        )

                # -------------------------------
                # MODE: MAG ONLY
                # -------------------------------
                elif mode == "mag":
                    for idx in range(signals_per_file):
                        plot_two_panel(
                            "WBF Magnitude",
                            shift_norm(wbf_signals[idx]),
                            "Recovered Magnitude",
                            shift_norm(recovery_signals[idx]),
                        )

                # -------------------------------
                # MODE: ANG ONLY
                # -------------------------------
                elif mode == "ang":
                    for idx in range(signals_per_file):
                        plot_two_panel(
                            "WBF Phase",
                            fftshift(wbf_signals[idx]),
                            "Recovered Phase",
                            fftshift(recovery_signals[idx]),
                        )

                # -------------------------------
                # MODE: REAL + IMAG
                # -------------------------------
                elif mode == "real_imag":
                    rec_real, rec_imag = recovery_signals["real"], recovery_signals["imag"]
                    wbf_real, wbf_imag = wbf_signals["real"], wbf_signals["imag"]

                    for idx in range(signals_per_file):
                        plot_four_panel(
                            ["WBF Real", "Recovered Real",
                            "WBF Imag", "Recovered Imag"],
                            [
                                shift_norm(wbf_real[idx]),
                                shift_norm(rec_real[idx]),
                                shift_norm(wbf_imag[idx]),
                                shift_norm(rec_imag[idx]),
                            ]
                        )

                # -------------------------------
                # MODE: REAL
                # -------------------------------
                elif mode == "real":
                    for idx in range(signals_per_file):
                        plot_two_panel(
                            "WBF Real",
                            shift_norm(wbf_signals[idx]),
                            "Recovered Real",
                            shift_norm(recovery_signals[idx]),
                        )

                # -------------------------------
                # MODE: IMAG
                # -------------------------------
                elif mode == "imag":
                    for idx in range(signals_per_file):
                        plot_two_panel(
                            "WBF Imag",
                            shift_norm(wbf_signals[idx]),
                            "Recovered Imag",
                            shift_norm(recovery_signals[idx]),
                        )

                # -------------------------------
                # MODE: COMPLEX
                # -------------------------------
                elif mode == "complex":
                    rec_c = recovery_signals["complex"]
                    wbf_c = wbf_signals["complex"]

                    for idx in range(signals_per_file):
                        plot_four_panel(
                            ["WBF Magnitude", "Recovered Magnitude",
                            "WBF Phase",     "Recovered Phase"],
                            [
                                shift_norm(np.abs(wbf_c[idx])),
                                shift_norm(np.abs(rec_c[idx])),
                                fftshift(np.angle(wbf_c[idx])),
                                fftshift(np.angle(rec_c[idx])),
                            ]
                        )
                     
    atexit.register(logger.info, "Completed Test\n")