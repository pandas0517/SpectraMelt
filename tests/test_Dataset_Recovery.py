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
    from spectramelt.Recovery import VALID_SAVED_FREQ_MODES

    load_dotenv()
    
    create_set = False
    use_mlp = False

    decode_recovery_set = False
    decode_wbf_set = False
    create_recovery_dataframe = False
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
                        
    if display_recovered_signals:
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)

        flat_filenames = dataset.get_flat_filenames()
        saved_output_freq_modes = dataset.get_outputset_params().get("saved_freq_modes")
        # selected_freq_modes = saved_output_freq_modes
        selected_freq_modes = saved_output_freq_modes[1:2]
        # input_time_signal_filename = flat_filenames.get('input.time_signal', "time_signals.npy")
        # input_freq_signal_filename = flat_filenames.get('input.freq.mag_sig', "freq_mag_signals.npy")
        # recovered_signal_filename = flat_filenames.get("recovered", "recovered.npy")
        FREQ_FILE_KEYS = DataSet.FREQ_FILE_KEYS

        wbf_time_filename = flat_filenames.get('wbf_time', "wbf_time.npy")
        wbf_freq_filename = flat_filenames.get('wbf_freq', "wbf_freq.npy")    
        wbf_time_file = output_dir / wbf_time_filename
        wbf_freq_file = output_dir / wbf_freq_filename
        wbf_time = np.load(wbf_time_file)
        wbf_freq = np.load(wbf_freq_file)
        
        signals_per_file = 3
        for mode in selected_freq_modes: 
            for file_path in recovery_dir.iterdir():

                if mode not in VALID_SAVED_FREQ_MODES:
                    logger.warning(f"Skipping invalid freq mode: {mode}")
                    continue

                key = FREQ_FILE_KEYS[mode]
                filename = flat_filenames.get(key)
                if not filename:
                    logger.error(f"No filename configured for freq mode '{mode}' (key='{key}')")
                    continue

                if mode == "mag_ang_sincos":
                    filename = str(Path(filename).with_suffix(".npz"))

                if file_path.is_file() and file_path.name.endswith(filename):
                    # 1. Extract identifying portion (for example, everything up to "signals.npy")
                    stem = file_path.name
                    key_part = stem.split(filename)[0]
                    wbf_file = output_dir / f"{key_part}wbf_{filename}"

                    if not wbf_file.exists():
                        logger.error(f"{file_path} does not have a matching wbf file {wbf_file}")
                        raise ValueError(f"{file_path} does not have a matching wbf file {wbf_file}")
                    
                    # input_time_signals = None
                    # input_freq_signals = None
                    # 2. Search for other files containing that portion
                    # for input_file in input_dir.iterdir():
                    #     if key_part in input_file.name and input_file.name.endswith(input_time_signal_filename):
                    #         input_time_signals = np.load(input_file)
                    #     if key_part in input_file.name and input_file.name.endswith(input_freq_signal_filename):
                    #         input_freq_signals = np.load(input_file)
                    # if input_time_signals is None:
                    #     logger.error("No matching input set file")
                    #     raise ValueError("No matching input set file") 
                        
                    recovery_signals = np.load(file_path)
                    recovery_mag_signals = recovery_signals["complex_mag"]
                    recovery_phase_signals = recovery_signals["complex_phase"]
                    
                    wbf_signals = np.load(wbf_file)
                    wbf_mag_signals = wbf_signals["complex_mag"]
                    wbf_phase_signals = wbf_signals["complex_phase"]

                    for idx, recovery_mag_signal in enumerate(recovery_mag_signals[:signals_per_file]):
                        # if input_freq_signals is None:
                        #     freq_signal = fftshift(np.abs(fft(time_signal))) / len(real_freq)
                        # else:
                        #     freq_signal = input_freq_signals[idx]
                        recovery_mag_signal = fftshift(recovery_mag_signal) / len(wbf_freq)
                        recovery_phase_signal = fftshift(recovery_phase_signals[idx])
                        wbf_mag_signal = fftshift(wbf_mag_signals[idx]) / len(wbf_freq)
                        wbf_phase_signal = fftshift(wbf_phase_signals[idx])
                            
                        # recovered_freq = recovery_signals[idx] / len(wbf_freq)
                        # recovered_time = ifft(ifftshift(recovered_freq))
                        
                        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
                        axes[0,0].plot(wbf_freq, wbf_mag_signal)
                        axes[0,0].set_title("WBF Frequency (Magnitude)")
                        # axes[0,0].set_xlim(-0.0002, 0.0002)
                        axes[0,1].plot(wbf_freq, wbf_phase_signal)
                        axes[0,1].set_title("WBF Frequency (Phase)")
                        axes[0,1].set_ylim(0, 0.25)
                        # axes[1].set_xlim(-400000, 400000)
                        axes[1,0].plot(wbf_freq, recovery_mag_signal)
                        axes[1,0].set_title("Recovered Frequency (Magnitude)")
                        # axes[1,0].set_xlim(-0.0002, 0.0002)
                        axes[1,1].plot(wbf_freq, recovery_phase_signal)
                        axes[1,1].set_title("Recovered Frequency (Phase)")
                        axes[1,1].set_ylim(0, 0.25)
                        # axes[1].set_xlim(-400000, 400000)
                        fig.suptitle(f"Actual vs Recovered Signals")
                        fig.tight_layout()
                        plt.show()
            
    atexit.register(logger.info, "Completed Test\n")