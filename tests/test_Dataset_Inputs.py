'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    import logging

    load_dotenv()

    create_set = False
    test_max_min = False
    display_signals = True

    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_signal = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    dataset = DataSet(input_config_name=input_signal.get_config_name(), config_file_path=Path(getenv('DATASET_CONF')))

    if create_set:
        dataset.create_input_set(input_signal)
        
    directories = dataset.get_directories()
    input_dir = directories.get('inputs', "Inputs")
    filenames = dataset.get_filenames()
    input_time_signal_filename = filenames.get('time_signals', "time_signals.npy")
    input_freq_signal_filename = filenames.get('freq_signals', "freq_signals.npz")
    
    if test_max_min:
        input_signal_params = input_signal.get_adc_params()
        v_ref_range = tuple(input_signal_params.get('v_ref_range', (0, 5)))
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                signals = np.load(file_path)
                for signal in signals:
                    if (signal > v_ref_range[1]).any() or (signal < v_ref_range[0]).any():
                        logger.info(f"Value found out of range {v_ref_range[0]} - {v_ref_range[1]} for input set {file_path.stem}")
                        break

    if display_signals:
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        input_signal_wave_params = input_signal.get_wave_params()
        freq_range = input_signal_wave_params.get('freq_range')
        amp_range = input_signal_wave_params.get('amp_range')
        input_wave_params_filename = filenames.get('input.wave_params', "wave_params.pkl")
        real_time_freq_filename = filenames.get('real_time_freq', "real_time_freq.npz")
        real_time_freq = np.load(input_dir / real_time_freq_filename)
        real_time = real_time_freq["real_time"]
        real_freq = real_time_freq["real_freq"]

        signals_per_file = 3
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                # 1. Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(input_time_signal_filename)[0]
                
                # 2. Search for other files containing that portion
                for other_file in input_dir.iterdir():
                    if key_part in other_file.name and other_file.name.endswith(input_wave_params_filename):
                        with open(other_file, "rb") as f:
                            wave_params = pickle.load(f)
                    elif key_part in other_file.name and other_file.name.endswith(input_freq_signal_filename):
                        freq_signals = np.load(other_file)
                        freq_mag_signals = freq_signals["mag"]
                        freq_phase_signals = freq_signals["ang"]
                        freq_real_signals = freq_signals["real"]
                        freq_imag_signals = freq_signals["imag"]
                                      
                time_signals = np.load(file_path)
                    
                for idx, time_signal in enumerate(time_signals[:signals_per_file]):
                    wave_param = wave_params[idx]
                    num_tones = len(wave_param)
                    # Extract amps and freqs
                    amps = [w["amp"] / 2 for w in wave_param]
                    freqs = [w["freq"] for w in wave_param]
                    phases = [
                        ((w["phase"] - 1.5*np.pi + np.pi) % (2*np.pi)) - np.pi
                        for w in wave_param
                    ]
                    test_phases = [((w["phase"] + 1.5*np.pi - np.pi) % (2*np.pi)) + np.pi for w in wave_param]
                    neg_freqs = [-f for f in freqs]
                    neg_phases = [-p for p in phases]
                    # compute real/imag for positive frequencies
                    real_pos = [a * np.cos(p) for a, p in zip(amps, test_phases)]
                    imag_pos = [a * np.sin(p) for a, p in zip(amps, test_phases)]

                    # compute real/imag for negative frequencies (conjugate)
                    real_neg = [a * np.cos(p) for a, p in zip(amps, neg_phases)]
                    imag_neg = [a * np.sin(p) for a, p in zip(amps, neg_phases)]

                    freq_mag_signal = freq_mag_signals[idx]
                    freq_phase_signal = freq_phase_signals[idx]
                    freq_real_signal = freq_real_signals[idx]
                    freq_imag_signal = freq_imag_signals[idx]

                    fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 1 rows, 2 columns
                    axes[0,0].plot(real_time, time_signal)
                    axes[0,0].set_title("Time")
                    axes[0,0].set_xlim(-0.0002, 0.0002)
                    axes[0,1].plot(real_freq, freq_mag_signal)
                    axes[0,1].scatter(freqs, amps, marker='x', color='red', s=100)  # s is marker size
                    axes[0,1].scatter(neg_freqs, amps, marker='x', color='red', s=100)  # s is marker size
                    axes[0,1].set_title("Frequency (Magnitude)")
                    axes[0,1].set_ylim(0, amp_range[1])
                    axes[0,1].set_xlim(-freq_range[1], freq_range[1])
                    axes[0,2].plot(real_freq, freq_phase_signal)
                    axes[0,2].scatter(freqs, phases, marker='x', color='red', s=100)  # s is marker size
                    axes[0,2].scatter(neg_freqs, neg_phases, marker='x', color='red', s=100)  # s is marker size
                    axes[0,2].set_title("Frequency (Phase)")
                    axes[0,2].set_xlim(-freq_range[1], freq_range[1])
                    axes[1,1].plot(real_freq, freq_real_signal)
                    axes[1,1].scatter(freqs, real_pos, marker='x', color='red', s=100)  # s is marker size
                    axes[1,1].scatter(neg_freqs, real_neg, marker='x', color='red', s=100)  # s is marker size
                    axes[1,1].set_title("Frequency (Real)")
                    axes[1,1].set_xlim(-freq_range[1], freq_range[1])
                    axes[1,2].plot(real_freq, freq_imag_signal)
                    axes[1,2].scatter(freqs, imag_pos, marker='x', color='red', s=100)  # s is marker size
                    axes[1,2].scatter(neg_freqs, imag_neg, marker='x', color='red', s=100)  # s is marker size
                    axes[1,2].set_title("Frequency (Imagingary)")
                    axes[1,2].set_xlim(-freq_range[1], freq_range[1])
                    fig.suptitle(f"Simulated Analog\n{num_tones}-Tone Signals")
                    fig.tight_layout()
                    plt.show()
            
    atexit.register(logger.info, "Completed Test\n")