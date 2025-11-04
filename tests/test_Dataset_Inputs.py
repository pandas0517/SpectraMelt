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
    from scipy.fft import fft, fftshift

    load_dotenv()
    
    create_set = True
    test_max_min = True
    display_signals = True

    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_signal = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    dataset = DataSet(input_config_name=input_signal.get_config_name(), config_file_path=Path(getenv('DATASET_CONF')))

    if create_set:
        dataset.create_input_set(input_signal)
        
    directories = dataset.get_directories()
    input_dir = directories.get('inputs', "Inputs")
    filenames = dataset.get_filenames()
    input_signal_filename = filenames.get('input_signal', "signals.npy")
    
    if test_max_min:
        input_signal_params = input_signal.get_adc_params()
        v_ref_range = tuple(input_signal_params.get('v_ref_range', (0, 5)))
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_signal_filename):
                signals = np.load(file_path)
                for signal in enumerate(signals):
                    if (signal > v_ref_range[1]).any() or (signal < v_ref_range[0]).any():
                        logger.info(f"Value found out of range {v_ref_range[0]} - {v_ref_range[1]} for input set {file_path.stem}")
                        break

    if display_signals:
        input_wave_params_filename = filenames.get('input_wave_params', "wave_params.pkl")
        real_time_filename = filenames.get('real_time', "real_time.npy")
        real_freq_filename = filenames.get('real_freq', "real_freq.npy")
        real_time = np.load(input_dir / real_time_filename)
        real_freq = np.load(input_dir / real_freq_filename)
        signals_per_file = 3
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_signal_filename):
                # 1. Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(input_signal_filename)[0]
                
                # 2. Search for other files containing that portion
                for other_file in input_dir.iterdir():
                    if key_part in other_file.name and other_file.name.endswith(input_wave_params_filename):
                        with open(other_file, "rb") as f:
                            wave_params = pickle.load(f)
                                      
                signals = np.load(file_path)
                    
                for idx, signal in enumerate(signals[:signals_per_file]):
                    wave_param = wave_params[idx]
                    num_tones = len(wave_param)
                    # Extract amps and freqs
                    amps = [w["amp"] / 2 for w in wave_param]
                    freqs = [w["freq"] for w in wave_param]
                    neg_freqs = [-f for f in freqs]
                    signal_freq = fftshift(np.abs(fft(signal))) / len(real_freq)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(8,4))  # 1 rows, 2 columns
                    axes[0].plot(real_time, signal)
                    axes[0].set_title("Time (File)")
                    axes[0].set_xlim(-0.0002, 0.0002)
                    axes[1].plot(real_freq, signal_freq)
                    axes[1].scatter(freqs, amps, marker='x', color='red', s=100)  # s is marker size
                    axes[1].scatter(neg_freqs, amps, marker='x', color='red', s=100)  # s is marker size
                    axes[1].set_title("Frequency (File)")
                    axes[1].set_ylim(0, 0.15)
                    axes[1].set_xlim(-200000, 200000)
                    fig.suptitle(f"Simulated Analog\n{num_tones}-Tone Signals")
                    fig.tight_layout()
                    plt.show()
            
    atexit.register(logger.info, "Completed Test\n")