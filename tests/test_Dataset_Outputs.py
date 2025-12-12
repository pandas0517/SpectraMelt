'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import load_config_from_json, get_logger
    from pathlib import Path
    from spectramelt.NYFR import NYFR
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift, ifftshift
    import logging
    import pickle

    load_dotenv()
    
    create_output_set = True
    create_premultiply_set = True

    create_wbf_wave_params = True
    create_nyfr_wave_params = True    

    display_nyfr_signals = False
    display_premultiply_signals = False
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr = NYFR(config_file_path=Path(getenv('NYFR_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr.get_config_name(),
                      config_file_path=Path(getenv('DATASET_CONF')))

    if create_output_set:
        dataset.create_output_set(nyfr)

    if create_premultiply_set:
        dataset.create_premultiply_set()
        
    if create_wbf_wave_params:
        dataset.create_wbf_wave_params()

    if create_nyfr_wave_params:
        dataset.create_nyfr_wave_params(nyfr)

    directories = dataset.get_directories()
    input_dir = directories.get('inputs', "Inputs")
    output_dir = directories.get('outputs', "Outputs")    
    
    flat_filenames = dataset.get_flat_filenames()
    input_time_signal_filename = flat_filenames.get('input.time_signal', "time_signals.npy")
    input_signal_wave_params = flat_filenames.get('input.wave_params', "wave_params.pkl")

    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)
        
    if display_nyfr_signals:
        DUT_type = type(nyfr).__name__
        samp_time_filename = flat_filenames.get('samp_time', "sampled_time.npy")
        samp_freq_filename = flat_filenames.get('samp_freq', "sampled_freq.npy")
        samp_time = np.load(output_dir / samp_time_filename)
        samp_freq = np.load(output_dir / samp_freq_filename)
        signals_per_file = 3
        for file_path in output_dir.iterdir():
            
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                stem = file_path.name
                key_part = stem.split(input_time_signal_filename)[0]
                nyfr_wave_file = output_dir / f"{key_part}{input_signal_wave_params}"
                with open(nyfr_wave_file, "rb") as f:
                    nyfr_wave_params = pickle.load(f)
                signals = np.load(file_path)
                
                for idx, signal in enumerate(signals[:signals_per_file]):
                    # Extract amps and freqs
                    num_tones = len(nyfr_wave_params[idx])
                    amps = [w["amp"] / 2 for w in nyfr_wave_params[idx]]
                    freqs = [w["freq"] for w in nyfr_wave_params[idx]]
                    neg_freqs = [-f for f in freqs]
                    signal_freq = fftshift(np.abs(fft(signal))) / len(samp_freq)
                    fig, axes = plt.subplots(1, 2, figsize=(8,4))  # 1 rows, 2 columns
                    axes[0].plot(samp_time, signal)
                    axes[0].set_title("Time (File)")
                    axes[0].set_xlim(-0.0002, 0.0002)
                    axes[1].plot(samp_freq, signal_freq)
                    axes[1].scatter(freqs, amps, marker='x', color='red', s=100)  # s is marker size
                    axes[1].scatter(neg_freqs, amps, marker='x', color='red', s=100)  # s is marker size
                    axes[1].set_title("Frequency (File)")
                    axes[1].set_ylim(0, 0.25)
                    # axes[1].set_xlim(-400000, 400000)
                    fig.suptitle(f"Output for DUT Type {DUT_type}\n{num_tones}-Tone Signals")
                    fig.tight_layout()
                    plt.show()
                        
    if display_premultiply_signals:
        premultiply_dir = directories.get('premultiply', "Premultiply")
        
        wbf_time_filename = flat_filenames.get('wbf_time', "wbf_time.npy")
        wbf_freq_filename = flat_filenames.get('wbf_freq', "wbf_freq.npy")    
        wbf_time_file = output_dir / wbf_time_filename
        wbf_freq_file = output_dir / wbf_freq_filename
        wbf_time = np.load(wbf_time_file)
        wbf_freq = np.load(wbf_freq_file)

        signals_per_file = 3
        for file_path in premultiply_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):                  
                signals = np.load(file_path)
                for idx, signal in enumerate(signals[:signals_per_file]):
                    signal_time = ifft(ifftshift(signal))
                    signal_freq = signal / len(wbf_freq)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(8,4))  # 1 rows, 2 columns
                    axes[0].plot(wbf_time, signal_time)
                    axes[0].set_title("Time (File)")
                    axes[0].set_xlim(-0.0002, 0.0002)
                    axes[1].plot(wbf_freq, signal_freq)
                    axes[1].set_title("Frequency (File)")
                    axes[1].set_ylim(0, 0.25)
                    # axes[1].set_xlim(-400000, 400000)
                    fig.suptitle(f"Initial Recovery Guess Using Dictionary")
                    fig.tight_layout()
                    plt.show()
            
    atexit.register(logger.info, "Completed Test\n")