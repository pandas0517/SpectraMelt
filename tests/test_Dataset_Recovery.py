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

    load_dotenv()
    
    create_set = True
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
    
    if create_set:
        dataset.create_recovery_set(recovery)
                        
    if display_recovered_signals:
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        
        directories = dataset.get_directories()
        flat_filenames = dataset.get_flat_filenames()
        
        output_dir = directories.get('outputs', "Outputs")
        
        input_dir = directories.get('inputs', "Inputs")
        recovery_dir = directories.get('recovery', "Recovery")
        
        input_time_signal_filename = flat_filenames.get('input.time_signal', "time_signals.npy")
        input_freq_signal_filename = flat_filenames.get('input.freq.mag_sig', "freq_mag_signals.npy")
        recovered_signal_filename = flat_filenames.get("recovered", "recovered.npy")
        
        wbf_time_filename = flat_filenames.get('wbf_time', "wbf_time.npy")
        wbf_freq_filename = flat_filenames.get('wbf_freq', "wbf_freq.npy")    
        wbf_time_file = output_dir / wbf_time_filename
        wbf_freq_file = output_dir / wbf_freq_filename
        wbf_time = np.load(wbf_time_file)
        wbf_freq = np.load(wbf_freq_file)
        
        real_time_filename = flat_filenames.get('real_time', "real_time.npy")
        real_freq_filename = flat_filenames.get('real_freq', "real_freq.npy")
        real_time_file = input_dir / real_time_filename
        real_freq_file = input_dir / real_freq_filename
        real_time = np.load(real_time_file)
        real_freq = np.load(real_freq_file)
        
        signals_per_file = 3
        for file_path in recovery_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(recovered_signal_filename):
                # 1. Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(recovered_signal_filename)[0]
                
                input_time_signals = None
                input_freq_signals = None
                # 2. Search for other files containing that portion
                for input_file in input_dir.iterdir():
                    if key_part in input_file.name and input_file.name.endswith(input_time_signal_filename):
                        input_time_signals = np.load(input_file)
                    if key_part in input_file.name and input_file.name.endswith(input_freq_signal_filename):
                        input_freq_signals = np.load(input_file)
                if input_time_signals is None:
                    logger.error("No matching input set file")
                    raise ValueError("No matching input set file")  
                     
                recovery_signals = np.load(file_path)
                for idx, time_signal in enumerate(input_time_signals[:signals_per_file]):
                    if input_freq_signals is None:
                        freq_signal = fftshift(np.abs(fft(time_signal))) / len(real_freq)
                    else:
                        freq_signal = input_freq_signals[idx]
                        
                    recovered_freq = recovery_signals[idx] / len(wbf_freq)
                    recovered_time = ifft(ifftshift(recovered_freq))
                    
                    fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
                    axes[0,0].plot(real_time, time_signal)
                    axes[0,0].set_title("Time (File)")
                    axes[0,0].set_xlim(-0.0002, 0.0002)
                    axes[0,1].plot(real_freq, freq_signal)
                    axes[0,1].set_title("Frequency (File)")
                    axes[0,1].set_ylim(0, 0.25)
                    # axes[1].set_xlim(-400000, 400000)
                    axes[1,0].plot(wbf_time, recovered_time)
                    axes[1,0].set_title("Time (File)")
                    axes[1,0].set_xlim(-0.0002, 0.0002)
                    axes[1,1].plot(wbf_freq, recovered_freq)
                    axes[1,1].set_title("Frequency (File)")
                    axes[1,1].set_ylim(0, 0.25)
                    # axes[1].set_xlim(-400000, 400000)
                    fig.suptitle(f"Actual vs Recovered Signals")
                    fig.tight_layout()
                    plt.show()
            
    atexit.register(logger.info, "Completed Test\n")