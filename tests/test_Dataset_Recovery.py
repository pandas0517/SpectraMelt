'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger
    from pathlib import Path
    from spectramelt.Recovery import Recovery
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift, ifftshift

    load_dotenv()
    
    create_set = True
    display_recovered_signals = True
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    recovery = Recovery(config_file_path=Path(getenv('RECOVERY_CONF')))
    dataset = DataSet(recovery=recovery, config_file_path=Path(getenv('DATASET_CONF')))
    
    directories = dataset.get_directories()
    filenames = dataset.get_filenames()
    
    output_dir = directories.get('outputs', "Outputs")
    output_signal_filename = filenames.get('output_signal', "signals.npy")
    
    if create_set:
        for file_path in output_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(output_signal_filename):
                dataset.create_recovery_set(file_path)
                        
    if display_recovered_signals:
        DUT_type = type(nyfr).__name__
        input_dir = directories.get('inputs', "Inputs")
        recovery_dir = directories.get('recovery', "Recovery")
        
        input_signal_filename = filenames.get('input_signal', "signals.npy")
        recovered_signal_filename = filenames.get("recovered", "recovered.npy")
        
        wbf_time_filename = filenames.get('wbf_time', "wbf_time.npy")
        wbf_freq_filename = filenames.get('wbf_freq', "wbf_freq.npy")    
        wbf_time_file = output_dir / wbf_time_filename
        wbf_freq_file = output_dir / wbf_freq_filename
        wbf_time = np.load(wbf_time_file)
        wbf_freq = np.load(wbf_freq_file)
        
        real_time_filename = filenames.get('real_time', "real_time.npy")
        real_freq_filename = filenames.get('real_freq', "real_freq.npy")
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
                
                input_signals = None
                # 2. Search for other files containing that portion
                for input_file in input_dir.iterdir():
                    if key_part in input_file.name and input_file.name.endswith(input_signal_filename):
                        input_signals = np.load(input_file)
                if input_signals is None:
                    logger.error("No matching input set file")
                    raise ValueError("No matching input set file")  
                     
                recovery_signals = np.load(file_path)
                for idx, signal in enumerate(input_signals[:signals_per_file]):
                    signal_freq = fftshift(np.abs(fft(signal))) / len(real_freq)
                    recovered_freq = recovery_signals[idx] / len(wbf_freq)
                    recovered_time = ifft(ifftshift(recovered_freq))
                    
                    fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
                    axes[0,0].plot(real_time, signal)
                    axes[0,0].set_title("Time (File)")
                    axes[0,0].set_xlim(-0.0002, 0.0002)
                    axes[0,1].plot(real_freq, signal_freq)
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