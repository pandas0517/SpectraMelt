'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.NYFR import NYFR
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift, ifftshift

    load_dotenv()
    
    create_set = True
    display_output_signals = True
    display_premultiply_signals = True
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_signal = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    nyfr = NYFR(config_file_path=Path(getenv('NYFR_CONF')))
    DUT_type = type(nyfr).__name__
    dataset = DataSet(input_signal, nyfr, config_file_path=Path(getenv('DATASET_CONF')))

    directories = dataset.get_directories()
    input_dir = directories.get('inputs', "Inputs")
    output_dir = directories.get('outputs', "Outputs")    
    
    filenames = dataset.get_filenames()
    input_signal_filename = filenames.get('input_signal', "signals.npy")
    
    if create_set:
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_signal_filename):
                dataset.create_output_set(file_path)
        
    if display_output_signals:
        output_signal_filename = filenames.get('output_signal', "signals.npy")
        samp_time_filename = filenames.get('samp_time', "sampled_time.npy")
        samp_freq_filename = filenames.get('samp_freq', "sampled_freq.npy")
        samp_time = np.load(output_dir / samp_time_filename)
        samp_freq = np.load(output_dir / samp_freq_filename)
        signals_per_file = 3
        for file_path in output_dir.iterdir():
            
            if file_path.is_file() and file_path.name.endswith(output_signal_filename):
                       
                    signals = np.load(file_path)
                    
                    for idx, signal in enumerate(signals[:signals_per_file]):
                        signal_freq = fftshift(np.abs(fft(signal))) / len(samp_freq)
                        fig, axes = plt.subplots(1, 2, figsize=(8,4))  # 1 rows, 2 columns
                        axes[0].plot(samp_time, signal)
                        axes[0].set_title("Time (File)")
                        axes[0].set_xlim(-0.0002, 0.0002)
                        axes[1].plot(samp_freq, signal_freq)
                        axes[1].set_title("Frequency (File)")
                        axes[1].set_ylim(0, 0.25)
                        # axes[1].set_xlim(-400000, 400000)
                        fig.suptitle(f"Output for DUT Type {DUT_type}")
                        fig.tight_layout()
                        plt.show()
                        
    if display_premultiply_signals:
        premultiply_dir = directories.get('premultiply', "Premultiply")
        
        wbf_time_filename = filenames.get('wbf_time', "wbf_time.npy")
        wbf_freq_filename = filenames.get('wbf_freq', "wbf_freq.npy")    
        wbf_time_file = output_dir / wbf_time_filename
        wbf_freq_file = output_dir / wbf_freq_filename
        wbf_time = np.load(wbf_time_file)
        wbf_freq = np.load(wbf_freq_file)

        signals_per_file = 3
        for file_path in premultiply_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(output_signal_filename):                  
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