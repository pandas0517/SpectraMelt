'''
@author: pete
'''
if __name__ == '__main__':
    import os
    import sys
    # Add the src directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from dotenv import load_dotenv
    load_dotenv()
    from pathlib import Path
    from InputSignal import InputSignal
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    import numpy as np

    input_signal_1 = InputSignal(config_file_path=Path(os.getenv('SYSTEM_CONF')))
    real_time_1 = input_signal_1.get_analog_time()
    real_input_1 = input_signal_1.get_input_signal()
    real_freq_1 = input_signal_1.get_analog_frequency()
    total_time_1 = input_signal_1.get_analog_signal_params().get('total_time')
    sim_freq_1 = input_signal_1.get_time_params().get('sim_freq')
    real_input_freq_1 = np.fft.fftshift(np.abs(fft(real_input_1))) / (sim_freq_1*total_time_1)
    
    input_signal_2 = InputSignal()
    real_time_2 = input_signal_2.get_analog_time()
    real_input_2 = input_signal_2.get_input_signal()
    real_freq_2 = input_signal_2.get_analog_frequency()
    total_time_2 = input_signal_2.get_analog_signal_params().get('total_time')
    sim_freq_2 = input_signal_2.get_time_params().get('sim_freq')
    real_input_freq_2 = np.fft.fftshift(np.abs(fft(real_input_2))) / (sim_freq_2*total_time_2)
    
    fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
    axes[0,0].plot(real_time_1, real_input_1)
    axes[0,0].set_title("Time (File)")
    axes[0,1].plot(real_freq_1, real_input_freq_1)
    axes[0,1].set_title("Frequency (File)")
    axes[0,1].set_xlim(-5, 5)
    axes[1,0].plot(real_time_2, real_input_2)
    axes[1,0].set_title("Time (File)")
    axes[1,1].plot(real_freq_2, real_input_freq_2, label="Frequency (File)")
    axes[1,1].set_title("Frequency (File)")
    axes[1,1].set_xlim(-5, 5)
    fig.suptitle("Simulated Analog Signals")
    fig.tight_layout()
    plt.show()
