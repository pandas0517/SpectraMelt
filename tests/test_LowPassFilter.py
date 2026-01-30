'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    load_dotenv()
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.Analog import Analog
    from spectramelt.LowPassFilter import LowPassFilter
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft, fftshift
    import numpy as np
    
    show_input_signals = False

    analog_1 = Analog(config_file_path=Path(getenv('INPUT_CONF')))
    analog_sig_1 = analog_1.create_analog()
    
    real_time_1 = analog_sig_1.time
    real_freq_1 = analog_sig_1.frequency
    total_time_1 = analog_sig_1.total_time
    sim_freq_1 = analog_1.get_time_params().get('sim_freq')

    input_signal_1 = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    input_signal_1_wave_params = input_signal_1.get_wave_params()
    input_signal_1_wave_params["v_ref_range"] = (-2.5, 2.5)
    input_signal_1.set_wave_params(input_signal_1_wave_params)
    
    real_input_1 = input_signal_1.create_input_signal(real_time=real_time_1)
    real_input_time_1 = real_input_1.input_signal
    real_input_freq_1 = fftshift(np.abs(fft(real_input_1.input_signal))) / (sim_freq_1*total_time_1)
    real_input_phase_1 = fftshift(np.angle(fft(real_input_1.input_signal)))
    
    if show_input_signals:
        fig, axes = plt.subplots(1, 3, figsize=(8,4))  # 2 rows, 3 columns
        axes[0].plot(real_time_1, real_input_time_1)
        axes[0].set_title("Time (Ideal)")
        axes[1].plot(real_freq_1, real_input_freq_1)
        axes[1].set_title("Frequency - Magnitude (Ideal)")
        axes[1].set_xlim(-300000, 300000)
        axes[2].plot(real_freq_1, real_input_phase_1)
        axes[2].set_title("Frequency - Phase (Ideal)")
        axes[2].set_xlim(-300000, 300000)
        # fig.suptitle("Simulated Analog Signals")
        fig.tight_layout()
        plt.show()
    
    lpf_1 = LowPassFilter(config_file_path=Path(getenv('NYFR_CONF')))
    lpf_1_params = lpf_1.get_lpf_params()
    lpf_1_params["cutoff_freq"] = 200000
    lpf_1.set_lpf_params(lpf_1_params)
    
    lpf_signal_1 = lpf_1.apply_filter(real_input_time_1, real_time_1)
    lpf_freq_1 = np.fft.fftshift(np.abs(fft(lpf_signal_1.filtered))) / (sim_freq_1*total_time_1)
    lpf_phase_1 = np.fft.fftshift(np.angle(fft(lpf_signal_1.filtered)))
    
    lpf_2 = LowPassFilter(config_file_path=Path(getenv('NYFR_CONF')))
    lpf_2_params = lpf_2.get_lpf_params()
    lpf_2_params["cutoff_freq"] = 200000
    lpf_2_params["noise_std"] = 2.0
    lpf_2.set_lpf_params(lpf_2_params)
    lpf_signal_2 = lpf_2.apply_filter(real_input_time_1, real_time_1)
    lpf_freq_2 = np.fft.fftshift(np.abs(fft(lpf_signal_2.filtered))) / (sim_freq_1*total_time_1)
    lpf_phase_2 = np.fft.fftshift(np.angle(fft(lpf_signal_2.filtered)))
    
    fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 3 columns
    axes[0,0].plot(real_time_1, lpf_signal_1.filtered)
    axes[0,0].set_title("Time (Ideal)")
    axes[0,0].set_xlim(-0.0002, 0.0002)
    axes[0,1].plot(real_freq_1, lpf_freq_1)
    axes[0,1].set_title("Frequency Magnitude (Ideal)")
    axes[0,1].set_xlim(-300000, 300000)
    axes[0,2].plot(real_freq_1, lpf_phase_1)
    axes[0,2].set_title("Frequency Phase (Ideal)")
    axes[0,2].set_xlim(-300000, 300000)
    axes[1,0].plot(real_time_1, lpf_signal_2.filtered)
    axes[1,0].set_title("Time (Noisy)")
    axes[1,0].set_xlim(-0.0002, 0.0002)
    axes[1,1].plot(real_freq_1, lpf_freq_2)
    axes[1,1].set_title("Frequency Magnitude (Noisy)")
    axes[1,1].set_xlim(-300000, 300000)
    axes[1,2].plot(real_freq_1, lpf_phase_2)
    axes[1,2].set_title("Frequency Phase (Noisy)")
    axes[1,2].set_xlim(-300000, 300000)
    # fig.suptitle("Mixed Signals")
    fig.tight_layout()
    plt.show()