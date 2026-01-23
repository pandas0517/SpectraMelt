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
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft, fftshift
    import numpy as np

    analog_1 = Analog(config_file_path=Path(getenv('INPUT_CONF')))
    
    analog_sig_1 = analog_1.create_analog()
    
    real_time_1 = analog_sig_1.time
    real_freq_1 = analog_sig_1.frequency
    total_time_1 = analog_sig_1.total_time
    sim_freq_1 = analog_1.get_time_params().get('sim_freq')

    input_signal_1 = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    input_signal_1_wave_params = input_signal_1.get_wave_params()
    input_signal_1_wave_params["v_ref_range"] = (0, 5)
    input_signal_1.set_wave_params(input_signal_1_wave_params)
    real_input_1 = input_signal_1.create_input_signal(real_time=real_time_1)
    real_input_time_1 = real_input_1.input_signal
    real_input_freq_1 = fftshift(np.abs(fft(real_input_1.input_signal))) / (sim_freq_1*total_time_1)
    real_input_phase_1 = fftshift(np.angle(fft(real_input_1.input_signal)))

    input_signal_2 = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    input_signal_2_env_params = input_signal_2.get_env_params()
    input_signal_2_env_params["noise_level"] = 0.5
    input_signal_2_env_params["attenuation"] = 0.7
    input_signal_2_env_params["doppler"] = 5e-5
    input_signal_2_env_params["delay"] = 0.5e-6
    input_signal_2_env_params["num_echoes"] = 2
    input_signal_2_env_params["max_delay"] = 1e-6
    input_signal_2_env_params["max_doppler"] = 1e-4
    input_signal_2_env_params["phase_inversion_prob"] = 0.2
    input_signal_2.set_env_params(input_signal_2_env_params)
    input_signal_2_wave_params = input_signal_2.get_wave_params()
    input_signal_2_wave_params["v_ref_range"] = (0, 5)
    input_signal_2.set_wave_params(input_signal_2_wave_params)
    real_input_2 = input_signal_2.create_input_signal(real_time=real_time_1)
    real_input_time_2 = real_input_2.input_signal
    real_input_freq_2 = fftshift(np.abs(fft(real_input_2.input_signal))) / (sim_freq_1*total_time_1)
    real_input_phase_2 = fftshift(np.angle(fft(real_input_2.input_signal)))
    
    fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 3 columns
    axes[0,0].plot(real_time_1, real_input_time_1)
    axes[0,0].set_title("Time (Ideal)")
    axes[0,1].plot(real_freq_1, real_input_freq_1)
    axes[0,1].set_title("Frequency - Magnitude (Ideal)")
    axes[0,1].set_xlim(-200000, 200000)
    axes[0,2].plot(real_freq_1, real_input_phase_1)
    axes[0,2].set_title("Frequency - Phase (Ideal)")
    axes[0,2].set_xlim(-200000, 200000)
    axes[1,0].plot(real_time_1, real_input_time_2)
    axes[1,0].set_title("Time (Real World)")
    axes[1,1].plot(real_freq_1, real_input_freq_2)
    axes[1,1].set_title("Frequency - Magnitude (Real World)")
    axes[1,1].set_xlim(-200000, 200000)
    axes[1,2].plot(real_freq_1, real_input_phase_2)
    axes[1,2].set_title("Frequency - Phase (Ideal)")
    axes[1,2].set_xlim(-200000, 200000)
    fig.suptitle("Simulated Analog Signals")
    fig.tight_layout()
    plt.show()
