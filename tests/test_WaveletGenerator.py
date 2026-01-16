'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.LocalOscillator import LocalOscillator
    from spectramelt.PulseGenerator import PulseGenerator
    from spectramelt.WaveletGenerator import WaveletGenerator
    from spectramelt.Mixer import Mixer
    from spectramelt.LowPassFilter import LowPassFilter
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft, fftshift
    import numpy as np
    import logging
    
    load_dotenv()

    display_input_signals = True
    display_lo_pulse_signals = True
    display_wavelet_signals = True
    display_mixed_signals = True
    display_lpf_signals = True

    input_signal_1 = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    input_time_params_1 = input_signal_1.get_time_params()
    real_time_1 = input_signal_1.get_analog_time()
    real_input_1 = input_signal_1.get_input_signal()
    real_freq_1 = input_signal_1.get_analog_frequency()
    total_time_1 = input_signal_1.get_analog_signal_params().get('total_time')
    sim_freq_1 = input_signal_1.get_time_params().get('sim_freq')
    real_input_freq_1 = fftshift(np.abs(fft(real_input_1))) / (sim_freq_1*total_time_1)
    
    input_signal_2 = InputSignal()
    input_time_params_2 = input_signal_2.get_time_params()
    input_time_params_2["time_range"] = (0, 100)
    input_time_params_2["sim_freq"] = 100000
    input_signal_2.set_time_params(input_time_params_2)
    input_signal_2.create_analog()
    input_signal_2.create_input_signal()
    real_time_2 = input_signal_2.get_analog_time()
    real_input_2 = input_signal_2.get_input_signal()
    real_freq_2 = input_signal_2.get_analog_frequency()
    total_time_2 = input_signal_2.get_analog_signal_params().get('total_time')
    sim_freq_2 = input_signal_2.get_time_params().get('sim_freq')
    real_input_freq_2 = fftshift(np.abs(fft(real_input_2))) / (sim_freq_2*total_time_2)
    
    if display_input_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, real_input_1)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, real_input_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[1,0].plot(real_time_2, real_input_2)
        axes[1,0].set_title("Time (Default)")
        # axes[1,0].set_xlim(0, 0.02)
        axes[1,1].plot(real_freq_2, real_input_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-2000, 2000)
        
        fig.suptitle("Simulated Analog Signals")
        fig.tight_layout()
        plt.show()
    
    lo_1 = LocalOscillator(config_file_path=Path(getenv('NFWBS_CONF')))
    lo_params_1 = lo_1.get_lo_params()
    lo_freq_1 = lo_params_1["freq"]
    time_shift = -5e-7
    lo_params_1["phase"] = -2 * np.pi * lo_freq_1 * time_shift
    lo_1.set_lo_params(lo_params_1)
    lo_signal_1 = lo_1.generate_signal(real_time_1)
    lo_pre_start_1 = lo_1.get_pre_start_lo()
    pulse_gen_1 = PulseGenerator(lo_signal_1, real_time_1, lo_pre_start_1,
                                 config_file_path=Path(getenv('NFWBS_CONF')))
    pulse_signal_1 = pulse_gen_1.get_pulse_signal()
    lo_freq_1 = fftshift(np.abs(fft(lo_signal_1))) / (sim_freq_1*total_time_1)
    pulse_freq_1 = fftshift(np.abs(fft(pulse_signal_1))) / (sim_freq_1*total_time_1)
    
    lo_2 = LocalOscillator()
    lo_params_2 = lo_2.get_lo_params()
    lo_params_2["freq"] = 0.1
    lo_params_2["mod_enabled"] = False
    lo_2.set_lo_params(lo_params_2)
    lo_signal_2 = lo_2.generate_signal(real_time_2)
    lo_pre_start_2 = lo_2.get_pre_start_lo()
    pulse_gen_2 = PulseGenerator()
    pulse_params_2 = pulse_gen_2.get_pulse_params()
    pulse_params_2["pulse_width"] = 0.05
    pulse_gen_2.set_pulse_params(pulse_params_2)
    pulse_signal_2 = pulse_gen_2.generate(lo_signal_2, real_time_2, lo_pre_start_2)
    lo_freq_2 = fftshift(np.abs(fft(lo_signal_2))) / (sim_freq_2*total_time_2)
    pulse_freq_2 = fftshift(np.abs(fft(pulse_signal_2))) / (sim_freq_2*total_time_2)
    
    if display_lo_pulse_signals:
        fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, lo_signal_1)
        axes[0,0].plot(real_time_1, pulse_signal_1)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.00002, 0.00002)
        axes[0,1].plot(real_freq_1, lo_freq_1)
        axes[0,1].set_title("LO Frequency (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[0,2].plot(real_freq_1, pulse_freq_1)
        axes[0,2].set_title("Pulse Frequency (File)")
        axes[1,0].plot(real_time_2, lo_signal_2)
        axes[1,0].plot(real_time_2, pulse_signal_2)
        axes[1,0].set_title("Time (Default)")
        # axes[1,0].set_xlim(0, 0.05)
        axes[1,1].plot(real_freq_2, lo_freq_2)
        axes[1,1].set_title("LO Frequency (Default)")
        axes[1,1].set_xlim(-120, 120)
        axes[1,2].plot(real_freq_2, pulse_freq_2)
        axes[1,2].set_title("Pulse Frequency (File)")
        fig.suptitle("LO and Pulse Signals")
        fig.tight_layout()
        plt.show()
    
    wavelet_gen_1 = WaveletGenerator(pulse_signal_1, real_time_1,
                                     config_file_path=Path(getenv('NFWBS_CONF')))
    wavelet_sig_1 = (wavelet_gen_1.get_wavelet_train()) / 5
    wavelet_freq_1 = fftshift(np.abs(fft(wavelet_sig_1))) / (sim_freq_1*total_time_1)
    
    wavelet_gen_2 = WaveletGenerator(pulse_signal_2, real_time_2)
    wavelet_sig_2 = wavelet_gen_2.get_wavelet_train()
    wavelet_freq_2 = fftshift(np.abs(fft(wavelet_sig_2))) / (sim_freq_2*total_time_2)
    
    if display_wavelet_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, wavelet_sig_1.real)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.00002, 0.00002)
        axes[0,1].plot(real_freq_1, wavelet_freq_1)
        axes[0,1].set_title("Frequency Magnitude (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[1,0].plot(real_time_2, wavelet_sig_2.real)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.05)
        axes[1,1].plot(real_freq_2, wavelet_freq_2)
        axes[1,1].set_title("Frequency Magnitude (Default)")
        axes[1,1].set_xlim(-120, 120)
        fig.suptitle("Gabor Wavelet Train Signals")
        fig.tight_layout()
        plt.show()
        
    mixed_1 = Mixer(real_input_1, wavelet_sig_1, config_file_path=Path(getenv('NFWBS_CONF')))
    mixed_signal_1 = mixed_1.get_mixed_signal()
    mixed_freq_1 = fftshift(np.abs(fft(mixed_signal_1))) / (sim_freq_1*total_time_1)

    mixed_2 = Mixer(real_input_2, wavelet_sig_2)
    mixed_signal_2 = mixed_2.get_mixed_signal()
    mixed_freq_2 = fftshift(np.abs(fft(mixed_signal_2))) / (sim_freq_2*total_time_2)
    
    if display_mixed_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, mixed_signal_1.real)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.00002, 0.00002)
        axes[0,1].plot(real_freq_1, mixed_freq_1)
        axes[0,1].set_title("Frequency Magnitude (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[1,0].plot(real_time_2, mixed_signal_2.real)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.05)
        axes[1,1].plot(real_freq_2, mixed_freq_2)
        axes[1,1].set_title("Frequency Magnitude (Default)")
        axes[1,1].set_xlim(-120, 120)
        fig.suptitle("Mixed Gabor Wavelet Train Signals")
        fig.tight_layout()
        plt.show()
        
    lpf_1 = LowPassFilter(mixed_1, config_file_path=Path(getenv('NFWBS_CONF')))
    lpf_signal_1 = lpf_1.get_signal_out()
    lpf_freq_1 = fftshift(np.abs(fft(lpf_signal_1))) / (sim_freq_1*total_time_1)
    
    lpf_2 = LowPassFilter(mixed_2)
    lpf_signal_2 = lpf_2.get_signal_out()
    lpf_freq_2 = fftshift(np.abs(fft(lpf_signal_2))) / (sim_freq_2*total_time_2)
        
    if display_lpf_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, lpf_signal_1.real)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.00002, 0.00002)
        axes[0,1].plot(real_freq_1, lpf_freq_1)
        axes[0,1].set_title("Frequency Magnitude (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[1,0].plot(real_time_2, lpf_signal_2.real)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.05)
        axes[1,1].plot(real_freq_2, lpf_freq_2)
        axes[1,1].set_title("Frequency Magnitude (Default)")
        axes[1,1].set_xlim(-120, 120)
        fig.suptitle("Mixed Lowpass Filtered Gabor Wavelet Train Signals")
        fig.tight_layout()
        plt.show()