'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from pathlib import Path
    from spectramelt.Analog import Analog
    from spectramelt.InputSignal import InputSignal
    from spectramelt.LocalOscillator import LocalOscillator
    from spectramelt.PulseGenerator import PulseGenerator
    from spectramelt.WaveletGenerator import WaveletGenerator
    from spectramelt.Mixer import Mixer
    from spectramelt.LowPassFilter import LowPassFilter
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft, fftshift
    import numpy as np
    import time
    import logging
    
    load_dotenv()

    display_input_signals = True
    display_lo_pulse_signals = True
    display_wavelet_signals = True
    display_mixed_signals = True
    display_lpf_signals = True

    analog_1 = Analog(config_file_path=Path(getenv('INPUT_CONF')))
    analog_sig_1 = analog_1.create_analog()
    
    real_time_1 = analog_sig_1.time
    real_freq_1 = analog_sig_1.frequency
    total_time_1 = analog_sig_1.total_time
    sim_freq_1 = analog_1.get_time_params().get('sim_freq')

    input_signal_1 = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    real_input_1 = input_signal_1.create_input_signal(real_time=real_time_1)
    real_input_time_1 = real_input_1.input_signal
    real_input_freq_1 = fftshift(np.abs(fft(real_input_1.input_signal))) / (sim_freq_1*total_time_1)
    
    analog_2 = Analog()
    analog_time_params_2 = analog_2.get_time_params()
    analog_time_params_2["time_range"] = (0, 10)
    analog_time_params_2["sim_freq"] = 100000
    analog_2.set_time_params(analog_time_params_2)
    analog_sig_2 = analog_2.create_analog()

    real_time_2 = analog_sig_2.time
    real_freq_2 = analog_sig_2.frequency
    total_time_2 = analog_sig_2.total_time
    sim_freq_2 = analog_2.get_time_params().get('sim_freq')

    input_signal_2 = InputSignal()
    real_input_2 = input_signal_2.create_input_signal(real_time=real_time_2)
    real_input_time_2 = real_input_2.input_signal
    real_input_freq_2 = fftshift(np.abs(fft(real_input_2.input_signal))) / (sim_freq_2*total_time_2)
    
    if display_input_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, real_input_time_1)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, real_input_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[1,0].plot(real_time_2, real_input_time_2)
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
    lo_pre_start_1 = lo_signal_1.pre_start_lo

    pulse_gen_1 = PulseGenerator(config_file_path=Path(getenv('NFWBS_CONF')))
    pulse_signal_1 = pulse_gen_1.generate(lo_signal_1.lo, real_time_1, lo_pre_start_1)
    lo_freq_1 = fftshift(np.abs(fft(lo_signal_1.lo))) / (sim_freq_1*total_time_1)
    pulse_freq_1 = fftshift(np.abs(fft(pulse_signal_1.pulses))) / (sim_freq_1*total_time_1)
    
    lo_2 = LocalOscillator()
    lo_params_2 = lo_2.get_lo_params()
    lo_params_2["freq"] = 1
    lo_params_2["mod_enabled"] = False
    lo_2.set_lo_params(lo_params_2)
    lo_signal_2 = lo_2.generate_signal(real_time_2)
    lo_pre_start_2 = lo_signal_2.pre_start_lo

    pulse_gen_2 = PulseGenerator()
    pulse_params_2 = pulse_gen_2.get_pulse_params()
    pulse_params_2["pulse_width"] = 0.05
    pulse_gen_2.set_pulse_params(pulse_params_2)
    pulse_signal_2 = pulse_gen_2.generate(lo_signal_2.lo, real_time_2, lo_pre_start_2)
    lo_freq_2 = fftshift(np.abs(fft(lo_signal_2.lo))) / (sim_freq_2*total_time_2)
    pulse_freq_2 = fftshift(np.abs(fft(pulse_signal_2.pulses))) / (sim_freq_2*total_time_2)
    
    if display_lo_pulse_signals:
        fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, lo_signal_1.lo)
        axes[0,0].plot(real_time_1, pulse_signal_1.pulses)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.00002, 0.00002)
        axes[0,1].plot(real_freq_1, lo_freq_1)
        axes[0,1].set_title("LO Frequency (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[0,2].plot(real_freq_1, pulse_freq_1)
        axes[0,2].set_title("Pulse Frequency (File)")
        axes[1,0].plot(real_time_2, lo_signal_2.lo)
        axes[1,0].plot(real_time_2, pulse_signal_2.pulses)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 25)
        axes[1,1].plot(real_freq_2, lo_freq_2)
        axes[1,1].set_title("LO Frequency (Default)")
        axes[1,1].set_xlim(-0.2, 0.2)
        axes[1,2].plot(real_freq_2, pulse_freq_2)
        axes[1,2].set_title("Pulse Frequency (File)")
        axes[1,2].set_xlim(-200, 200)
        fig.suptitle("LO and Pulse Signals")
        fig.tight_layout()
        plt.show()
    
    wavelet_gen_1 = WaveletGenerator(config_file_path=Path(getenv('NFWBS_CONF')))
    # start = time.time()
    wavelet_sig_1 = wavelet_gen_1.generate_wavelet_train(pulse_signal_1.pulses, real_time_1, device="gpu")
    # stop = time.time()
    # wavelet_gen_gpu_time = stop - start
    # print(wavelet_gen_gpu_time)
    # start = time.time()
    # wavelet_sig_1_test = wavelet_gen_1.generate_wavelet_train(pulse_signal_1.pulses, real_time_1)
    # stop = time.time()
    # wavelet_gen_cpu_time = stop - start
    # print(wavelet_gen_cpu_time)  
    wavelet_freq_1 = fftshift(np.abs(fft(wavelet_sig_1.wavelet_train))) / (sim_freq_1*total_time_1)
    
    wavelet_gen_2 = WaveletGenerator()
    wavelet_param_2 = wavelet_gen_2.get_wavelet_params()
    wavelet_param_2["center_freq"] = 500
    wavelet_gen_2.set_wavelet_params(wavelet_param_2)
    wavelet_sig_2 = wavelet_gen_2.generate_wavelet_train(pulse_signal_2.pulses, real_time_2, device="gpu")
    wavelet_freq_2 = fftshift(np.abs(fft(wavelet_sig_2.wavelet_train))) / (sim_freq_2*total_time_2)
    
    if display_wavelet_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, wavelet_sig_1.wavelet_train.real)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.00002, 0.00002)
        axes[0,1].plot(real_freq_1, wavelet_freq_1)
        axes[0,1].set_title("Frequency Magnitude (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[1,0].plot(real_time_2, wavelet_sig_2.wavelet_train.real)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 10)
        axes[1,1].plot(real_freq_2, wavelet_freq_2)
        axes[1,1].set_title("Frequency Magnitude (Default)")
        # axes[1,1].set_xlim(4900, 5100)
        fig.suptitle("Gabor Wavelet Train Signals")
        fig.tight_layout()
        plt.show()
        
    mixed_1 = Mixer(config_file_path=Path(getenv('NFWBS_CONF')))
    mixed_signal_1 = mixed_1.mix(real_input_1.input_signal, wavelet_sig_1.wavelet_train)
    mixed_freq_1 = fftshift(np.abs(fft(mixed_signal_1.mixed))) / (sim_freq_1*total_time_1)

    mixed_2 = Mixer()
    mixed_signal_2 = mixed_2.mix(real_input_2.input_signal, wavelet_sig_2.wavelet_train)
    mixed_freq_2 = fftshift(np.abs(fft(mixed_signal_2.mixed))) / (sim_freq_2*total_time_2)
    
    if display_mixed_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, mixed_signal_1.mixed.real)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.00002, 0.00002)
        axes[0,1].plot(real_freq_1, mixed_freq_1)
        axes[0,1].set_title("Frequency Magnitude (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[1,0].plot(real_time_2, mixed_signal_2.mixed.real)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 10)
        axes[1,1].plot(real_freq_2, mixed_freq_2)
        axes[1,1].set_title("Frequency Magnitude (Default)")
        # axes[1,1].set_xlim(4900, 5100)
        fig.suptitle("Mixed Gabor Wavelet Train Signals")
        fig.tight_layout()
        plt.show()
        
    lpf_1 = LowPassFilter(config_file_path=Path(getenv('NFWBS_CONF')))
    lpf_signal_1 = lpf_1.apply_filter(mixed_signal_1.mixed, real_time_1)
    lpf_freq_1 = fftshift(np.abs(fft(lpf_signal_1.filtered))) / (sim_freq_1*total_time_1)
    
    lpf_2 = LowPassFilter()
    lpf_params_2 = lpf_2.get_lpf_params()
    lpf_params_2["cutoff_freq"] = 750
    lpf_2.set_lpf_params(lpf_params_2)
    lpf_signal_2 = lpf_2.apply_filter(mixed_signal_2.mixed, real_time_2)
    lpf_freq_2 = fftshift(np.abs(fft(lpf_signal_2.filtered))) / (sim_freq_2*total_time_2)
        
    if display_lpf_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, lpf_signal_1.filtered.real)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.00002, 0.00002)
        axes[0,1].plot(real_freq_1, lpf_freq_1)
        axes[0,1].set_title("Frequency Magnitude (File)")
        axes[0,1].set_xlim(-200000, 200000)
        axes[1,0].plot(real_time_2, lpf_signal_2.filtered.real)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.05)
        axes[1,1].plot(real_freq_2, lpf_freq_2)
        axes[1,1].set_title("Frequency Magnitude (Default)")
        # axes[1,1].set_xlim(-120, 120)
        fig.suptitle("Mixed Lowpass Filtered Gabor Wavelet Train Signals")
        fig.tight_layout()
        plt.show()