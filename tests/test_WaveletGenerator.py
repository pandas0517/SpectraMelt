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

    show_input_signal = False
    show_lo_signals = False
    show_wavelet_signals = True
    show_mixed_signals = True
    show_lpf_signals = True

    analog_1 = Analog(config_file_path=Path(getenv('INPUT_CONF')))
    analog_1_params = analog_1.get_time_params()
    analog_1_params["time_range"] = (-2, 2)
    analog_1_params["sim_freq"] = 1e6
    analog_1.set_time_params(analog_1_params)
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
    
    if show_input_signal:
        fig, axes = plt.subplots(1, 3, figsize=(8,4))  # 1 row, 3 columns
        axes[0].plot(real_time_1, real_input_time_1)
        axes[0].set_title("Time (Ideal)")
        axes[0].set_xlim(-0.0002, 0.0002)
        axes[1].plot(real_freq_1, real_input_freq_1)
        axes[1].set_title("Frequency - Magnitude (Ideal)")
        axes[1].set_xlim(-300000, 300000)
        axes[2].plot(real_freq_1, real_input_phase_1)
        axes[2].set_title("Frequency - Phase (Ideal)")
        axes[2].set_xlim(-300000, 300000)
        # fig.suptitle("Simulated Analog Signals")
        fig.tight_layout()
        plt.show()
    
    lo_1 = LocalOscillator(config_file_path=Path(getenv('NFWBS_CONF')))
    lo_params_1 = lo_1.get_lo_params()
    lo_params_1["freq"] = 1
    lo_params_1["mod_enabled"] = False
    lo_1.set_lo_params(lo_params_1)
    lo_signal_1 = lo_1.generate_signal(real_time_1)
    lo_pre_start_1 = lo_signal_1.pre_start_lo
    lo_freq_1 = fftshift(np.abs(fft(lo_signal_1.lo))) / (sim_freq_1*total_time_1)

    pulse_gen_1 = PulseGenerator(config_file_path=Path(getenv('NFWBS_CONF')))
    pulse_params_1 = pulse_gen_1.get_pulse_params()
    pulse_params_1["pulse_width"] = 0.05
    pulse_gen_1.set_pulse_params(pulse_params_1)
    pulse_signal_1 = pulse_gen_1.generate(lo_signal_1.lo, real_time_1, lo_pre_start_1)
    pulse_freq_1 = fftshift(np.abs(fft(pulse_signal_1.pulses))) / (sim_freq_1*total_time_1)
    
    if show_lo_signals:
        fig, axes = plt.subplots(1, 2, figsize=(4,2))  # 1 row, 2 columns
        axes[0].plot(real_time_1, pulse_signal_1.pulses, label="Pulses")
        axes[0].plot(real_time_1, lo_signal_1.lo, label="LO")
        axes[0].set_title("Time (Ideal)")
        axes[0].set_xlim(-1.5, 1.5)
        axes[0].legend()
        axes[1].plot(real_freq_1, pulse_freq_1)
        axes[1].set_title("Frequency Magnitude (Ideal)")
        axes[1].set_xlim(-1000, 1000)
        # fig.suptitle("LO and Pulse Signals")
        fig.tight_layout()
        plt.show()
    
    wavelet_gen_1 = WaveletGenerator(config_file_path=Path(getenv('NFWBS_CONF')))
    wavelet_param_1 = wavelet_gen_1.get_wavelet_params()
    wavelet_param_1["center_freq"] = 15000
    wavelet_gen_1.set_wavelet_params(wavelet_param_1)
    wavelet_sig_1 = wavelet_gen_1.generate_wavelet_train(pulse_signal_1.pulses, real_time_1,
                                                         device="cpu", return_scaling_factor=True)
    test = np.max(wavelet_sig_1.wavelet_train.real)
    wavelet_freq_1 = fftshift(np.abs(fft(wavelet_sig_1.wavelet_train))) / (sim_freq_1*total_time_1)
    wavelet_phase_1 = fftshift(np.angle(fft(wavelet_sig_1.wavelet_train)))
    
    wavelet_gen_2 = WaveletGenerator(config_file_path=Path(getenv('NFWBS_CONF')))
    wavelet_param_2 = wavelet_gen_2.get_wavelet_params()
    wavelet_param_2["center_freq"] = 15000
    wavelet_param_2["amp_noise_std"] = 0.8
    wavelet_param_2["freq_drift_ppm"] = 2e5
    wavelet_param_2["harmonic_distortion"] = 0.8
    wavelet_param_2["phase_noise_std"] = 2.0
    wavelet_param_2["threshold"] = 1.5e-1
    wavelet_gen_2.set_wavelet_params(wavelet_param_2)
    wavelet_sig_2 = wavelet_gen_2.generate_wavelet_train(pulse_signal_1.pulses, real_time_1, device="cpu")
    wavelet_freq_2 = fftshift(np.abs(fft(wavelet_sig_2.wavelet_train))) / (sim_freq_1*total_time_1)
    wavelet_phase_2 = fftshift(np.angle(fft(wavelet_sig_2.wavelet_train)))    
    
    if show_wavelet_signals:
        fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 1 rows, 3 columns
        axes[0,0].plot(real_time_1, wavelet_sig_1.wavelet_train.real)
        axes[0,0].set_title("Time (Ideal)")
        axes[0,0].set_xlim(-0.05, 0.1)
        axes[0,1].plot(real_freq_1, wavelet_freq_1)
        axes[0,1].set_title("Frequency Magnitude (Ideal)")
        axes[0,1].set_xlim(14900, 15100)
        axes[0,2].plot(real_freq_1, wavelet_phase_1)
        axes[0,2].set_title("Frequency Phase (Ideal)")
        axes[0,2].set_xlim(14900, 15100)
        axes[1,0].plot(real_time_1, wavelet_sig_2.wavelet_train.real)
        axes[1,0].set_title("Time (Noisy)")
        axes[1,0].set_xlim(-0.05, 0.1)
        axes[1,1].plot(real_freq_1, wavelet_freq_2)
        axes[1,1].set_title("Frequency Magnitude (Noisy)")
        axes[1,1].set_xlim(14900, 15100)
        axes[1,2].plot(real_freq_1, wavelet_phase_2)
        axes[1,2].set_title("Frequency Phase (Noisy)")
        axes[1,2].set_xlim(14900, 15100)
        # fig.suptitle("Gabor Wavelet Train Signals")
        fig.tight_layout()
        plt.show()
        
    mixed_1 = Mixer(config_file_path=Path(getenv('NFWBS_CONF')))
    mixed_signal_1 = mixed_1.mix(real_input_1.input_signal, wavelet_sig_1.wavelet_train)
    mixed_freq_1 = fftshift(np.abs(fft(mixed_signal_1.mixed))) / (sim_freq_1*total_time_1)
    
    if show_mixed_signals:
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