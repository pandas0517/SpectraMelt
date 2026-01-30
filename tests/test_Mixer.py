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
    from spectramelt.LocalOscillator import LocalOscillator
    from spectramelt.PulseGenerator import PulseGenerator
    from spectramelt.LowPassFilter import LowPassFilter
    from spectramelt.Mixer import Mixer
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft, fftshift
    import numpy as np
    
    show_input_signal = False
    show_wbf_signal = False
    show_LO_signals = False

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

    lpf_1 = LowPassFilter(config_file_path=Path(getenv('NYFR_CONF')))
    lpf_1_params = lpf_1.get_lpf_params()
    lpf_1_params["cutoff_freq"] = 200000
    lpf_1.set_lpf_params(lpf_1_params)
    
    lpf_signal_1 = lpf_1.apply_filter(real_input_time_1, real_time_1)
    lpf_freq_1 = np.fft.fftshift(np.abs(fft(lpf_signal_1.filtered))) / (sim_freq_1*total_time_1)
    lpf_phase_1 = np.fft.fftshift(np.angle(fft(lpf_signal_1.filtered)))
    
    if show_wbf_signal:
        fig, axes = plt.subplots(1, 3, figsize=(8,4))  # 1 row, 3 columns
        axes[0].plot(real_time_1, lpf_signal_1.filtered)
        axes[0].set_title("Time (Ideal)")
        axes[0].set_xlim(-0.0002, 0.0002)
        axes[1].plot(real_freq_1, lpf_freq_1)
        axes[1].set_title("Frequency Magnitude (Ideal)")
        axes[1].set_xlim(-300000, 300000)
        axes[2].plot(real_freq_1, lpf_phase_1)
        axes[2].set_title("Frequency Phase (Ideal)")
        axes[2].set_xlim(-300000, 300000)
        # fig.suptitle("WBF Signals")
        fig.tight_layout()
        plt.show()

    lo_1 = LocalOscillator(config_file_path=Path(getenv('NYFR_CONF')))
    lo_signal_1 = lo_1.generate_signal(real_time_1)
    lo_pre_start_1 = lo_signal_1.pre_start_lo
    
    pulse_gen_1 = PulseGenerator(config_file_path=Path(getenv('NYFR_CONF')))
    pulse_signal_1 = pulse_gen_1.generate(lo_signal_1.lo, real_time_1, lo_pre_start_1)
    pulse_freq_1 = np.fft.fftshift(np.abs(fft(pulse_signal_1.pulses))) / (sim_freq_1*total_time_1)
    pulse_phase_1 = np.fft.fftshift(np.angle(fft(pulse_signal_1.pulses)))
    
    if show_LO_signals:
        fig, axes = plt.subplots(1, 3, figsize=(8,4))  # 1 row, 3 columns
        axes[0].plot(real_time_1, pulse_signal_1.pulses, label="Pulses")
        axes[0].plot(real_time_1, lo_signal_1.lo, label="LO")
        axes[0].set_title("Time (Ideal)")
        axes[0].set_xlim(-0.00002, 0.00002)
        axes[0].legend()
        axes[1].plot(real_freq_1, pulse_freq_1)
        axes[1].set_title("Frequency Magnitude (Ideal)")
        axes[1].set_xlim(-300000, 300000)
        axes[2].plot(real_freq_1, pulse_phase_1)
        axes[2].set_title("Frequency Phase (Ideal)")
        axes[2].set_xlim(-300000, 300000)
        # fig.suptitle("LO and Pulse Signals")
        fig.tight_layout()
        plt.show()
    
    mixed_1 = Mixer(config_file_path=Path(getenv('NYFR_CONF')))
    mixed_signal_1 = mixed_1.mix(lpf_signal_1.filtered, pulse_signal_1.pulses)
    mixed_freq_1 = np.fft.fftshift(np.abs(fft(mixed_signal_1.mixed))) / (sim_freq_1*total_time_1)
    mixed_phase_1 = np.fft.fftshift(np.angle(fft(mixed_signal_1.mixed)))
    
    mixed_2 = Mixer(config_file_path=Path(getenv('NYFR_CONF')))
    mixed_2_params = mixed_2.get_mixer_params()
    mixed_2_params["lo_leakage"] = 0.1
    mixed_2_params["rf_leakage"] = 0.08
    mixed_2_params["nonlinearity_coeff"] = 0.25
    mixed_2_params["noise_std"] = 0.01
    mixed_2.set_mixer_params(mixed_2_params)
    
    mixed_signal_2 = mixed_2.mix(lpf_signal_1.filtered, pulse_signal_1.pulses)
    mixed_freq_2 = np.fft.fftshift(np.abs(fft(mixed_signal_2.mixed))) / (sim_freq_1*total_time_1)
    mixed_phase_2 = np.fft.fftshift(np.angle(fft(mixed_signal_2.mixed)))
    
    fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 3 columns
    axes[0,0].plot(real_time_1, mixed_signal_1.mixed)
    axes[0,0].set_title("Time (Ideal)")
    axes[0,0].set_xlim(-0.0002, 0.0002)
    axes[0,1].plot(real_freq_1, mixed_freq_1)
    axes[0,1].set_title("Frequency Magnitude (Ideal)")
    axes[0,1].set_xlim(-300000, 300000)
    axes[0,2].plot(real_freq_1, mixed_phase_1)
    axes[0,2].set_title("Frequency Phase (Ideal)")
    axes[0,2].set_xlim(-300000, 300000)
    axes[1,0].plot(real_time_1, mixed_signal_2.mixed)
    axes[1,0].set_title("Time (Noisy)")
    axes[1,0].set_xlim(-0.0002, 0.0002)
    axes[1,1].plot(real_freq_1, mixed_freq_2)
    axes[1,1].set_title("Frequency Magnitude (Noisy)")
    axes[1,1].set_xlim(-300000, 300000)
    axes[1,2].plot(real_freq_1, mixed_phase_2)
    axes[1,2].set_title("Frequency Phase (Noisy)")
    axes[1,2].set_xlim(-300000, 300000)
    # fig.suptitle("Mixed Signals")
    fig.tight_layout()
    plt.show()