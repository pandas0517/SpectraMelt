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
    from spectramelt.ADC import ADC
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft, fftshift
    import numpy as np
    
    show_input_signal = False

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
    
    adc_1 = ADC(config_file_path=Path(getenv('NYFR_CONF')))
    adc_1_params = adc_1.get_adc_params()
    adc_1_params["v_ref_range"] = (-2.5, 2.5)
    adc_1.set_adc_params(adc_1_params)
    adc_signal_1 = adc_1.analog_to_digital(real_input_time_1, real_time_1,
                                           return_sample_hold=True,
                                           return_conditioned=True)
    
    conditioned_input_1 = adc_signal_1.conditioned.signal
    conditioned_time_1 = adc_signal_1.conditioned.time
    sh_output_adc_1 = adc_signal_1.sample_hold.output_signal
    mid_times_adc_1 = adc_signal_1.quantized.mid_times
    quantized_adc_1 = adc_signal_1.quantized.quantized_values
    
    adc_2 = ADC(config_file_path=Path(getenv('NYFR_CONF')))
    adc_2_params = adc_2.get_adc_params()
    adc_2_params["v_ref_range"] = (-2.5, 2.5)
    adc_2_params["thermal_noise_std_dev"] = 0.5
    adc_2_params["jitter_std"] = 1.5e-7
    adc_2_params["hold_noise_std"] = 0.1
    adc_2.set_adc_params(adc_2_params)
    adc_signal_2 = adc_2.analog_to_digital(real_input_time_1, real_time_1,
                                           return_sample_hold=True,
                                           return_conditioned=True)
    
    sh_output_adc_2 = adc_signal_2.sample_hold.output_signal
    mid_times_adc_2 = adc_signal_2.quantized.mid_times
    quantized_adc_2 = adc_signal_2.quantized.quantized_values
    
    fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
    axes[0,0].plot(conditioned_time_1, sh_output_adc_1, label="Sample/Hold")
    axes[0,0].plot(conditioned_time_1, conditioned_input_1, label="Conditioned Input Signal")
    axes[0,0].set_title("Sample and Hold - Time (Ideal)")
    axes[0,0].set_xlim(-5e-5, 5e-5)
    axes[0,0].legend()
    axes[0,1].plot(conditioned_time_1, sh_output_adc_1, label="Sample/Hold")
    axes[0,1].step(mid_times_adc_1, quantized_adc_1, color='red', where='mid', label="Quantized")
    axes[0,1].set_title(f"{adc_2_params["num_bits"]}-bit quantizer (Ideal)")
    axes[0,1].set_xlim(-1.5e-5, 1.5e-5)
    axes[0,1].set_ylim(-0.7, 0.15)
    axes[0,1].legend()
    axes[1,0].plot(conditioned_time_1, sh_output_adc_2, label="Sample/Hold")
    axes[1,0].plot(conditioned_time_1, conditioned_input_1, label="Conditioned Input Signal")
    axes[1,0].set_title("Sample and Hold - Time (Noisy)")
    axes[1,0].set_xlim(-5e-5, 5e-5)
    axes[1,0].legend()
    axes[1,1].plot(conditioned_time_1, sh_output_adc_2, label="Sample/Hold")
    axes[1,1].step(mid_times_adc_2, quantized_adc_2, color='red', where='mid', label="Quantized")
    axes[1,1].set_title(f"{adc_2_params["num_bits"]}-bit quantizer (Noisy)")
    axes[1,1].set_xlim(-1.5e-5, 1.5e-5)
    axes[1,1].set_ylim(-1.5, 0.15)
    axes[1,1].legend()
    # fig.suptitle("ADC Signals")
    fig.tight_layout()
    plt.show()