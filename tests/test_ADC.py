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
    from Input_Signal import Input_Signal
    from ADC import ADC
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    import numpy as np

    input_signal_1 = Input_Signal(config_file_path=Path(os.getenv('SYSTEM_CONF')))
    real_time_1 = input_signal_1.get_analog_time()
    real_input_1 = input_signal_1.get_input_signal()
    real_freq_1 = input_signal_1.get_analog_frequency()
    total_time_1 = input_signal_1.get_analog_signal_params().get('total_time')
    sim_freq_1 = input_signal_1.get_time_params().get('sim_freq')
    real_input_freq_1 = np.fft.fftshift(np.abs(fft(real_input_1))) / (sim_freq_1*total_time_1)
    
    input_signal_2 = Input_Signal()
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
    
    adc_1 = ADC(real_input_1,real_time_1[0],config_file_path=Path(os.getenv('SYSTEM_CONF')))
    bits_adc_1 = adc_1.get_adc_params().get('num_bits')
    sh_output_adc_1 = adc_1.get_sh_signals().get('output_signal')
    mid_times_adc_1 = adc_1.get_quantizer_signals().get('mid_times')
    quantized_adc_1 = adc_1.get_quantizer_signals().get('quantized_values')
    adc_2 = ADC(real_input_2, real_time_2[0])
    bits_adc_2 = adc_2.get_adc_params().get('num_bits')
    sh_output_adc_2 = adc_2.get_sh_signals().get('output_signal')
    mid_times_adc_2 = adc_2.get_quantizer_signals().get('mid_times')
    quantized_adc_2 = adc_2.get_quantizer_signals().get('quantized_values')
    
    fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
    axes[0,0].plot(real_time_1, sh_output_adc_1)
    axes[0,0].set_title("Sample and Hold - Time (File)")
    axes[0,1].step(mid_times_adc_1, quantized_adc_1, color='green', where='mid')
    axes[0,1].set_title(f"{bits_adc_1}-bit quantizer (File)")
    axes[1,0].plot(real_time_2, sh_output_adc_2)
    axes[1,0].set_title("Sample and Hold - Time (Default)")
    axes[1,1].step(mid_times_adc_2, quantized_adc_2, color='green', where='mid')
    axes[1,1].set_title(f"{bits_adc_2}-bit quantizer (Default)")
    fig.suptitle("ADC Signals")
    fig.tight_layout()
    plt.show()