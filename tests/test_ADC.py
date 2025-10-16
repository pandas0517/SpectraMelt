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
    input_signal_1.create_analog()
    input_signal_1.create_input_signal()
    real_time_1 = input_signal_1.get_analog_signals().get('time')
    real_input_1 = input_signal_1.get_input_signal()
    real_freq_1 = input_signal_1.get_analog_signals().get('frequency')
    tot_time_1 = input_signal_1.get_analog_signals().get('total_time')
    real_input_freq_1 = np.fft.fftshift(np.abs(fft(real_input_1))) / (input_signal_1.get_time_params().get('sim_freq')*tot_time_1)
    input_signal_2 = Input_Signal()
    input_signal_2.create_analog()
    input_signal_2.create_input_signal()
    real_time_2 = input_signal_2.get_analog_signals().get('time')
    real_input_2 = input_signal_2.get_input_signal()
    real_freq_2 = input_signal_2.get_analog_signals().get('frequency')
    real_input_freq_2 = np.fft.fftshift(np.abs(fft(real_input_2))) / (input_signal_2.get_time_params().get('sim_freq'))
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(real_time_1, real_input_1, label="Simulated Analog - Time (File)")
    plt.subplot(2,2,2)
    plt.plot(real_freq_1, real_input_freq_1, label="Simulated Analog - Frequency (File)")
    plt.xlim(-5, 5)
    plt.subplot(2,2,3)
    plt.plot(real_time_2, real_input_2, label="Simulated Analog - Time (Default)")
    plt.subplot(2,2,4)
    plt.plot(real_freq_2, real_input_freq_2, label="Simulated Analog - Frequency (Default)")
    plt.xlim(-5, 5)
    plt.show()
    
    adc_1 = ADC(config_file_path=Path(os.getenv('SYSTEM_CONF')))
    adc_1.analog_to_digital(real_input_1,real_time_1[0])
    adc_2 = ADC()
    adc_2.analog_to_digital(real_input_2, real_time_2[0])
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(real_time_1, adc_1.get_sh_signals().get('output_signal'), label="Sample and Hold - Time (File)")
    plt.subplot(2,2,2)
    plt.step(adc_1.get_quantizer_signals().get('mid_times'),
             adc_1.get_quantizer_signals().get('quantized_values'),
             color='green', where='mid')
    plt.subplot(2,2,3)
    plt.plot(real_time_2, adc_2.get_sh_signals().get('output_signal'), label="Sample and Hold - Time (Default)")
    plt.subplot(2,2,4)
    plt.step(adc_2.get_quantizer_signals().get('mid_times'),
             adc_2.get_quantizer_signals().get('quantized_values'),
             color='green', where='mid')
    plt.show()