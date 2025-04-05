'''
@author: pete
'''
if __name__ == '__main__':
    import os
    import sys
    # Add the src directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from pathlib import Path
    from dotenv import load_dotenv
    from NYFR import NYFR
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    import numpy as np

    load_dotenv()

    nyfr = NYFR(file_path=Path(os.getenv('SYSTEM_CONF')))
    nyfr.initialize()
    analog_input, _ = nyfr.create_input_signal(file_path=Path(os.getenv('WAVE_PARAMS')))
    output = nyfr.simulate_system(input_signal=analog_input)
    input = nyfr.sample_signals(data=analog_input, sample_rate=nyfr.get_wb_nyquist_rate())
    dictionary = nyfr.create_dict()
    real_time = nyfr.get_real_time()
    real_bins = nyfr.get_real_frequncy_bins()
    time = nyfr.get_time()
    system_params = nyfr.get_system_params()
    real_input_frequency = fft(analog_input)
    input_frequency = fft(input)/nyfr.get_wb_nyquist_rate()
    output_frequency = fft(output)
    frequency = nyfr.get_frequncy_bins()
    sampled_frequency = nyfr.get_sampled_freq_bins()
    pseudo = np.linalg.pinv(dictionary)
    # pseudo = np.linalg.pinv(2*dictionary/system_params['adc_clock_freq'])
    model_input_guess = np.dot(pseudo, output)
    model_output_guess = fft(np.dot(dictionary, input_frequency))
    # SPGL_prediction = nyfr.recover_signal(2*dictionary/system_params['adc_clock_freq'], output)/(2*system_params['adc_clock_freq'])
    SPGL_prediction = (nyfr.recover_signal(dictionary, output))/system_params['adc_clock_freq']

    plt.figure()
    plt.subplot(8,1,1)
    plt.plot(real_time, analog_input)
    plt.subplot(8,1,2)
    plt.plot(real_bins, np.fft.fftshift(np.abs(real_input_frequency)))
    plt.xlim(-400,400)
    plt.subplot(8,1,3)
    plt.plot(time, input)
    plt.subplot(8,1,4)
    plt.plot(frequency, np.fft.fftshift(np.abs(input_frequency)))
    plt.subplot(8,1,5)
    plt.plot(frequency, np.fft.fftshift(np.abs(model_input_guess)))
    plt.subplot(8,1,6)
    plt.plot(frequency, np.fft.fftshift(np.abs(SPGL_prediction)))
    plt.subplot(8,1,7)
    plt.plot(sampled_frequency, np.fft.fftshift(np.abs(output_frequency)))
    plt.subplot(8,1,8)
    plt.plot(sampled_frequency, np.fft.fftshift(np.abs(model_output_guess)))
    plt.show()