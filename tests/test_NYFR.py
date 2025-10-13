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
    from utility import load_settings
    from scipy.fftpack import fft
    import numpy as np

    load_dotenv()

    nyfr = NYFR(file_path=Path(os.getenv('SYSTEM_CONF')))
    nyfr.initialize()
    system_config = load_settings(Path(os.getenv('WAVE_PARAMS')))
    wave_params = system_config['wave_params']
    num_tones = 0
    for params in wave_params:
        if params['amp'] > 0:
            num_tones += 2
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
    # SPGL_prediction = (nyfr.recover_signal(dictionary, output))/system_params['adc_clock_freq']
    # IHT_prediction = nyfr.recover_signal(dictionary, output, num_tones=num_tones, sigma=0.001)
    plt.figure()
    # plt.subplot(2,1,1)
    plt.plot(real_time, analog_input, label="Simulated Analog")
    plt.xlim(-0.05, 0.05)
    # plt.subplot(8,1,2)
    # plt.plot(real_bins, np.fft.fftshift(np.abs(real_input_frequency)))
    # plt.xlim(-400,400)
    # plt.subplot(2,1,2)
    plt.plot(time, input, 'r--', label="Wideband Filtered")
    plt.xlim(-0.05, 0.05)
    plt.ylabel("Amplitude")
    plt.xlabel("Time(s)")
    plt.title("SpectraMelt Input Signals")
    plt.legend()
    # plt.subplot(8,1,4)
    # plt.plot(frequency, np.fft.fftshift(np.abs(input_frequency)))
    # plt.subplot(8,1,5)
    # plt.plot(frequency, np.fft.fftshift(np.abs(model_input_guess)))
    # plt.subplot(8,1,6)
    # plt.plot(frequency, np.fft.fftshift(np.abs(IHT_prediction)))
    # plt.plot(frequency, np.fft.fftshift(np.abs(SPGL_prediction)))
    # plt.subplot(8,1,7)
    # plt.plot(sampled_frequency, np.fft.fftshift(np.abs(output_frequency)))
    # plt.subplot(8,1,8)
    # plt.plot(sampled_frequency, np.fft.fftshift(np.abs(model_output_guess)))
    plt.show()