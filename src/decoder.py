'''
Created on Jul 26, 2023

@author: pete
'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from queue import Queue
    from signals import simulate_system
    from scipy.fft import fft, ifft
    from math import pi

    system_params = {
        'dictionary': 'real',
        'filter': 'butter',
        'sampled_LO': 'y',
        'wavelets': 'n',
        'rand_demod': 'n',
        'rd_clock_freq': 4,
        'adc_clock_freq': 100,
        'wave_freq': [10,20],
        'start': -0.5,
        'stop': 0.5,
        'spacing': 0.0001,
        'recovery': 'spgl1'}
    #Input signal parameters
    wave_params = [
        {'amp': 0,
         'freq': 15,
         'phase': 0},
        {'amp': 0,
         'freq': 70,
         'phase': 0},
        {'amp': 0,
         'freq': 110,
         'phase': 0},
        {'amp': 1,
         'freq': 215,
         'phase': 0},
        {'amp': 1,
         'freq': 345,
         'phase': 0}]
    #Phase modulated local oscillator (NYFR) parameters
    LO_params = {
        'amp':1,
        'freq':100,
        'phase':0,
        'phase_delta': 0.4,
        'phase_freq': 0.5,
        'phase_offset': 0}
    # Gabor atoms with Gaussian window parameters
    # f_c = center frequencies of atoms
    # width = Gaussian Standard Deviation
    # large width increases frequency resolution while reducing time resolution
    # Filtering angle of fractional Fourier domain 
    psi_params = [
        {
            'amp': 0.5,
            'f_c': 20,
            'width': 0.001,
            'shift': 0,
            'angle': pi/4
        }
    ]
    # Filter parameters
    filter_params = {
        'order': 6,
        'angle': pi/2,
        'cutoff_freq': 50,
        'window_size': 150}
    # eps = 1+0*1j
    eps = 15
    max_lo_freq = 4 + LO_params['freq']
    # num_lo_freq = int(( max_lo_freq - LO_params['freq'] ) / 2 )
    num_lo_freq = int( max_lo_freq - LO_params['freq'] )
    # lo_freq_range = np.linspace(LO_params['freq'], max_lo_freq, num_lo_freq, endpoint=False)
    lo_freq_range = [LO_params['freq']]
    manager_queue = Queue()
    for lo_freq in lo_freq_range:
        LO_params['freq'] = lo_freq
        #system_params['adc_clock_freq'] = lo_freq
        # print(lo_freq) 
        simulate_system(wave_params, eps, LO_params, system_params, psi_params, filter_params, manager_queue)
    # plt.title("Recovered signal from noise-free measurements")
    # plt.stem(idx_r, coef[idx_r])
    while not manager_queue.empty():
        signals = manager_queue.get()
        LO_freq = signals[0]
        adc_freq = signals[1]
        complex_tf = signals[2]
        xf = signals[3]
        coef = signals[4]
        matching_tones = signals[5]
        zero_crossings = signals[6]
        LO_mix = signals[7]
        y_filtered = signals[8]
        t = signals[9]
        t_sampled = signals[10]
        tf_sampled = signals[11]
        y_sampled = signals[12]
        test_model = signals[13]
        filt_freq = signals[14]
        filt_freq_down = signals[15]
        y_mixed = signals[16]
        LO = signals[17]
        x = signals[18]
        wavelet_train = signals[19]
        single_wavelet = signals[20]
        filt_down = signals[21]
        downsample_train = signals[22]
        y_start = np.where( complex_tf == -1000 )[0][0]
        y_end = np.where( complex_tf == 1000 )[0][0]
        test_y1 = np.copy(np.fft.fftshift(abs(y_mixed)))
        y_mixed_max = test_y1[y_start:y_end].max()