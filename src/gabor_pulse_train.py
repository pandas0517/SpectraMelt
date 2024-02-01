'''
Created on Oct 24, 2023

@author: pete
'''

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from queue import Queue
    from signals import multi_tone_sine_wave
    from signals import generate_LO
    from signals import generate_wavelet_train
    from math import pi
    from scipy.fft import fft
    
    system_params = {
        'dictionary': 'real',
        'filter': 'butter',
        'sampled_LO': 'y',
        'wavelets': 'y',
        'rand_demod': 'n',
        'rd_clock_freq': 4,
        'adc_clock_freq': 20,
        'start': -3,
        'stop': 3,
        'spacing': 0.001,
        'recovery': 'spg_bp'}
    #Input signal parameters
    wave_params = [
        {'amp': 1,
         'freq': 15,
         'phase': 0},
        {'amp': 1,
         'freq': 35,
         'phase': 0},
        {'amp': 1,
         'freq': 55,
         'phase': 0},
        {'amp': 1,
         'freq': 75,
         'phase': 0},
        {'amp': 1,
         'freq': 95,
         'phase': 0}]
    #Phase modulated local oscillator (NYFR) parameters
    LO_params = {
        'amp':1,
        'freq':20,
        'phase':0,
        'phase_delta': 0.002,
        'phase_freq': 0.02,
        'phase_offset': 0}
    # Gabor atoms with Gaussian window parameters
    # f_c = center frequencies of atoms
    # width = Gaussian Standard Deviation
    # large width increases frequency resolution while reducing time resolution
    # Filtering angle of fractional Fourier domain 
    psi_params = [
        {
            'amp': 0.5,
            'f_c': 5,
            'width': 0.01,
            'shift': 0,
            'angle': pi/4
        }
    ]
    # Filter parameters
    filter_params = {
        'order': 6,
        'angle': pi/2,
        'cutoff_freq': 10,
        'window_size': 150}
    
    x, t, num_tones = multi_tone_sine_wave(system_params, wave_params, filter_params)
    xf = fft(x)
    complex_tf = np.linspace(-1/(2*system_params['spacing']), 1/(2*system_params['spacing']), int(t.size))
    # Create Phase Modulated Local Oscillator
    LO_mod, rising_zero_crossings, LO = generate_LO(t, LO_params, system_params)
    if ( system_params['wavelets'] == 'y' ):
        #Create wavelet train
        first_modulation = generate_wavelet_train(system_params, psi_params, rising_zero_crossings, t)
    else:
        first_modulation = rising_zero_crossings
    #Mix wavelet train with input signal
    y_mixed = np.copy(x*first_modulation)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot( t,rising_zero_crossings )
    plt.subplot(2,2,2)
    plt.plot( t, abs(first_modulation) )
    plt.subplot(2,2,3)
    plt.plot( complex_tf,np.fft.fftshift(abs(y_mixed)) )
    plt.show()
    