'''
Created on Aug 9, 2023

@author: pete
'''

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from signals import multi_tone_sine_wave
    from signals import generate_LO
    from signals import downsample
    from signals import create_nyfr_dict
    from math import pi
    
    system_params = {
        'dictionary': 'real',
        'filter': 'butter',
        'wavelets': 'n',
        'rand_demod': 'n',
        'rd_clock_freq': 4,
        'adc_clock_freq': 20,
        'start': -3,
        'stop': 3,
        'spacing': 0.001}
    #Input signal parameters
    wave_params = [
        {'amp': 1,
         'freq': 10,
         'phase': 0},
        {'amp': 1,
         'freq': 20,
         'phase': 0},
        {'amp': 1,
         'freq': 30,
         'phase': 0},
        {'amp': 1,
         'freq': 40,
         'phase': 0},
        {'amp': 1,
         'freq': 50,
         'phase': 0}]
    #Phase modulated local oscillator (NYFR) parameters
    LO_params = {
        'amp':1,
        'freq':20,
        'phase':0,
        'phase_delta': 1,
        'phase_freq': 0.1,
        'phase_offset': 0}
    # Filter parameters
    filter_params = {
        'order': 6,
        'angle': pi/2,
        'cutoff_freq': 100,
        'window_size': 150}
    #RF input and time vector creation
    x, t, num_tones = multi_tone_sine_wave(system_params, wave_params, filter_params)
    complex_tf = np.linspace(-1/(2*system_params['spacing']), 1/(2*system_params['spacing']), int(t.size))
    LO_mod, rising_zero_crossings, LO = generate_LO(t, LO_params, system_params)
    #Downsample with ADC
    y_sampled, LO_mod_sampled, sample_train = downsample(x, LO_mod, system_params)
    #Create NYFR Dictionary for signal reconstruction
    R, S, PSI = create_nyfr_dict(t, LO_mod_sampled, system_params)
    R_row, R_col = R.shape
    S_row, S_col = S.shape
    PSI_row, PSI_col = PSI.shape
    count_col = 0
    # f = open("dictionary_test.txt", "w")
    for i in range(0, R_col, R_row):
        R_sub = np.copy(R[0:,i:i+R_row])
        file_name = "./dict_tests_R/test{}.csv".format(count_col)
        np.savetxt(file_name,R_sub,delimiter = ',',fmt='%i')
        count_col += 1
    count_col = 0
    count_row = 0
    for i in range(0, S_row, R_row):
        for k in range(0, S_col, R_row):
            R_sub = np.copy(S[i:i+R_row,k:k+R_row])
            file_name = "./dict_tests_S/test{}_{}.csv".format(count_row,count_col)
            np.savetxt(file_name,R_sub,delimiter = ',',fmt='%1.1f')
            count_col += 1
        count_col = 0
        count_row += 1