'''
Created on May 17, 2023

@author: pete
'''

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # from signals import generate_LO
    # from signals import generate_wavelet_train
    # from signals import random_demodulation
    # from signals import filter_compressed_signal
    # from signals import downsample
    # from signals import create_nyfr_dict
    # from sklearn.linear_model import OrthogonalMatchingPursuit
    # from sklearn.linear_model import orthogonal_mp
    import numpy as np
    # from signals import multi_tone_sine_wave
    from signals import simulate_system
    import multiprocessing
    # import scipy.fft as fft
    from math import pi
    from queue import Queue
    
    system_params = {
        'dictionary': 'real',
        'filter': 'butter',
        'sampled_LO': 'y',
        'wavelets': 'n',
        'rand_demod': 'n',
        'rd_clock_freq': 4,
        'adc_clock_freq': 20,
        'start': -3,
        'stop': 3,
        'spacing': 0.001,
        'recovery': 'spgl1'}
    #Input signal parameters
    wave_params = [
        {'amp': 1,
         'freq': 5,
         'phase': 0},
        {'amp': 1,
         'freq': 15,
         'phase': 0},
        {'amp': 1,
         'freq': 25,
         'phase': 0},
        {'amp': 1,
         'freq': 35,
         'phase': 0},
        {'amp': 1,
         'freq': 45,
         'phase': 0}]
    #Phase modulated local oscillator (NYFR) parameters
    LO_params = {
        'amp':1,
        'freq':20,
        'phase':0,
        'phase_delta': 0.001,
        'phase_freq': 0.001,
        'phase_offset': 0}
    # Gabor atoms with Gaussian window parameters
    # f_c = center frequencies of atoms
    # width = Gaussian Standard Deviation
    # large width increases frequency resolution while reducing time resolution
    # Filtering angle of fractional Fourier domain 
    psi_params = [
        {
            'amp': 0.5,
            'f_c': 10,
            'width': 0.1,
            'shift': 0,
            'angle': pi/4
        },
        {
            'amp': 0.5,
            'f_c': 20,
            'width': 0.2,
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
    #RF input and time vector creation
    # x, t, num_tones = multi_tone_sine_wave(system_params, wave_params, filter_params)
    # xf = fft.fft(x)

    best_LO_params = dict(LO_params)
    # non_zero_matches = 0
    # best_recovery = 0
    # matching_tones = np.zeros_like(xf)
    # update_best = 'y'
    eps = 1+0*1j
    # max_adc_freq = 50 + system_params['adc_clock_freq']
    max_mod_freq = 0.1 + LO_params['phase_freq']
    max_mod_delta = 0.1 + LO_params['phase_delta']
    max_procs = 50
    # num_lo_freq = int(( max_lo_freq - LO_params['freq'] ) / 2 )
    # num_adc_freq = int( max_adc_freq - system_params['adc_clock_freq'])
    num_mod_freq = 100
    num_mod_delta = 100
    manager_queue = multiprocessing.Queue()
    # max_procs = multiprocessing.cpu_count()
    # adc_freq_range = np.linspace(system_params['adc_clock_freq'], max_adc_freq, num_adc_freq, endpoint=False, dtype='int')
    phase_freq_range = np.linspace(LO_params['phase_freq'], max_mod_freq, num_mod_freq, endpoint=False)
    # phase_freq_range_split = np.split(phase_freq_range, 2)
    phase_delta_range_total = np.linspace(LO_params['phase_delta'], max_mod_delta, num_mod_delta, endpoint=False)
    # range_split = int(num_mod_delta/num_cores)
    phase_delta_range_split = np.split(phase_delta_range_total, 2)
    jobs = Queue()
    f = open("best_recovery.log", "w")
    f.write("Start of parameter search\n")
    f.write("\n")
    f.close
    # for adc_freq in adc_freq_range:
    #     system_params['adc_clock_freq'] = adc_freq
    #     min_lo_freq = 1
    #     max_lo_freq = 50 + min_lo_freq
    #     num_lo_freq = int ( max_lo_freq - min_lo_freq )
    #     lo_freq_range = np.linspace(min_lo_freq, max_lo_freq, num_lo_freq, endpoint=False, dtype='int')
    #     for lo_freq in lo_freq_range:
    #         LO_params['freq'] = lo_freq   
    # for phase_freq_range in phase_freq_range_split:
    #     for idx_freq, phase_freq in enumerate(phase_freq_range, 1):
            # LO_params['phase_freq'] = phase_freq
            # for phase_delta_range in phase_delta_range_split: 
            # for idx_delta, phase_delta in enumerate(phase_delta_range, 1):
    for phase_freq in phase_freq_range:
        LO_params['phase_freq'] = phase_freq
        for phase_delta_range in phase_delta_range_split:
            for phase_delta in phase_delta_range:
                LO_params['phase_delta'] = phase_delta
                p = multiprocessing.Process(target=simulate_system, args=(wave_params, eps, LO_params, system_params, psi_params, filter_params, manager_queue))
                jobs.put(p)
                p.start()
        # if ( idx_freq * idx_delta >= max_procs ):
            while not jobs.empty():
                proc = jobs.get()
                proc.join()
            
            while not manager_queue.empty():
                return_val = manager_queue.get()
                non_zero_matches = return_val[0]
                current_params = return_val[1]
                num_tones = return_val[2] 
                # if ( update_best == 'n' ):
                #     if ( non_zero_matches > best_recovery):
                #         update_best = 'y'
                # if ( update_best == 'y' ):
                if ( non_zero_matches == num_tones ):
                    # update_best = 'n'
                    # best_recovery = non_zero_matches
                    # best_LO_params = dict(current_params)
                    f = open("best_recovery.log", "a")
                    f.write("Number of matches: " + str(non_zero_matches) +"\n")
                    # f.write(str(best_LO_params) + "\n")
                    f.write(str(current_params) + "\n")
                    f.write("\n")
                    f.close()
                # # print(a,b,sep=", ")
                # # Create Phase Modulated Local Oscillator
                # LO_mod, rising_zero_crossings, LO = generate_LO(t, LO_params, system_params)
                # if ( system_params['wavelets'] == 'y' ):
                #     #Create wavelet train
                #     first_modulation = generate_wavelet_train(psi_params, rising_zero_crossings, t)
                # else:
                #     first_modulation = rising_zero_crossings
                # #Mix wavelet train with input signal
                # y_mixed = np.copy(x*first_modulation)
                # #Random Demodulation
                # y_demod = random_demodulation(y_mixed, system_params)
                # #Filter demodulated signal
                # y_filtered = filter_compressed_signal(y_demod, t, filter_params, system_params)
                # #Downsample with ADC
                # y_sampled, LO_mod_sampled, sample_train = downsample(y_filtered, LO_mod, system_params)
                # y_sampled_norm = np.linalg.norm(y_sampled)
                # #Create NYFR Dictionary for signal reconstruction
                # dictionary = create_nyfr_dict(t, LO_mod_sampled, system_params)
                # # print(dictionary.shape)
                # # print(y_sampled.shape)
    #             #Recover signal
    #             coef_real = orthogonal_mp(np.real(dictionary),y_sampled/y_sampled_norm,n_nonzero_coefs=num_tones)
    #             coef_imag = orthogonal_mp(np.imag(dictionary),y_sampled/y_sampled_norm,n_nonzero_coefs=num_tones)
    #             coef = coef_real + coef_imag*1j
    #             matching_tones = np.multiply(np.abs(xf),np.abs(coef))
    #             matching_tones[ np.abs(matching_tones) < eps ] = 0
    # print(best_LO_params)
    # print(np.count_nonzero(np.multiply(np.abs(xf),np.abs(coef))))
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(complex_tf,np.abs(xf))
    # plt.subplot(2,1,2)
    # plt.plot(complex_tf,np.multiply(np.abs(xf),np.abs(coef)))
    # plt.show()
    # plt.title("Recovered signal from noise-free measurements")
    # plt.stem(idx_r, coef[idx_r])

    # plt.subplot(2,2,1)
    # plt.plot(tf_imag,np.real(xf_input))
    # plt.subplot(2,2,2)
    # plt.plot(complex_tf,np.imag(xf_input))
    # plt.subplot(2,2,3)
    # plt.plot(complex_tf,coef_real)
    # plt.subplot(2,2,4)
    # plt.plot(complex_tf,coef_imag)
    # plt.plot(t,x)
    # plt.plot(t,rising_zero_crossings)
    # plt.plot(complex_tf,np.fft.fftshift(abs(xf)))
    # plt.plot(complex_tf,np.fft.fftshift(abs(coef)))
    # plt.subplot(3,1,3)
    # plt.plot(t,rising_zero_crossings)
    # plt.plot(t,sample_train)    
    # plt.plot(tf_imag,np.fft.fftshift(abs(coef)))
    # plt.plot(tf_imag,abs(coef))
    # plt.plot(tf_imag,coef_real)
    # plt.plot(tf_imag,omp_abs.coef_)
    # plt.plot(t,np.real(wavelet[20]))
    # plt.subplot(2,2,1)
    # plt.plot(complex_tf,xf)
    # plt.subplot(2,2,2)
    # plt.plot(complex_tf,test_xf1)
    # plt.subplot(2,2,3)
    # plt.plot(complex_tf,test_xf3)
    # plt.subplot(2,2,4)
    # plt.plot(complex_tf,test_xf2)  