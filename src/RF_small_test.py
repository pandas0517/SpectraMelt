'''
Created on Jul 26, 2023

@author: pete
'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from queue import Queue
    from signals import simulate_system
    from scipy.fft import fft
    from math import pi
    
    system_params = {
        'dictionary': 'real',
        'filter': 'butter',
        'sampled_LO': 'y',
        'wavelets': 'y',
        'rand_demod': 'n',
        'rd_clock_freq': 4,
        'adc_clock_freq': 100,
        'wave_freq': [2,20],
        'start': -3,
        'stop': 3,
        'spacing': 0.001,
        'recovery': 'spgl1'}
    #Input signal parameters
    wave_params = [
        {'amp': 0,
         'freq': 15,
         'phase': 0},
        {'amp': 1,
         'freq': 215,
         'phase': 0},
        {'amp': 1,
         'freq': 345,
         'phase': 0},
        {'amp': 0,
         'freq': 75,
         'phase': 0},
        {'amp': 0,
         'freq': 395,
         'phase': 0}]
    #Phase modulated local oscillator (NYFR) parameters
    LO_params = {
        'amp':1,
        'freq':100,
        'phase':0,
        'phase_delta': 0.01,
        'phase_freq': 0.1,
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
        # print(matching_tones)
        # test1 = np.nonzero(matching_tones)
        # for ii in test1:
        #     print(abs(matching_tones[ii]))
        plt.figure()
        plt.suptitle("Sample Train Generated from Local Oscillator Rising-Zero Crossings\nLO Frequency: {}Hz | Modulation Frequency: {}Hz | Modulation Delta: {}"\
.format(LO_params['freq'], LO_params['phase_freq'], LO_params['phase_delta']))
        # plt.subplot(4,1,1)
        # plt.title("Multi-tone Real-Valued Input Signal: Time Domain | Tones at {}Hz and {}Hz".format(wave_params[1]['freq'],wave_params[2]['freq']))
        # plt.xlim(-0.1,0.1)
        # plt.plot(t,x)
        plt.subplot(1,2,1)
        plt.title("Time Domain")
        plt.plot(t,zero_crossings, label="Zero Crossings", color="blue")
        plt.plot(t, LO, label="Modulated LO", color="orange")
        plt.legend()
        plt.xlim(-0.05,0.05)
        plt.ylabel("Amplitude")
        plt.xlabel("Seconds")
        plt.subplot(1,2,2)
        plt.title("Frequency Domain")
        plt.plot(complex_tf,np.fft.fftshift(abs(fft(zero_crossings))))
        plt.ylabel("Magnitude")
        plt.xlabel("Hertz")       
        # plt.subplot(4,1,1)
        # plt.plot(complex_tf,np.fft.fftshift(abs(xf)))
        # plt.title("Multi-tone Real-Valued Input Signal: Frequency Domain | Tones at {}Hz and {}Hz".format(wave_params[1]['freq'],wave_params[2]['freq']))
        # plt.subplot(4,1,2)
        # plt.title("ADC frequency: " + str(adc_freq) + " | LO frequency: " + str(LO_freq))
        # plt.title("LO frequency (f_s1): " + str(LO_freq) + "Hz | Modulation Frequency (f_mod): " + \
        #           str(LO_params['phase_freq']) + "Hz | Modulation Deviation (f_delta): " + \
        #           str(LO_params['phase_delta']) + "\nLO Mixing Frequency Domain Comb Function")
        # plt.title("LO Mixing Frequency Domain Comb Function | LO frequency (f_s1): " + str(LO_freq) + "Hz | Modulation Frequency (f_mod): " + \
        #           str(LO_params['phase_freq']) + "Hz | Modulation Deviation (f_delta): " + \
        #           str(LO_params['phase_delta']))
        # plt.plot(complex_tf,np.fft.fftshift(abs(np.fft.fft(zero_crossings))))
        # plt.plot(complex_tf,np.fft.fftshift(abs(xf)))
        # plt.xlim(-0.1,0.1)
        # plt.plot(t,LO)
        # plt.subplot(5,1,1)
        # plt.title("Single Wavelet Zero Time Shift")
        # plt.plot(t, single_wavelet)
        # plt.plot(complex_tf,np.fft.fftshift(abs(np.fft.fft(single_wavelet))))
        # plt.plot(tf_sampled,np.fft.fftshift(abs(test_model)))
        # plt.subplot(5,1,2)
        # plt.title("Wavelet Train")
        # plt.plot(t, wavelet_train)
        # plt.plot(tf_sampled,np.fft.fftshift(abs(y_sampled)))
        # plt.plot(complex_tf,np.fft.fftshift(abs(np.fft.fft(wavelet_train))))       
        # plt.subplot(5,1,3)
        # plt.title("Mixing Action Between Input Signal and LO: Frequency Domain")
        # plt.title("Mixing Action Between Input Signal and LO | {}Hz Low Pass Filter Overlay | Frequency Domain".format(filter_params['cutoff_freq']))
        # plt.plot(complex_tf,np.fft.fftshift(abs(y_mixed)))
        # plt.plot(t,zero_crossings)
        # plt.xlim(-0.1,0.1)
        # plt.subplot(6,1,3)
        # plt.plot(complex_tf,np.fft.fftshift(abs(coef)))
        # plt.plot(complex_tf,np.fft.fftshift(abs(y_mixed)), complex_tf,np.fft.fftshift(abs(y_filtered).max()*abs(filt_freq)))
        # plt.plot(complex_tf,np.fft.fftshift(abs(filt_freq)))
        # plt.subplot(5,1,4)
        # plt.title("LO Mixing Frequency Domain Comb Function | LO frequency (f_s1): " + str(LO_freq) + "Hz | Modulation Frequency (f_mod): " + \
        #         str(LO_params['phase_freq']) + "Hz | Modulation Deviation (f_delta): " + \
        #         str(LO_params['phase_delta']) + "\nFiltered Input Signal using Maximally Flat {}th Order Butterworth Low-pass FIR Filter".format(filter_params['order']))
        # plt.title("Filtered Input Signal using Maximally Flat {}th Order Butterworth Low-pass FIR Filter".format(filter_params['order']))
        # test1 = abs(filt_freq[np.argmax(abs(y_filtered))])
        # test2 = abs(y_filtered).max() / test1
        # plt.plot(complex_tf,np.fft.fftshift(abs(y_filtered)), complex_tf,np.fft.fftshift(test2*abs(filt_freq)))
        # plt.subplot(5,1,5)
        # plt.title("Sampled Filtered Input Signal: Frequency Domain | ADC Sample Rate: {}Hz".format(system_params['adc_clock_freq']))
        # plt.xticks(np.arange(min(tf_sampled), max(tf_sampled)+1, 5))
        # plt.plot(tf_sampled,np.fft.fftshift(abs(y_sampled)))
        # plt.plot(tf_sampled,np.fft.fftshift(abs(y_sampled)), tf_sampled, np.fft.fftshift(abs(y_sampled).max()*abs(filt_freq_down)))
        # plt.subplot(4,1,3)
        # plt.xticks(np.arange(min(tf_sampled), max(tf_sampled)+1, 5))
        # plt.title("Improved Output From CS Model With Low-Pass Filter Compensation")
        # plt.plot(tf_sampled,np.fft.fftshift(abs(test_model)))
        # plt.plot(tf_sampled,np.fft.fftshift(abs(test_model)), tf_sampled, np.fft.fftshift(abs(y_sampled).max()*abs(filt_freq_down)))
        # plt.subplot(6,1,6)
        # plt.subplot(5,1,5)
        # plt.title("Reconstructed Input Signal Using SPGL-1: Frequency Domain")
        # plt.plot(complex_tf,np.fft.fftshift(abs(coef)))
        # plt.plot(tf_sampled,np.fft.fftshift(abs(coef)))
        # plt.plot(complex_tf, abs(coef))
        # plt.tight_layout()
        plt.show()   
    # plt.subplot(1,2,1)
    # plt.title("Pulse Train From Non-Modulated LO Rising Zero Crossings")
    # plt.plot(t,rising_zero_crossings)
    #plt.plot(tf_imag,np.abs(xf))
    # plt.plot(tf_imag,np.fft.fftshift(abs(LO_f)))
    # plt.plot(tf_imag,xf)
    # plt.subplot(1,2,2)
    # plt.title("Pseudorandom Bit Sequence")
    # plt.plot(tf_imag,np.fft.fftshift(abs(xf)/xf.size))
    # plt.plot(t, PRBS)
    # plt.plot(tf_imag,coef_real)
    # plt.plot(tf_imag,omp_abs.coef_)
    #plt.plot(t,y_mixed)
    # plt.plot(t,np.real(wavelet[20]))
    # plt.show()
    #Recover signal
    # y_sampled_rnorm = np.linalg.norm(np.real(y_sampled))
    # y_sampled_inorm = np.linalg.norm(np.imag(y_sampled))
    # y_sampled_abs_norm = np.linalg.norm(np.abs(y_sampled))
    # y_sampled_angle_norm = np.linalg.norm(np.angle(y_sampled))
    # omp_real = OrthogonalMatchingPursuit(n_nonzero_coefs=num_tones)
    # omp_imag = OrthogonalMatchingPursuit(n_nonzero_coefs=num_tones)
    # omp_abs = OrthogonalMatchingPursuit()
    # omp_angle = OrthogonalMatchingPursuit()
    # omp_real.fit(np.real(dictionary), np.real(y_sampled)/y_sampled_rnorm)
    # omp_imag.fit(np.imag(dictionary), np.imag(y_sampled)/y_sampled_inorm)
    # omp_abs.fit(np.abs(dictionary), np.abs(y_sampled)/y_sampled_abs_norm)
    # omp_angle.fit(np.angle(dictionary), np.angle(y_sampled)/y_sampled_angle_norm)
    # coef_real = omp_real.coef_
    # coef_imag = omp_imag.coef_
    # coef_real = orthogonal_mp(np.real(dictionary),np.real(y_sampled)/y_sampled_rnorm,n_nonzero_coefs=num_tones)
    # coef_imag = orthogonal_mp(np.imag(dictionary),np.imag(y_sampled)/y_sampled_inorm,n_nonzero_coefs=num_tones)
    # coef_real = orthogonal_mp(np.abs(dictionary),np.abs(y_sampled),n_nonzero_coefs=num_tones)
    # coef_imag = orthogonal_mp(np.angle(dictionary),np.angle(y_sampled),n_nonzero_coefs=num_tones)
    
    # complex_tf = np.linspace(-1/(2*system_params['spacing']), 1/(2*system_params['spacing']), int(t.size))
    # tf = np.linspace(0.0, 1/(2*system_params['spacing']), int(t.size/2))
    # xf = fft.fft(x)
    # plt.figure()
    # plt.title("Recovered signal from noise-free measurements")
    # plt.stem(idx_r, coef[idx_r])