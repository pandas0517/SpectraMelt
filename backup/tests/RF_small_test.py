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
        # test_11 = -5e-8*np.power(complex_tf,4) + 2e-12*np.power(complex_tf,3) - 5e-6*np.power(complex_tf,2) + 4e-9*complex_tf + 0.9988
        # test_11[ test_11 < 0 ] = 0
        # print(matching_tones)
        # test1 = np.nonzero(matching_tones)
        # for ii in test1:
        #     print(abs(matching_tones[ii]))
        plt.figure()
#         plt.suptitle("Local Oscillator Frequency {}: {}Hz | Modulation Frequency {}: {}Hz | Modulation Delta {}: {} | Input Signal Harmonics at: {}Hz, {}Hz, {}Hz, {}Hz\nLow-pass Maximally Flat 6th Order Butterworth Filter: Cuttoff Frequency {}Hz | ADC Sampling Frequency: {}Hz | Simulator Frequency: {}Hz"\
# .format("$f_{s1}$", LO_params['freq'], "$f_{mod}$", LO_params['phase_freq'], "$f_{∆}$", LO_params['phase_delta'],wave_params[1]['freq'],wave_params[2]['freq'],wave_params[3]['freq'],wave_params[4]['freq'],filter_params['cutoff_freq'], system_params['adc_clock_freq'], int(1/system_params['spacing'])))
#         plt.subplot(4,2,1)
        # plt.title("Time Domain:\nInput Signal")
        # plt.plot(t, y_filtered * downsample_train )
        # plt.plot(complex_tf,np.fft.fftshift(abs(fft(y_filtered))) / 100 , label="Mixed and Filtered Input", color='blue')
        # filt_time = ifft(filt_freq)
        # convolve = fft( y_filtered * downsample_train )
        # plt.plot(tf_sampled,np.fft.fftshift(abs(filt_down)), label="Mixed and Filtered Input", color='blue')
        # plt.plot(complex_tf,np.fft.fftshift(abs(convolve)), label="Mixed and Filtered Input", color='red')
        # plt.plot(complex_tf,test_11, label="Mixed and Filtered Input", color='red')
        # plt.plot(t, y_filtered )
        # plt.plot(t, x)
        # plt.ylabel("Amplitude")
        # plt.xlim(-50,50)
        # plt.subplot(4,2,2)
        # diff = 1 - ((np.fft.fftshift(abs(fft(y_filtered))) / 100) - np.fft.fftshift(abs(fft(y_filtered * downsample_train))))
        # diff_curve = np.zeros_like(diff)
        # filt_curve = np.fft.fftshift(abs(filt_freq))
        # curve_points = [
        #     np.where( complex_tf == -49 ),
        #     np.where( complex_tf == -44 ),
        #     np.where( complex_tf == -34 ),
        #     np.where( complex_tf == -14 ),
        #     np.where( complex_tf == 14 ),
        #     np.where( complex_tf == 34 ),
        #     np.where( complex_tf == 44 ),
        #     np.where( complex_tf == 49 )]
        # diff_curve_list = []
        # filt_curve_list = []
        # for points in curve_points:
        #     diff_curve[points] = diff[points]
        #     diff_curve_list.append(diff[points])
        #     filt_curve_list.append(filt_curve[points])
        # points_array = np.array( curve_points )
        # plt.title("Frequency Domain:\nMixed Input Signal - Low-pass Filtered")
        # plt.plot(complex_tf,np.fft.fftshift(abs(fft(convolve))) * y_mixed_max / 100, label="Mixed and Filtered Input", color='blue')
        # plt.plot(complex_tf,np.fft.fftshift(abs(fft(downsample_train))), label="Mixed and Filtered Input", color='green')
        # plt.plot(complex_tf,np.fft.fftshift(abs(fft(y_filtered))), label="Mixed and Filtered Input", color='blue')
        # plt.plot(complex_tf,np.fft.fftshift(abs(filt_freq)) * y_mixed_max, label="Low-pass Filter", color='red')   
        # plt.plot(complex_tf,np.fft.fftshift(abs(xf)))
        # plt.plot(complex_tf,np.fft.fftshift(abs(fft(zero_crossings))))
        plt.xlim(-50,50)
        plt.ylabel("Magnitude")
        plt.xlabel("Hertz")
        plt.legend()
        # plt.subplot(4,2,3)
        plt.title("CS Model Output: Original")
        # plt.plot(tf_sampled,np.fft.fftshift(abs(fft(y_sampled))))
        # plt.plot(t_sampled, y_sampled)
        # plt.ylabel("Amplitude")
        # plt.subplot(4,2,4)
        # plt.plot(complex_tf,np.fft.fftshift(abs(fft(y_filtered * downsample_train))))
        # plt.title("Low-pass Filtered Mixed Input Signal: Decimated in time")
        # plt.title("Low-pass Filtered Mixed Input Signal: Sampled")
        # plt.ylabel("Magnitude")
        # plt.xlim(-50,50)
        # plt.subplot(4,2,5)
        # plt.title("CS Model Output: Orientation and Filter Magnitude Correction")
        # plt.plot(t_sampled, np.real(test_model) )
        # plt.ylabel("Amplitude")
        # plt.subplot(4,2,6)
        plt.plot(tf_sampled,np.fft.fftshift(abs(fft(test_model))))
        # plt.title("CS Model Output: Orientation and Filter Magnitude Correction")
        # plt.ylabel("Magnitude")
        # plt.subplot(4,2,7)
        # plt.title("CS Reconstruction Algorithm: {}".format(system_params['recovery'].upper()))
        # plt.plot(t, np.real(ifft(coef + np.roll(np.flip(coef),1))))
        # plt.plot(t, np.real(ifft(coef)))
        # plt.ylabel("Amplitude")
        # plt.xlabel("Seconds")
        # plt.subplot(4,2,8)
        # testy = np.fft.fftshift(abs(coef + np.flip(coef)))
        # markers_on = []
        # for wave_param in wave_params:
        #     if ( wave_param['amp'] == 1 ) :
        #         x_neg = np.where( complex_tf == (-1 * wave_param['freq']) )[0][0]
        #         x_pos = np.where( complex_tf == (wave_param['freq']) )[0][0]
        #         markers_on.append(x_neg)
        #         markers_on.append(x_pos)
        # np.where( complex_tf == wave_params[1]['freq'] )
        # plt.plot(complex_tf,np.fft.fftshift(abs(coef)), '-D', markevery=markers_on)
        # plt.title("CS Reconstruction Algorithm: {}".format(system_params['recovery'].upper()))
        # plt.ylabel("Magnitude")
        # plt.xlabel("Hertz")
        # plt.xlim(-500,500)
        # plt.subplot(3,2,5)
        # plt.title("Low-pass Filtered Mixed Input Signal: Sampled")
        # plt.plot(t_sampled, np.real(test_model))
        # plt.ylabel("Amplitude")
        # plt.xlabel("Seconds")
        # plt.subplot(3,2,6)
        # plt.plot(tf_sampled,np.fft.fftshift(abs(fft(test_model))))
        # plt.title("Low-pass Filtered Mixed Input Signal: Sampled")
        # plt.ylabel("Magnitude")
        # plt.xlabel("Hertz")
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
        # plt.subplots_adjust(hspace=0.4)
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