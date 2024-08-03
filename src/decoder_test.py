'''
Created on Jul 10, 2024

@author: pete
'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from queue import Queue
    from signals import simulate_system
    from scipy.fft import fft, ifft
    from math import pi
    from decoder import create_test_set, create_decoder
    # import tensorflow as tf

    system_params = {
        'dictionary': 'real',
        'filter': 'butter',
        'sampled_LO': 'y',
        'wavelets': 'n',
        'rand_demod': 'n',
        'rd_clock_freq': 4,
        'adc_clock_freq': 100,
        'wave_freq': [10,20],
        'start': -2,
        'stop': 2,
        'spacing': 0.001,
        'wbf_cut_mod': 4,
        'recovery': 'decode'}
    #Input signal parameters
    wave_params = [
        {'amp': 0,
         'freq': 15,
         'phase': 0},
        {'amp': 0.1,
         'freq': 80,
         'phase': 0},
        {'amp': 0.1,
         'freq': 115,
         'phase': 0},
        {'amp': 0.1,
         'freq': 180,
         'phase': 0},
        {'amp': 0.1,
         'freq': 340,
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
    # tttt_1d = np.arange(45)
    # tttt_3d = tttt_1d.reshape((3,3,5))
    # tttt_3d = tttt_3d[:,:,np.random.permutation(tttt_3d.shape[2])]
    # for n in range(5):
    #     print(tttt_3d[:,:,n])
    #     pass
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
        dictionary = signals[23]
        # y_start = np.where( complex_tf == -1000 )[0][0]
        # y_end = np.where( complex_tf == 1000 )[0][0]
        # test_y1 = np.copy(np.fft.fftshift(abs(y_mixed)))
        # y_mixed_max = test_y1[y_start:y_end].max()
    # test_set, encoded_test_set, dic_test_set = create_test_set(dictionary, t, system_params, wave_params, filter_params, LO_params)
    # decoder = create_decoder(dictionary, test_set, dic_test_set)
    # decoder_real = tf.keras.models.load_model('decoder_real.keras')
    # decoder_imag = tf.keras.models.load_model('decoder_imag.keras')
    # test_set_size = (test_set.shape)[0]
    # for i in range(0, test_set_size):
    #     plt.figure()
    #     plt.subplot(3,1,1)
    #     plt.plot(complex_tf,np.fft.fftshift(abs(test_set[i])))
    #     plt.xlim(-500,500)
    #     plt.subplot(3,1,2)
    #     plt.plot(tf_sampled,np.fft.fftshift(abs(fft(encoded_test_set[i]))))
    #     plt.subplot(3,1,3)
    #     plt.plot(complex_tf,np.fft.fftshift(abs(dic_test_set[i])))
        # plt.subplot(5,1,4)
        # pseudo = np.linalg.pinv(dictionary)
        # # x_pre = np.linalg.inv(np.dot(np.conjugate(dictionary.T),dictionary)).dot(np.conjugate(dictionary.T)).dot(dic_test_set[i])
        # x_pre = np.dot(pseudo,dic_test_set[i])
        # plt.plot(complex_tf,np.fft.fftshift(abs(x_pre)))
        # plt.subplot(5,1,5)
        # x_pre1 =np.dot(dictionary.T,dic_test_set[i])
        # plt.plot(complex_tf,np.fft.fftshift(abs(x_pre1)))
        # plt.show()       
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(complex_tf,np.fft.fftshift(abs(xf)))
    plt.subplot(2,1,2)
    plt.plot(complex_tf,np.fft.fftshift(abs(coef)))
    # plt.subplot(2,1,2)
    # plt.plot(complex_tf,np.fft.fftshift(np.real(coef)))
    plt.show()
    