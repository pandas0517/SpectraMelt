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
    from signals import filter_signal
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
        'wave_freq': [10,20],
        'start': -3,
        'stop': 3,
        'spacing': 0.001,
        'recovery': 'spg_bp'}
    #Input signal parameters
    wave_params = [
        {'amp': 1,
         'freq': 215,
         'phase': 0},
        {'amp': 1,
         'freq': 345,
         'phase': 0},
        {'amp': 0,
         'freq': 55,
         'phase': 0},
        {'amp': 0,
         'freq': 75,
         'phase': 0},
        {'amp': 0,
         'freq': 95,
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
            'f_c': 18,
            'width': 0.1,
            'shift': 0,
            'angle': pi/4},
        {
            'amp': 0.5,
            'f_c': -18,
            'width': 0.1,
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
    
    x, t, num_tones = multi_tone_sine_wave(system_params, wave_params, filter_params)
    xf = fft(x)
    complex_tf = np.linspace(-1/(2*system_params['spacing']), 1/(2*system_params['spacing']), int(t.size))
    # Create Phase Modulated Local Oscillator
    LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t, LO_params, system_params)
    if ( system_params['wavelets'] == 'y' ):
        #Create wavelet train
        first_modulation, no_shift_wavelet, wavelet_test = generate_wavelet_train(system_params, psi_params, rising_zero_crossings, sample_train, t)
        modulation, no_shift_wavelet_fast, wavelet_test_fast = generate_wavelet_train(system_params, psi_params, rising_zero_crossings, sample_train_fast, t)
    else:
        first_modulation = rising_zero_crossings
    y_mixed = np.multiply(x,rising_zero_crossings)
    #Mix wavelet train with input signal
    short_start = np.where( t == -2.5 )[0][0]
    short_end = np.where( t == 2.5 )[0][0]
    zero = np.where( t == 0 )[0][0]

    # y_mixed = np.copy(x*first_modulation)
    y_filtered, filt_freq, filt_freq_down = filter_signal(y_mixed, t, filter_params, system_params)
    y_filt_wavelet = np.copy(y_filtered*no_shift_wavelet)
    zero_pad_y_wavelet = np.pad(y_filt_wavelet, (clock_ticks-1,0), 'constant', constant_values=(0,0))
    y_integrate = np.zeros_like(y_filt_wavelet)
    for counter in range(0, zero_pad_y_wavelet.size - clock_ticks + 1):
        y_integrate[counter] = np.trapz(zero_pad_y_wavelet[counter:counter+clock_ticks-1])

    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(no_shift_wavelet[short_start:short_end]))))
    plt.subplot(4,1,2)
    plt.plot(complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(y_filt_wavelet[short_start:short_end]))))
    plt.subplot(4,1,3)
    plt.plot(complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(y_filtered[short_start:short_end]))))
    plt.subplot(4,1,4)
    plt.plot(complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(y_integrate[short_start:short_end]))))
    plt.show()
#     plt.subplot(3,2,1)
#     # plt.plot( t[short_start:short_end], abs(first_modulation[short_start:short_end]) )
#     plt.plot( t[short_start:short_end], abs(rising_zero_crossings[short_start:short_end]) )
#     plt.title('Time Domain Magnitude Representation: Wavelet Modulation\nLO Modulation Frequency: {}Hz \
# | LO Modulation Delta: {}'.format(LO_params['freq'],LO_params['phase_freq'],LO_params['phase_delta']))
#     plt.ylabel('Modulated LO\nLO Frequency: {}Hz\nWavelet Sample Train'.format(LO_params['freq']))
#     plt.subplot(3,2,2)
#     # plt.plot( complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(first_modulation[short_start:short_end]))) )
#     plt.plot( complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(rising_zero_crossings[short_start:short_end]))) )
#     plt.title('Frequency Domain Magnitude Representation: Wavelet Modulation\nGabor Wavelet - Center Frequencies: {}Hz and {}Hz | Width: \
# {} | Scaling: {}'.format(psi_params[0]['f_c'],psi_params[1]['f_c'],psi_params[0]['width'],psi_params[0]['amp']))
#     plt.ylabel('Modulated LO\nLO Frequency: {}Hz\nWavelet Sample Train'.format(LO_params['freq']))
#     plt.subplot(3,2,3)
#     plt.plot( t[short_start:short_end], abs(rising_zero_crossings[short_start:short_end]))
#     plt.plot( t[short_start:short_end], abs(LO[short_start:short_end]))
#     plt.ylabel('No Modulation\nLO Frequency: {}Hz\nWavelet Sample Train'.format(system_params['wave_freq'][0]))
#     plt.subplot(3,2,4)
#     plt.plot(complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(rising_zero_crossings[short_start:short_end]))))
#     plt.ylabel('No Modulation\nLO Frequency: {}Hz\nWavelet Sample Train'.format(system_params['wave_freq'][0]))
#     plt.subplot(3,2,5)
#     plt.plot( t[short_start:short_end], abs(y_mixed[short_start:short_end]) )
#     plt.xlabel('Seconds')
#     plt.ylabel('No Modulation\nLO Frequency: {}Hz\nWavelet Sample Train'.format(system_params['wave_freq'][1]))
#     plt.subplot(3,2,6)
#     plt.plot( complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(y_mixed[short_start:short_end]))) )
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('No Modulation\nLO Frequency: {}Hz\nWavelet Sample Train'.format(system_params['wave_freq'][1]))
#     plt.subplot(3,3,1)
#     plt.plot( t, np.real(wavelet_test[6]['wavelet']), label="Time Shift -1.5 Seconds", color="blue" )
#     plt.title("Real Part of Gabor Wavelet at Different Time Shifts\nScaling: {} | Center Frequency: {}Hz | \
# Width: {}".format(wavelet_test[6]['amp'], wavelet_test[6]['f_c'], wavelet_test[6]['width']))
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.subplot(3,3,2)
#     plt.title("Magnitude of Frequency Response for Different Time Shifts")
#     plt.plot( complex_tf, np.fft.fftshift(abs(fft(wavelet_test[6]['wavelet']))), label="Time Shift -1.5 Seconds", color="blue")
#     plt.ylabel('Magnitude')
#     plt.legend()
#     plt.subplot(3,3,3)
#     plt.title("Phase of Frequency Response for Different Time Shifts")
#     plt.phase_spectrum(wavelet_test[6]['wavelet'], Fs=(1/system_params['spacing']), label="Time Shift -1.5 Seconds", color="blue")
#     plt.ylabel('Radians')
#     plt.xlabel('')
#     plt.legend()
#     plt.subplot(3,3,4)
#     plt.plot( t, np.real(wavelet_test[7]['wavelet']), label="No Time Shift", color="green" )
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.subplot(3,3,5)
#     plt.plot( complex_tf, np.fft.fftshift(abs(fft(wavelet_test[7]['wavelet']))), label="No Time Shift", color="green" )
#     plt.ylabel('Magnitude')
#     plt.legend()
#     plt.subplot(3,3,6)
#     plt.phase_spectrum(wavelet_test[7]['wavelet'], Fs=(1/system_params['spacing']), label="No Time Shift", color="green" )
#     plt.ylabel('Radians')
#     plt.xlabel('')
#     plt.legend()
#     plt.subplot(3,3,7)
#     plt.plot( t, np.real(wavelet_test[8]['wavelet']), label="Time Shift 1.5 Seconds", color="Red" )
#     plt.ylabel('Amplitude')
#     plt.xlabel('Seconds')
#     plt.legend()
#     plt.subplot(3,3,8)
#     plt.plot( complex_tf, np.fft.fftshift(abs(fft(wavelet_test[8]['wavelet']))), label="Time Shift 1.5 Seconds", color="Red")
#     plt.ylabel('Magnitude')
#     plt.xlabel('Frequency (Hz)')
#     plt.legend()
#     plt.subplot(3,3,9)
#     plt.phase_spectrum(wavelet_test[8]['wavelet'], Fs=(1/system_params['spacing']), label="Time Shift 1.5 Seconds", color="Red")
#     plt.ylabel('Radians')
#     plt.xlabel('Frequency (Hz)')
#     plt.legend()
#     plt.figure()
#     plt.subplot(3,2,1)
#     plt.plot( t[short_start:short_end],rising_zero_crossings[short_start:short_end] )
#     plt.title('Time Domain Magnitude Representation: No Wavelet Modulation\nLO Modulation Frequency: {}Hz \
# | LO Modulation Delta: {}'.format(LO_params['phase_freq'],LO_params['phase_delta']))
#     plt.ylabel('Modulated LO\nLO Frequency: {}Hz\nSample Train'.format(LO_params['freq']))
#     plt.subplot(3,2,2)
#     plt.plot( complex_tf[short_start:short_end],np.fft.fftshift(abs(fft(rising_zero_crossings[short_start:short_end]))) )
#     plt.title('Frequency Domain Magnitude Representation\nNo Wavelet Modulation')
#     plt.ylabel('Modulated LO\nLO Frequency: {}Hz\nSample Train'.format(LO_params['freq']))
#     plt.subplot(3,2,3)
#     plt.plot( t[short_start:short_end], sample_train[short_start:short_end] )
#     plt.ylabel('No Modulation\nLO Frequency: {}Hz\nSample Train'.format(system_params['wave_freq'][0]))
#     plt.subplot(3,2,4)
#     plt.plot( complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(sample_train[short_start:short_end]))) )
#     plt.ylabel('No Modulation\nLO Frequency: {}Hz\nSample Train'.format(system_params['wave_freq'][0]))
#     plt.subplot(3,2,5)
#     plt.plot( t[short_start:short_end], sample_train_fast[short_start:short_end] )
#     plt.xlabel('Seconds')
#     plt.ylabel('No Modulation\nLO Frequency: {}Hz\nSample Train'.format(system_params['wave_freq'][1]))
#     plt.subplot(3,2,6)
#     plt.plot( complex_tf[short_start:short_end], np.fft.fftshift(abs(fft(sample_train_fast[short_start:short_end]))) )
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('No Modulation\nLO Frequency: {}Hz\nSample Train'.format(system_params['wave_freq'][1]))
    