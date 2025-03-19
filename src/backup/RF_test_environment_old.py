'''
Created on May 17, 2023

@author: pete
'''

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import trapz
    from signals import butter_lowpass_filter
    from signals import multi_tone_sine_wave
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import orthogonal_mp
    import scipy.fft as fft
    import math
    
    # Constants
    e = math.e
    pi = math.pi
    #Time vector parameters in seconds
    time_params = {
                    'start':-3,
                    'stop':3,
                    'spacing':0.001}
    #Input signal parameters
    wave_params = [
        {'amp':1,
         'freq':10,
         'phase': 0},
        {'amp':0,
         'freq':20,
         'phase':0},
        {'amp':1,
         'freq':30,
         'phase': 0},
        {'amp':0,
         'freq':40,
         'phase':0},
        {'amp':1,
         'freq':50,
         'phase': 0}]
    #RF input and time vector creation
    x, t = multi_tone_sine_wave(time_params, wave_params)
    tf = np.linspace(0.0, 1/(2*time_params['spacing']), int(t.size/2))
    tf_imag = np.linspace(-1/(2*time_params['spacing']), 1/(2*time_params['spacing']), int(t.size))
    #dftmtx = fft.fft(np.eye(t.size))

# Filter requirements.
    order = 6
    fs = 1/spacing       # sample rate, Hz
    cutoff = 120 # desired cutoff frequency of the filter, Hz
    complex_tf = np.linspace(-1/(2*spacing), 1/(2*spacing), int(t.size))
    # x = []
    # x_input = np.zeros_like(t)
    # for param in signal_param:
    #     new_signal = param['amp']*np.sin(2*pi*param['freq']*t+param['phase'])
    #     x.append(new_signal)
    #     x_input = np.add(x_input,new_signal)

    xf_input = fft.fft(x)
    #xf_test = np.matmul(dft_norm,x_input)
    mag_xf_input = np.copy(2*np.abs(xf_input[0:int(t.size/2)])/t.size)
    #mag_xf_test = np.copy(2*np.abs(xf_test[0:int(t.size/2)]))
# Phase modulated local oscillator (NYFR)
    Mod_param = {
        'delta': 2000,
        'freq': 50,
        'phase': 0}
    LO_modulation = (Mod_param['delta']/Mod_param['freq'])*np.sin(2*pi*Mod_param['freq']*t+Mod_param['phase'])

    LO_param = {
        'amp':1,
        'freq':50,
        'phase':0,
    }
    LO = LO_param['amp']*np.sin(2*pi*LO_param['freq']*t+LO_param['phase']+LO_modulation)
#zero-crossing pulse generator    
    zero_crossings = np.where(np.diff(np.sign(LO)))[0] + 1
    if (LO[zero_crossings[0]] < LO[zero_crossings[1]]):
        rising_zero_crossings = np.copy(zero_crossings[1::2])
    else:
        rising_zero_crossings = np.copy(zero_crossings[0::2])
    rising_zero_crossing = np.zeros_like(t)
    rising_zero_crossing[rising_zero_crossings] = 1
    # dictionary = dftmtx[rising_zero_crossings]
    pulse_train_f = fft.fft(rising_zero_crossing)
    mag_pulse_train_f = np.copy(2*np.abs(pulse_train_f[0:int(t.size/2)])/t.size)
    # dictionary = np.matmul(rising_zero_crossing, dftmtx)
# Gabor atom with Gaussian window
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
# Adding shifts from rising zero crossings into the Gabor atom parameters
    psi_par = []
    for idx in range(0, rising_zero_crossings.size, len(psi_params)):
        count = 0
        for params in psi_params:
            param = dict(params)
            if (idx+count < rising_zero_crossings.size):
                param['shift'] = int(rising_zero_crossings[idx+count])
            psi_par.append(param)
            count+=1
#Gabor wavelet train construction
    wavelet = []
    wavelet_train = np.zeros_like(t)
    for param in psi_par:
        psi = param['amp']*((2**(1/4))/(math.sqrt(param['width'])*pi**(1/4)))* \
            e**(2*pi*param['f_c']*(t-t[param['shift']])*1j)* \
            e**(-(((t-t[param['shift']])/param['width'])**2))
        wavelet.append(psi)
        wavelet_train = np.add(psi,wavelet_train)
# Random Demodulation
    rd_clock_period = 4 #integer multiple of spacing time
    pseudo = np.repeat(np.random.choice([-1,1], size=int(t.size/rd_clock_period)), rd_clock_period)
# Filter in Fractional Fourier Domain
    y = np.zeros_like(t, dtype=np.complex_)
    y_int = np.zeros_like(t, dtype=np.complex_)
    y_int = np.add(x_input,y_int)
    # y_measurments = np.zeros_like(t, dtype=np.complex_)
    # y_measurments[rising_zero_crossings] = y_int[rising_zero_crossings]
    y_int = np.copy(y_int*rising_zero_crossing)
    # y_int = np.multiply(y_int, wavelet_train)
    # y_int = np.multiply(y_int, pseudo)
    adc_clock_period = 20
    # for params in psi_params:
    #     FrFT = e**(((-1/2)*t**2*1j)*(1/math.tan(params['angle'])))
    #     y_int = np.multiply(FrFT,y_int)
    #     for i in range(0, t.size, adc_clock_period):
    #         t_window = np.copy(t[i:i+adc_clock_period-1])
    #         y_window = np.copy(y[i:i+adc_clock_period-1])
    #         integral = trapz(y_window, t_window)
    #         y[i+adc_clock_period-1] = integral
    #     iFrFT = e**(((1/2)*t**2*1j)*(1/math.tan(params['angle'])))
    #     y = np.multiply(iFrFT, y)
    # FrFT = e**(((-1/2)*t**2*1j)*(1/math.tan(psi_params[0]['angle'])))
    # y_int = np.multiply(FrFT,y_int)
    y_int_pad = np.pad(y_int, (adc_clock_period,0), 'constant', constant_values=(0,0))
    t_pad = np.arange(start-(adc_clock_period*spacing), stop, spacing)
    # for i in range(0, t.size, adc_clock_period):
    #     t_window = np.copy(t[i:i+adc_clock_period])
    #     y_window = np.copy(y_int[i:i+adc_clock_period])
    #     integral = trapz(y_window, t_window)
    #     y[i+adc_clock_period-1] = integral
    for i in range(t.size):
        t_window = np.copy(t_pad[i:i+adc_clock_period])
        y_window = np.copy(y_int_pad[i:i+adc_clock_period])
        integral = trapz(y_window, t_window)
        y[i] = integral
    # iFrFT = e**(((1/2)*t**2*1j)*(1/math.tan(psi_params[0]['angle'])))
    # y = np.multiply(iFrFT, y)
# Filter using Butterworth Lowpass
    y_filtered = butter_lowpass_filter(y_int, cutoff, fs, order)
    y_filtered_f = fft.fft(y_filtered)
    y_sampled = np.zeros(int(t.size/adc_clock_period),dtype='complex')
    index = 0
    for i in range(0, t.size, adc_clock_period):
        y_sampled[index] = y_filtered[i]
        index += 1
# Recover signal
    n_nonzero_coefs = 3
    # omp_real = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    # omp_imag = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp_real = OrthogonalMatchingPursuit()
    omp_imag = OrthogonalMatchingPursuit()
    omp = OrthogonalMatchingPursuit()
# Create NYFR dictionary
    # Zones = int(fs*2/LO_param['freq'])
    Zones = int(t.size/y_sampled.size)
    # R_init = np.eye(int(Zones/2))
    K_band = int(y_sampled.size)
    R_init = np.eye(K_band)
    R = np.copy(R_init)
    for i in (range(Zones-1)):
        R = np.hstack((R,R_init))
    R_row, R_col = R.shape 
    S = np.zeros((R_col,R_col),dtype='complex')
    PSI = np.zeros_like(S)
    idft_norm = fft.ifft(np.eye(K_band))/K_band
    M_index = [0]
    M_pattern = [-1,1]
    for i in range(1,K_band+1):
        if i == K_band+1:
            M_temp = [-i]
        else:
            M_temp = [x * i for x in M_pattern]
        M_index = M_index + M_temp
    index = 0 
    for i in range(0, R_col, K_band):
        t_a = t[index*adc_clock_period]
        LO_mod = LO_modulation[index*adc_clock_period]
        if M_index[index] == 0:
            S[i:i+R_row,i:i+R_row] = np.copy(R_init)
        else:
            # S[i:i+R_row,i:i+R_row] = np.copy(e**(M_index[index]*LO_modulation*1j)*R_init)
            S[i:i+R_row,i:i+R_row] = np.copy(e**(M_index[index]*LO_mod*1j)*R_init)
        PSI[i:i+R_row,i:i+R_row] = np.copy(idft_norm)
        index += 1
    #print(S[-R_row:,-R_row:])
    dictionary = np.matmul(np.matmul(R,S),PSI)
    #dictionary = np.matmul(dictionary,PSI)
    mag_y_filtered_f = np.copy(2*np.abs(y_filtered_f[0:int(t.size/2)])/t.size)
    y_int_f = fft.fft(y_int)
    mag_y_int_f = np.copy(2*np.abs(y_int_f[0:int(t.size/2)])/t.size)
    dft_norm_in = fft.fft(np.eye(t.size))/t.size
    dft_norm_out = fft.fft(np.eye(y_sampled.size))/y_sampled.size
    #y_sampled_f = fft.fft(y_sampled)
    y_sampled_f = np.matmul(dft_norm_out,y_sampled)
    xf_test = np.matmul(dft_norm_in,x_input)
    tt = np.imag(dictionary)
    tpt = np.real(dictionary)
    ttt = np.real(y_sampled)/y_sampled.size
    # mag_y_f = np.copy(2*np.abs(y_f[0:int(t.size/2)])/t.size)
    #mag_dft = np.copy(2*np.abs(dftmtx[0:int(t.size/2)])/t.size)
    #mag_dft_normed = mag_dft / mag_dft.max(axis=0)
    # mag_y_f_normed = mag_y_f / mag_y_f.max(axis=0)
    # y_f_normed = y_f / y_f.max(axis=0)
    # omp_real.fit(np.real(dictionary), np.real(y_sampled/y_sampled.size))
    # omp_imag.fit(np.imag(dictionary), np.imag(y_sampled/y_sampled.size))
    test_real = orthogonal_mp(np.real(dictionary),np.real(y_sampled),n_nonzero_coefs=n_nonzero_coefs)
    test_imag = orthogonal_mp(np.imag(dictionary),np.real(y_sampled)/y_sampled.size,n_nonzero_coefs=n_nonzero_coefs)
    # omp_real.fit(np.real(dftmtx), np.real(y_f_normed))
    # omp_imag.fit(np.imag(dftmtx), np.imag(y_f_normed))
    # omp.fit(abs(dftmtx),abs(y_f))
    # coef_real = omp_real.coef_
    # coef_real = np.zeros_like(omp_real.coef_)
    # coef_imag = omp_imag.coef_
    #coef_imag = np.zeros_like(coef_real)
    # coef = coef_real + coef_imag*1j
    coef = test_real + test_imag*1j
    # coef = omp.coef_
    # (idx_r,) = coef.nonzero()
    plt.figure()
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
    
    plt.subplot(2,1,1)
    plt.plot(tf_imag,np.abs(xf_input))
    #plt.plot(tf_imag,np.fft.fftshift(abs(xf_test)))
    plt.subplot(2,1,2)
    plt.plot(tf_imag,np.abs(coef))
    # plt.plot(t,np.real(wavelet[0]))
    # plt.plot(t,np.real(wavelet[20]))
    plt.show()