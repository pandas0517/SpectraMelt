'''
Created on Jul 17, 2023

@author: pete
'''

# class MyClass(object):
#     '''
#     classdocs
#     '''
#
#
#     def __init__(self, params):
#         '''
#         Constructor
#         '''
from scipy.signal import butter, sosfilt, sosfreqz #,lfilter
from scipy.integrate import trapz
from scipy.fft import fft, ifft
from scipy.linalg import dft
import numpy as np
from numpy import sin, sqrt #,tan
from math import e, pi, sqrt
from sklearn.linear_model import orthogonal_mp, OrthogonalMatchingPursuit
from spgl1 import spg_bp
from spgl1 import spgl1
from scipy import linalg
from OMP import OMP
import tensorflow as tf

def simulate_system(wave_params, eps, LO_params, system_params, psi_params, filter_params, manager_queue):
    x, t, num_tones = multi_tone_sine_wave(system_params, wave_params, filter_params)
    xf = fft(x)
    complex_tf = np.linspace(-1/(2*system_params['spacing']), 1/(2*system_params['spacing']), int(t.size), endpoint=False)
    # Create Phase Modulated Local Oscillator
    LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t, LO_params, system_params)
    if ( system_params['wavelets'] == 'y' ):
        #Create wavelet train
        first_modulation, no_shift_wavelet, wavelet_test = generate_wavelet_train(system_params, psi_params, rising_zero_crossings, sample_train, t)
    else:
        first_modulation = rising_zero_crossings
        no_shift_wavelet = 0
    #Mix wavelet train with input signal
    y_mixed = np.copy(x*rising_zero_crossings)
    # y_mixed = np.copy(x*first_modulation)
    #Random Demodulation
    y_demod = random_demodulation(y_mixed, system_params)
    #Filter demodulated signal
    y_filtered, filt_freq, filt_freq_down = filter_signal(y_demod, t, filter_params, system_params)
    # x_filtered, filt_freq = filter_signal(x, t, filter_params, system_params)
    #Downsample with ADC
    y_sampled, LO_mod_sampled, t_sampled, tf_sampled, filt_sampled, downsample_train = downsample(y_filtered, LO_mod, t, system_params, rising_zero_crossings, filt_freq)
    #Create NYFR Dictionary for signal reconstruction
    dictionary = create_nyfr_dict(t, LO_mod, LO_mod_sampled, filt_sampled, system_params, tf_sampled)
    # dictionary = create_nyfr_dict(t, LO_mod, system_params)
    # test_model = np.matmul(abs(filt_freq_down)*np.eye(filt_freq_down.size),np.matmul(dictionary,xf))
    # test_model = np.matmul(np.real(dictionary),np.real(xf)) + np.matmul(np.imag(dictionary),np.imag(xf))*1j
    # test_model = np.matmul(np.imag(dictionary),np.imag(xf))*1j
    # test_model = np.matmul(np.real(dictionary),np.real(xf))
    test_model = np.matmul(dictionary,xf)
    #Recover signal
    coef = recover_signal(dictionary, y_sampled, system_params, num_tones)
    # coef = xf
    # coef1 = np.multiply(test,fft(test_model))
    matching_tones = np.multiply(xf,coef)
    # matching_tones = np.multiply(np.abs(xf),np.abs(coef))
    matching_tones[ matching_tones < eps ] = 0
    non_zero_matches = np.count_nonzero(matching_tones)
    current_run = []
    current_run.append(LO_params['freq'])
    current_run.append(system_params['adc_clock_freq'])
    current_run.append(complex_tf)
    current_run.append(xf)
    current_run.append(coef)
    current_run.append(non_zero_matches)
    current_run.append(rising_zero_crossings)
    current_run.append(y_mixed)
    current_run.append(y_filtered)
    current_run.append(t)
    current_run.append(t_sampled)
    current_run.append(tf_sampled)
    current_run.append(y_sampled)
    current_run.append(test_model)
    current_run.append(filt_freq)
    current_run.append(filt_freq_down)
    current_run.append(fft(y_mixed))
    current_run.append(LO)
    current_run.append(x)
    current_run.append(first_modulation)
    current_run.append(no_shift_wavelet)
    current_run.append(fft(filt_sampled))
    current_run.append(downsample_train)
    current_run.append(dictionary)
    # current_run = [ LO_params['freq'], system_params['adc_clock_freq'], complex_tf, xf, coef, matching_tones ]
    manager_queue.put(current_run)

def multi_tone_sine_wave(system_params, wave_params, filter_params):
    if ( system_params['filter'] == 'integrate' ):
        start_time = system_params['start'] - filter_params['window_size'] * system_params['spacing']
    else:
        start_time = system_params['start']
    stop_time = system_params['stop']
    total_time = abs( start_time - stop_time )
    points_per_second = 1/system_params['spacing']
    K_band = int( points_per_second / system_params['adc_clock_freq'] )
    K_band_remainder = int(points_per_second % system_params['adc_clock_freq'])
    if ( K_band_remainder != 0 ):
        # adjust_spacing = input("ADC frequency doesn't line up with spacing")
        # print("ADC frequency doesn't line up with spacing. Adjusting")
        # if ( adjust_spacing == 'y' ):
        points_per_second -= K_band_remainder        # plt.plot(tf_sampled,np.fft.fftshift(abs(test_model)))
        K_band = int( points_per_second / system_params['adc_clock_freq'] )
    if ( K_band % 2 != 0 ):
        # print("K band not even number. Adjusting") 
        points_per_second += int( system_params['adc_clock_freq'] )
        K_band = int( points_per_second / system_params['adc_clock_freq'] )
    system_params['spacing'] = 1/points_per_second
    # print(system_params['spacing'])
    num_time_points = int ( total_time * points_per_second )
    t = np.linspace(start_time, stop_time, num_time_points, endpoint=False)
    x = []
    x_input = np.zeros_like(t)
    n_non_zero_amps = 0
    for param in wave_params:
        if ( param['amp'] != 0 ):
            n_non_zero_amps += 1
        new_signal = param['amp']*sin(2*pi*param['freq']*t+param['phase'])
        x.append(new_signal)
        x_input = np.add(x_input,new_signal)
    return x_input, t, n_non_zero_amps

def generate_LO(t, LO_params, system_params):
    
    LO_modulation = (LO_params['phase_delta']/LO_params['phase_freq'])* \
        sin(2*pi*LO_params['phase_freq']*t+LO_params['phase_offset'])
    LO = LO_params['amp']*sin(2*pi*LO_params['freq']*t+LO_params['phase']+LO_modulation)
    #zero-crossing pulse generator 
    zero_crossings = np.where(np.diff(np.signbit(LO)))[0] + 1
    # test 
    rising_zero_crossings = np.zeros_like(t)
    #edge case
    start = t[0]-system_params['spacing']
    end = t[0]
    start_LO_mod = (LO_params['phase_delta']/LO_params['phase_freq'])* \
        sin(2*pi*LO_params['phase_freq']*start+LO_params['phase_offset'])
    start_LO = LO_params['amp']*sin(2*pi*LO_params['freq']*start+LO_params['phase']+start_LO_mod)
    end_LO_mod = (LO_params['phase_delta']/LO_params['phase_freq'])* \
        sin(2*pi*LO_params['phase_freq']*end+LO_params['phase_offset'])
    end_LO = LO_params['amp']*sin(2*pi*LO_params['freq']*end+LO_params['phase']+end_LO_mod)
    if ( start_LO*end_LO < 0 and start_LO < end_LO ):
        rising_zero_crossings[0] = 1
    for i in zero_crossings:
        if (LO[i] > LO[i-1]):
            rising_zero_crossings[i] = 1
    
    clock_ticks = round(1/(system_params['spacing']*system_params['wave_freq'][0]))
    adc_start_time = (rising_zero_crossings!=0).argmax(axis=0)
    sample_train = np.zeros_like(t)   
    for i in range(adc_start_time, t.size, clock_ticks):
        sample_train[i] = 1

    clock_ticks_fast = round(1/(system_params['spacing']*system_params['wave_freq'][1]))
    sample_train_fast = np.zeros_like(t)   
    for i in range(adc_start_time, t.size, clock_ticks_fast):
        sample_train_fast[i] = 1

    return LO_modulation, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks

def generate_wavelet_train(system_params, psi_params, rising_zero_crossings, sample_train, t):
    rising_zero_crossings_idx = np.nonzero(rising_zero_crossings)[0]
    sample_train_idx = np.nonzero(sample_train)[0]
    # Adding shifts from rising zero crossings into the Gabor atom parameters
    psi_par_mod_lo = []
    for idx in range(0, rising_zero_crossings_idx.size, len(psi_params)):
        count = 0
        for params in psi_params:
            param_mod_lo = dict(params)
            if (idx+count < rising_zero_crossings_idx.size):
                param_mod_lo['shift'] = int(rising_zero_crossings_idx[idx+count])
            if ( param_mod_lo['shift'] <= len(t) ):
                psi_par_mod_lo.append(param_mod_lo)
            count+=1
            
    psi_par_adc_shift = []
    for count, idx in enumerate(sample_train_idx):
        current_wave = count % 2
        params = psi_params[current_wave]
        param_adc_shift = dict(params)         
        param_adc_shift['shift'] = idx
        psi_par_adc_shift.append(param_adc_shift)
    #Gabor wavelet train construction
    wavelet = []
    wavelet_train = np.zeros_like(rising_zero_crossings)
    for param in psi_par_mod_lo:
        psi = param['amp']*((2**(1/4))/(sqrt(param['width'])*pi**(1/4)))* \
            e**(2*pi*param['f_c']*(t-t[param['shift']])*1j)* \
            e**(-(((t-t[param['shift']])/param['width'])**2))
        wavelet.append(psi)
        wavelet_train = np.add(psi,wavelet_train)
    wavelet_no_shift = []
    no_shift_wavelet = np.zeros_like(t)
    for param in psi_par_adc_shift:
        psi = param['amp']*((2**(1/4))/(sqrt(param['width'])*pi**(1/4)))* \
            e**(2*pi*param['f_c']*(t-t[param['shift']])*1j)* \
            e**(-(((t-t[param['shift']])/param['width'])**2))
        wavelet_no_shift.append(psi)
        no_shift_wavelet = np.add(psi, no_shift_wavelet)      
    # time_shifts = [ np.where( t == -1.5 )[0][0], np.where( t == 0 )[0][0], np.where( t == 1.5 )[0][0] ]
    time_shifts = [ 1, 2, 3 ]
    wavelet_test = []
    test_params = [
        {
            'amp': 0.5,
            'f_c': -35,
            'shift': 0,
            'width': 0.01,
            'wavelet': no_shift_wavelet},
        {
            'amp': 0.5,
            'f_c': 0,
            'shift': 0,
            'width': 0.01,
            'wavelet': no_shift_wavelet},
        {
            'amp': 0.5,
            'f_c': 35,
            'shift': 0,
            'width': 0.1,
            'wavelet': no_shift_wavelet}
        ]
    for t_param in test_params:
        for time_shift in time_shifts:
            param = dict(t_param)
            psi_test = t_param['amp']*((2**(1/4))/(sqrt(t_param['width'])*pi**(1/4)))* \
                e**(2*pi*t_param['f_c']*(t-t[time_shift])*1j)* \
                e**(-(((t-t[time_shift])/t_param['width'])**2))
            param['wavelet'] = psi_test
            param['shift'] = time_shift
            wavelet_test.append(param)
    return wavelet_train, no_shift_wavelet, wavelet_test

def random_demodulation(data, system_params):
    clock_ticks = 1/(system_params['rd_clock_freq']*system_params['spacing'])
    pseudo = np.repeat(np.random.choice([-1,1], size=int(data.size/clock_ticks)), clock_ticks)
    if ( system_params['rand_demod'] == 'y' ):
        return np.multiply(data,pseudo)
    else:
        return data

def filter_signal(data, t, filter_params, system_params):
    # FrFT = e**(((-1/2)*t**2*1j)*(1/tan(filter_params['angle'])))
    # data_int = np.multiply(FrFT,data)
    data_int = data
    clock_ticks = int(data.size*system_params['spacing']*system_params['adc_clock_freq']/2)
    if (system_params['filter'] == 'butter'):
        samp_freq = 1/system_params['spacing']
        adc_freq = 1/system_params['adc_clock_freq']
        sos = butter(filter_params['order'], filter_params['cutoff_freq'], fs=samp_freq, btype='lowpass', analog=False, output='sos')
        w, filt_freq = sosfreqz(sos, worN=data.size, whole=True)
        filt_freq_down = np.concatenate((filt_freq[:clock_ticks],filt_freq[-clock_ticks:]))
        y_int = sosfilt(sos, data_int)
        # y_int = lfilter(b, a, data_int)
    else:
        y_int = np.zeros_like(t, dtype=np.complex_)
        for i in range(t.size):
            t_window = np.copy(t[i:i+filter_params['window_size']])
            data_window = np.copy(data_int[i:i+filter_params['window_size']])
            integral = trapz(data_window, t_window)
            y_int[i] = integral
    # iFrFT = e**(((1/2)*t**2*1j)*(1/tan(filter_params['angle'])))
    # y = np.real(np.multiply(iFrFT, y_int))
    y = y_int
    return y, filt_freq, filt_freq_down

def downsample(data, LO_modulation, t, system_params, rising_zero_crossings, filt_freq):
    filt_time = ifft(filt_freq)
    test = fft(filt_time)
    clock_ticks = round(1/(system_params['spacing']*system_params['adc_clock_freq']))
    downsampled_data_size = int(data.size/clock_ticks) 
    downsample_train = np.zeros_like(data)
    downsampled_data = np.zeros(downsampled_data_size,dtype=data.dtype)
    downsampled_LO = np.zeros(downsampled_data_size,dtype=data.dtype)
    downsampled_t = np.zeros(downsampled_data_size,dtype=data.dtype)
    downsampled_filt = np.zeros(downsampled_data_size,dtype=filt_freq.dtype)
    downsampled_range = data.size
    # test = np.where( t == 0 )   
    # adc_start_time = (rising_zero_crossings!=0).argmax(axis=0)
    adc_start_time = 0
    for index_data,i in enumerate(range(adc_start_time, downsampled_range, clock_ticks)):
        downsample_train[i] = 1
        downsampled_data[index_data] = data[i]
        downsampled_LO[index_data] = LO_modulation[i]
        downsampled_t[index_data] = t[i]
        downsampled_filt[index_data] = filt_time[i]
    downsampled_tf = np.linspace(-1/(2*system_params['spacing']*clock_ticks), 1/(2*system_params['spacing']*clock_ticks), downsampled_data_size, endpoint=False)
    return downsampled_data, downsampled_LO, downsampled_t, downsampled_tf, downsampled_filt, downsample_train

def create_nyfr_dict(t, LO_modulation, LO_mod_sampled, filt_down, system_params, tf_sampled):
    # K_band = t.size*(system_params['spacing']*system_params['adc_clock_freq'])
    # K_band = ceil(t.size*(system_params['spacing']*system_params['adc_clock_freq']))
    K_band = round(t.size*(system_params['spacing']*system_params['adc_clock_freq']))
    Zones = int(t.size/K_band)
    filt_freq_down = fft(filt_down)
    M_index = []
    M_temp = [0,0]
    # M_temp = [-2,1]
    M_pattern = [0, 1]
    for i in range(0,int(Zones/2)):
        M_temp[0] = M_pattern[0] + i
        M_temp[1] = -( M_pattern[1] + i )
        # M_temp[0] = M_temp[0] + 2
        # M_temp[1] = M_temp[1] - 2
        M_index = M_index + M_temp
 
    R_init = np.eye(K_band)
    # test_init = (abs(filt_freq_down)*abs(filt_freq_down)) * np.eye(K_band)
    # test_init = abs(filt_freq_down) * np.eye(K_band)
    # test_11 = 1/ ( 0.001*np.power(tf_sampled,2) - 2.1 )
    # test_11 = -5e-8*np.power(tf_sampled,4) + 2e-12*np.power(tf_sampled,3) - 5e-6*np.power(tf_sampled,2) + 4e-9*tf_sampled + 0.9988
    test_init = filt_freq_down * np.eye(K_band)
    # test_init = test_11 * np.eye(K_band)
    R = np.copy(R_init)
    dft_matrix = dft(K_band)
    # dft_matrix = np.matmul(test_init,dft(K_band))
    ## dft_matrix = np.matmul( dft(filt_freq_down ), np.eye(K_band) )
    idft_norm = np.transpose(np.conjugate(dft_matrix))/(100*K_band)
    # idft_norm = np.transpose(np.conjugate(dft_matrix))/(K_band)
    if (system_params['dictionary'] == 'complex'):
        for i in (range(Zones-1)):
            R = np.hstack((R,R_init))
        R_row, R_col = R.shape 
        S = np.zeros((R_col,R_col),dtype='complex')
        PSI = np.zeros_like(S)
        index = 0 
        for i in range(0, R_col - K_band, K_band):
            # if (index == 0):
            #     LO_mod = LO_modulation[index*clock_ticks]
            # else:
            #     LO_mod = LO_modulation[index*clock_ticks-1]
            if M_index[index] == 0:
                S[i:i+R_row,i:i+R_row] = np.copy(R_init)
            else:
                S[i:i+R_row,i:i+R_row] = np.copy(e**(M_index[index]*LO_modulation[index]*1j)*R_init)
            PSI[i:i+R_row,i:i+R_row] = np.copy(idft_norm)
            index += 1
    elif (system_params['dictionary'] == 'real'):
        for i in (range(2*Zones-1)):
            R = np.hstack((R,R_init))
        # R_filt = fft(filt_freq_down) * np.eye(K_band)
        R_row, R_col = R.shape 
        S = np.zeros((R_col,R_col),dtype='complex')
        PSI = np.zeros((R_col, int(R_col/2)),dtype='complex')
        PSI_1 = np.zeros((R_col, int(R_col/2)),dtype='complex')
        idft_split = np.hsplit(idft_norm,2)
        zero_fill = np.zeros_like(idft_split[0])
        U_idft = np.hstack((idft_split[0], zero_fill))
        U_idft_test = np.hstack((zero_fill, idft_split[0]))
        L_idft_test = np.hstack((idft_split[1], zero_fill))
        L_idft = np.hstack((zero_fill, idft_split[1]))
        UL_idft = np.vstack((U_idft,L_idft))
        UL_idft_1 = np.vstack((U_idft, U_idft_test))
        UL_idft_2 = np.vstack((L_idft_test, L_idft))
        M_index_reverse = [i * -1 for i in M_index]
        M_index_reverse.reverse()
        double_M_index = (M_index + M_index_reverse).copy()
        # double_M_index = (M_index_reverse + M_index).copy()
        if ( system_params['sampled_LO'] == 'y' ):
            LO_list = []
            for i in (range(0,Zones)):
                # LO_list.append(np.flip(LO_mod_sampled))
                LO_list.append(LO_mod_sampled)
            LO_mod_concat = np.concatenate(LO_list)
            double_LO_modulation = np.concatenate((LO_mod_concat,LO_mod_concat))
        elif ( system_params['sampled_LO'] == 't' ):
            LO_list = []
            for LO_sampled in LO_mod_sampled:
                LO_samp = np.empty(Zones)
                LO_samp.fill(LO_sampled)
                LO_list.append(LO_samp)
            LO_mod_concat = np.concatenate(LO_list)
            double_LO_modulation = np.concatenate((LO_mod_concat,np.flip(LO_mod_concat)))
        else:
            double_LO_modulation = np.concatenate((LO_modulation,np.flip(LO_modulation)))
        pass
        for index,i in enumerate(range(0, R_col, K_band)):
            LO_mod = double_LO_modulation[i:i+R_row]
            S[i:i+R_row,i:i+R_row] = np.diag(e**(double_M_index[index]*LO_mod*1j))
            # S[i:i+R_row,i:i+R_row] = np.matmul(R_filt,np.diag(e**(double_M_index[index]*LO_mod*1j)))
        for i in range (0, R_col, 2*K_band):
            PSI[i:i+2*K_band,int(i/2):int(i/2)+K_band] = np.copy(UL_idft)
            if ( i >= int(R_col/2) ):
                PSI_1[i:i+2*K_band,int(i/2):int(i/2)+K_band] = np.copy(UL_idft_2)
            else:
                PSI_1[i:i+2*K_band,int(i/2):int(i/2)+K_band] = np.copy(UL_idft_1)
    # test = abs(filt_freq_down)*abs(filt_freq_down)
    # R_filt = ifft(test)*np.eye(filt_freq_down.size)
    # R_filt_2 = np.matmul(R_filt, R_filt)
    # dictionary = np.matmul(R_filt, np.matmul(np.matmul(R,S),PSI))
    dictionary = np.matmul(np.matmul(R,S),PSI)
    return dictionary
    # return R, S, PSI
def recover_signal(dictionary, y_sampled, system_params, num_tones):
    y_sampled_norm = np.linalg.norm(y_sampled)
    coef_real = 0
    coef_imag = 0
    match system_params['recovery']:
    # if ( system_params['recovery'] == 'omp' ):
        case 'omp':
            # coef_real = OMP(dictionary, y_sampled/y_sampled_norm)
            omp_real = OrthogonalMatchingPursuit(n_nonzero_coefs=num_tones)
            omp_imag = OrthogonalMatchingPursuit(n_nonzero_coefs=num_tones)
            # omp_real.fit(np.real(dictionary), y_sampled/y_sampled_norm)
            # omp_imag.fit(np.imag(dictionary), y_sampled/y_sampled_norm)
            omp_real.fit(np.real(dictionary), y_sampled)
            omp_imag.fit(np.imag(dictionary), y_sampled)
            coef_imag = omp_imag.coef_
            coef_real = omp_real.coef_
    # elif ( system_params['recovery'] == 'o_mp' ):
        case 'o_omp':
            coef_real = orthogonal_mp(np.real(dictionary),y_sampled/y_sampled_norm)
            coef_imag = orthogonal_mp(np.imag(dictionary),y_sampled/y_sampled_norm,n_nonzero_coefs=(2*num_tones))
    # elif ( system_params['recovery'] == 'spg_bp' ):
        case 'spg_bp':
            # coef_real,resid,grad,info = spg_bp(dictionary,y_sampled/y_sampled_norm)
            coef_real,resid,grad,info = spg_bp(dictionary,y_sampled)
            # coef_real,resid_real,grad_real,info_real = spg_bp(np.real(dictionary),y_sampled/y_sampled_norm)
            # coef_imag,resid_imag,grad_imag,info_imag = spg_bp(np.imag(dictionary),y_sampled/y_sampled_norm)
    # elif ( system_params['recovery'] == 'spgl1' ):
        case 'spgl1':
            # coef_real,resid,grad,info = spgl1(dictionary,y_sampled/y_sampled_norm)
            coef_real,resid,grad,info = spgl1(dictionary,y_sampled)
            #coef_real,resid_real,grad_real,info_real = spgl1(np.real(dictionary),y_sampled/y_sampled_norm)
            #coef_imag,resid_imag,grad_imag,info_imag = spgl1(np.imag(dictionary),y_sampled/y_sampled_norm)
        case 'decode':
            # decoder_real = tf.keras.models.load_model('decoder_real.keras')
            # decoder_imag = tf.keras.models.load_model('decoder_imag.keras')
            # # test111 = np.real(np.reshape(y_sampled,[]))
            # y_sampled_real = np.real(y_sampled)
            # y_sampled_imag = np.imag(y_sampled)
            # y_sampled_real = y_sampled_real.reshape((1,y_sampled_real.shape[0]))
            # y_sampled_imag = y_sampled_imag.reshape((1,y_sampled_imag.shape[0]))
            # coef_real = np.transpose(decoder_real.predict(y_sampled_real))
            # coef_imag = np.transpose(decoder_imag.predict(y_sampled_imag))
            
            decoder_mag = tf.keras.models.load_model('decoder_mag.keras')
            decoder_ang = tf.keras.models.load_model('decoder_ang.keras')
            # test111 = np.real(np.reshape(y_sampled,[]))
            y_sampled_mag = np.abs(y_sampled)
            y_sampled_ang = np.angle(y_sampled)
            y_sampled_mag = y_sampled_mag.reshape((1,y_sampled_mag.shape[0]))
            y_sampled_ang = y_sampled_ang.reshape((1,y_sampled_ang.shape[0]))
            coef_mag = np.transpose(decoder_mag.predict(y_sampled_mag))
            coef_ang = np.transpose(decoder_ang.predict(y_sampled_ang))
            coef_real = coef_mag
            pass
        case _:
            print("No recovery performed")
    coef = coef_real + coef_imag*1j
    return coef