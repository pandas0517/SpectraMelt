from utility import load_settings
import numpy as np
from numpy import sin
from math import pi
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.integrate import trapezoid
from scipy.fftpack import fft, ifft, dft

class NYFR:
    def __init__(self, system_name='System_Config_1', 
                 filter=None,
                 adc_clock_freq=100, 
                 start=-1, 
                 stop=1, 
                 spacing=0.001, 
                 noise=0,
                 processing_systems=None, 
                 LO_params=None, 
                 wave_params=None,
                 file_path=None, ) -> None:
        
        if file_path is not None:
            print("Loading Settings from file: {}", file_path)
            system_config = load_settings(file_path)

            self.system_name = system_config['system_name']
            self.filter = system_config['filter']
            self.adc_clock_freq = system_config['adc_clock_freq']
            self.start = system_config['start']
            self.stop = system_config['stop']
            self.spacing = system_config['spacing']
            self.noise = system_config['noise']
            self.processing_systems = system_config['processing_systems']
            self.LO_params = system_config['LO_params']
        
        else:
            self.system_name = system_name
            self.filter = filter
            self.adc_clock_freq = adc_clock_freq
            self.start = start
            self.stop = stop
            self.spacing = spacing
            self.noise = noise
            self.processing_systems = processing_systems
            self.LO_params = LO_params
            self.wave_params = wave_params

        self.sampled_t = None
        self.sampled_tf = None
        self.LO = None
        self.LO_modulation = None
        self.__create_real_time()

    def __create_real_time(self):
        if ( self.filter == 'integrate' ):
            self.start = self.start - self.filter['window_size'] * self.spacing
        total_time = abs( self.start - self.stop )
        points_per_second = round(1/self.spacing)

        #adjusting points_per_second to be evenly divisible by adc_clock_freq
        band = int( points_per_second / self.adc_clock_freq )
        band_remainder = int(points_per_second % self.adc_clock_freq)
        if ( band_remainder != 0 ):
            points_per_second -= band_remainder
        # K_band must be an even number
        if ( band % 2 != 0 ):
            points_per_second += int( self.adc_clock_freq )
        self.points_per_second = points_per_second
        self.adj_spacing = 1/points_per_second

        self.num_time_points = int ( total_time * points_per_second )
        self.t = np.linspace(self.start, self.stop, self.num_time_points, endpoint=False)
        self.tf = np.linspace(-1/(2*self.adj_spacing), 1/(2*self.adj_spacing), int(self.t.size), endpoint=False)
        self.adc_clock_ticks = int(points_per_second / self.adc_clock_freq)
        self.K_band = round( self.num_time_points*self.adj_spacing*self.adc_clock_freq )
        self.Zones = int( self.num_time_points/self.K_band )

    def generate_LO(self):
        if self.LO_params is None:
            self.set_LO_params()

        self.LO_modulation = (self.LO_params['phase_delta']/self.LO_params['phase_freq'])* \
            sin(2*pi*self.LO_params['phase_freq']*self.t+self.LO_params['phase_offset'])
        
        self.LO = self.LO_params['amp']*sin(2*pi*self.LO_params['freq']*self.t+self.LO_params['phase']+self.LO_modulation)
    
    def generate_rising_zero_crossings(self):
        if self.LO is None:
            self.generate_LO()
        #zero-crossing pulse generator 
        zero_crossings = np.where(np.diff(np.signbit(self.LO)))[0] + 1
        rising_zero_crossings = np.zeros_like(self.t)

        #Testing for edge of time window cases
        start = self.t[0]-self.adj_spacing
        end = self.t[0]
        start_LO_mod = (self.LO_params['phase_delta']/self.LO_params['phase_freq'])* \
            sin(2*pi*self.LO_params['phase_freq']*start+self.LO_params['phase_offset'])
        start_LO = self.LO_params['amp']*sin(2*pi*self.LO_params['freq']*start+self.LO_params['phase']+start_LO_mod)
        end_LO_mod = (self.LO_params['phase_delta']/self.LO_params['phase_freq'])* \
            sin(2*pi*self.LO_params['phase_freq']*end+self.LO_params['phase_offset'])
        end_LO = self.LO_params['amp']*sin(2*pi*self.LO_params['freq']*end+self.LO_params['phase']+end_LO_mod)
        if ( start_LO*end_LO < 0 and start_LO < end_LO ):
            rising_zero_crossings[0] = 1
        for i in zero_crossings:
            if (self.LO[i] > self.LO[i-1]):
                rising_zero_crossings[i] = 1

        return rising_zero_crossings

    def create_input_signal(self, wave_params=None, file_path=None):
        if ( file_path is not None ):
            print("Loading Settings from file: {}", file_path)
            system_config = load_settings(file_path)
            wave_params = system_config['wave_params']
        elif ( wave_params is None ):
            print("No wave_params provided. Adding new wave parameter")
            add_wave_param = True
            while add_wave_param:
                amp = float(input("Enter amplitude: "))
                freq = float(input("Enter frequency: "))
                phase = float(input("Enter phase: "))
                wave_params.append({"amp": amp, "freq": freq, "phase": phase})
                add_wave_param = input("Add another wave parameter? (y/n): ").lower() == 'y'
        
        x_input = np.zeros_like(self.t)
        n_non_zero_amps = 0

        for param in wave_params:
            if ( param['amp'] != 0 ):
                n_non_zero_amps += 1
            new_signal = param['amp']*sin(2*pi*param['freq']*(self.t+param['phase']))
            x_input = np.add(x_input,new_signal)

        if ( self.noise != 0 ):
            noise = np.random.normal(0, self.noise, x_input.size)
            x_input = x_input + noise

        return [ x_input, n_non_zero_amps ]

    def filter_signal(self, data):
        if self.filter_params == None:
            print("No filter parameters provided.")
            self.set_filter_params()
        if (self.filter == 'butter'):
            system_nyquist_rate = 1/self.adj_spacing
            sos = butter(self.filter_params['order'], self.filter_params['cutoff_freq'], fs=system_nyquist_rate, btype='lowpass', analog=False, output='sos')
            w, filt_freq = sosfreqz(sos, worN=self.t.size, whole=True)
            filt_freq_down = np.concatenate((filt_freq[:self.adc_clock_ticks],filt_freq[-self.adc_clock_ticks:]))
            filtered_data = sosfilt(sos, data)
        else:
            filtered_data = np.zeros_like(self.t, dtype=np.complex_)
            for i in range(self.t.size):
                t_window = np.copy(self.t[i:i+self.filter_params['window_size']])
                data_window = np.copy(data[i:i+self.filter_params['window_size']])
                filtered_data = trapezoid(data_window, t_window)
            filt_freq_down = None

        return [ filtered_data, filt_freq_down ]
    
    def sample_signals(self, data=None):
        sampled_data_list = []
        sampled_time_list = []
        for i in range(0, self.t.size, self.adc_clock_ticks):
            if data is not None:
                sampled_data.append(data[i])
            sampled_time_list.append(self.t[i])

        self.sampled_t = np.array(sampled_time_list)
        self.sampled_tf = np.linspace(-1/(2*self.adj_spacing*self.adc_clock_ticks), 1/(2*self.adj_spacing*self.adc_clock_ticks), len(sampled_time_list), endpoint=False)
        sampled_data = np.array(sampled_data_list)

        return sampled_data
    
    def create_dict(self, filt_down=None):
        R_init = np.eye(self.K_band)
        dft_matrix = dft(self.K_band)
        idft_norm = np.transpose(np.conjugate(dft_matrix))/(self.num_time_points)
        if filt_down is not None:
            filt_freq_down = fft(filt_down)
            dft_matrix_adj = np.matmul(filt_freq_down * R_init, dft_matrix)

        M_index = []
        M_temp = [0,0]
        M_pattern = [0, 1]
        for i in range(0,int(self.Zones/2)):
            M_temp[0] = M_pattern[0] + i
            M_temp[1] = -( M_pattern[1] + i )
            M_index = M_index + M_temp
    
        R_init = np.eye(self.K_band)
        R = np.copy(R_init)

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

    
    def set_LO_params(self):
        amp = float(input("Enter LO amplitude: "))
        freq = float(input("Enter LO frequency: "))
        phase = float(input("Enter LO phase: "))
        phase_delta = float(input("Enter LO phase delta: "))
        phase_freq = float(input("Enter LO phase frequency: "))
        phase_offset = float(input("Enter LO phase offset: "))
        self.LO_params = {
            "amp": amp,
            "freq": freq,
            "phase": phase,
            "phase_delta": phase_delta,
            "phase_freq": phase_freq,
            "phase_offset": phase_offset
        }

    def set_filter_params(self):
        type = input("Enter filter type (butter/integrate): ")
        order = int(input("Enter filter order: "))
        cutoff_freq = float(input("Enter cutoff frequency: "))
        angle = float(input("Enter filter angle: "))
        window_size = int(input("Enter filter window size: "))
        self.filter_params = {
            "type": type,
            "order": order,
            "cutoff_freq": cutoff_freq,
            "angle": angle,
            "window_size": window_size
        }

    def get_system_name(self):
        return self.system_name
    def get_filter(self):
        return self.filter
    def get_adc_clock_freq(self):
        return self.adc_clock_freq
    def get_start(self):
        return self.start
    def get_ringing_zero_crossings(self):
        return self.rising_zero_crossings
    def get_real_time(self):
        return self.t
    def get_points_per_second(self):
        return self.points_per_second
    def get_adjusted_spacing(self):
        return self.adj_spacing
    def get_K_band(self):
        return self.K_band
    def get_num_time_points(self):
        return self.num_time_points


def simulate_system(wave_params, eps, LO_params, system_params, psi_params, filter_params, manager_queue):
    x, t, num_tones = multi_tone_sine_wave(system_params, wave_params, filter_params)
    xf = fft(x)
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
    # coef = recover_signal(dictionary, y_sampled, system_params, num_tones)
    coef = recover_signal(dictionary, y_sampled, system_params)
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

def create_nyfr_dict(t, LO_modulation, LO_mod_sampled, filt_down, system_params, tf_sampled):
    # K_band = t.size*(system_params['spacing']*system_params['adc_clock_freq'])
    # K_band = ceil(t.size*(system_params['spacing']*system_params['adc_clock_freq']))
    adc_freq = system_params['adc_clock_freq']
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
    # idft_norm = np.transpose(np.conjugate(dft_matrix))/(100*K_band)
    idft_norm = np.transpose(np.conjugate(dft_matrix))/(2*Zones*K_band)
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
