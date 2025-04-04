from utility import load_settings
import numpy as np
from numpy import sin
from math import e, pi
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.integrate import trapezoid
from scipy.fftpack import fft
from scipy.linalg import dft
from NYFR_ML_Models import model_prediction
from sklearn.linear_model import orthogonal_mp
from OMP import OMP
from spgl1 import spgl1

class NYFR:
    def __init__(self, 
                 filter_params=None,
                 system_params=None,
                 time_params=None,
                 system_config_name=None,
                 dictionary_params=None, 
                 LO_params=None, 
                 recovery_params=None,
                 file_path=None, ) -> None:
        
        if file_path is not None:
            print("Loading Initial Settings from file: ", file_path)
            system_config = load_settings(file_path)
            system_config_name = system_config['system_config_name']
            system_params = system_config['system_params']
            time_params = system_config['time_params']
            filter_params = system_config['filter_params']
            LO_params = system_config['LO_params']
            dictionary_params = system_config['dictionary_params']
            recovery_params = system_config['recovery_params']

        self.set_dictionary_params(dictionary_params=dictionary_params)
        self.set_filter_params(filter_params=filter_params)
        self.set_LO_params(LO_params=LO_params)
        self.set_system_config_name(system_config_name=system_config_name)
        self.set_time_params(time_params=time_params)
        self.set_system_params(system_params=system_params)
        self.set_recovery_params(recovery_params=recovery_params)

        self.points_per_second = 0
        self.adj_spacing = 0
        self.num_time_points = 0
        self.adc_clock_ticks = 0
        self.total_time = 0
        self.K_band = 0
        self.Zones = 0
        self.real_t = None
        self.t = None
        self.tf = None
        self.sampled_t = None
        self.sampled_tf = None
        self.LO = None
        self.LO_sampled = None
        self.LO_modulation = None
        self.LO_modulation_sampled = None
        self.rising_zero_crossings = None
        self.sos = None
        self.real_LO = None

    def set_real_time(self, time_params=None):
        if time_params is not None or self.time_params is None:
            if self.time_params is None:
                print("No time parameters provided during initialization. Adding new time parameters")
            else:
                print("Adding new time parameters")
            self.set_time_params(time_params)

        if ( self.filter_params['type'] == 'integrate' ):
            # self.time_params['start'] = self.time_params['start'] - self.filter_params['window_size'] * self.time_params['spacing']
            self.time_params['start'] = self.time_params['start'] - ( self.filter_params['window_size'] / self.time_params['sim_freq'] )

        points_per_second = round(self.time_params['sim_freq'])
        #adjusting points_per_second to be evenly divisible by adc_clock_freq
        band = int( points_per_second / self.system_params['adc_clock_freq'] )
        band_remainder = int(points_per_second % self.system_params['adc_clock_freq'])
        if ( band_remainder != 0 ):
            points_per_second -= band_remainder
        # K_band must be an even number
        if ( band % 2 != 0 ):
            points_per_second += int( self.system_params['adc_clock_freq'] )

        self.real_points_per_second = points_per_second
        # self.__set_points_per_second()

        self.adj_spacing = 1/self.real_points_per_second
        self.total_time = abs( self.time_params['start'] - self.time_params['stop'] )
        self.num_real_time_points = int ( self.total_time * self.real_points_per_second )
        self.real_t = np.linspace(self.time_params['start'], self.time_params['stop'], self.num_real_time_points, endpoint=False)
        self.real_tf = np.linspace(-self.real_points_per_second/2, self.real_points_per_second/2, int(self.real_t.size), endpoint=False)
        # self.t = np.linspace(self.time_params['start'], self.time_params['stop'], self.num_time_points, endpoint=False)
        # self.tf = np.linspace(-1/(2*self.adj_spacing), 1/(2*self.adj_spacing), int(self.t.size), endpoint=False)
        # self.adc_clock_ticks = int(self.points_per_second / self.system_params['adc_clock_freq'])
        # self.K_band = round( self.num_time_points*self.adj_spacing*self.system_params['adc_clock_freq'] )
        # self.Real_K_band = None
        # self.Zones = int( self.num_time_points/self.K_band )
        # self.Real_Zones = None
        # if self.dictionary_params['type'] == 'real':
        #     self.Real_Zones = 2 * self.Zones
        #     self.Real_K_band = round(self.K_band / 2)

    # def __set_points_per_second(self):
    #     points_per_second = round(self.time_params['sim_freq'])
    #     #adjusting points_per_second to be evenly divisible by adc_clock_freq
    #     band = int( points_per_second / self.system_params['adc_clock_freq'] )
    #     band_remainder = int(points_per_second % self.system_params['adc_clock_freq'])
    #     if ( band_remainder != 0 ):
    #         points_per_second -= band_remainder
    #     # K_band must be an even number
    #     if ( band % 2 != 0 ):
    #         points_per_second += int( self.system_params['adc_clock_freq'] )

    #     self.points_per_second = points_per_second

    def set_LO(self, t=None):
        if t is None:
            t = self.t
        if self.LO_params is None:
            self.set_LO_params()

        self.LO_modulation = (self.LO_params['phase_delta']/self.LO_params['phase_freq'])* \
            sin(2*pi*self.LO_params['phase_freq']*t+self.LO_params['phase_offset'])
        
        self.LO = self.LO_params['amp']*sin(2*pi*self.LO_params['freq']*t+self.LO_params['phase']+self.LO_modulation)
    
    def generate_rising_zero_crossings(self, t=None):
        if t is None:
            t = self.t
        if self.LO is None:
            self.set_LO()
        #zero-crossing pulse generator 
        zero_crossings = np.where(np.diff(np.signbit(self.LO)))[0] + 1
        rising_zero_crossings = np.zeros_like(t)

        #Testing for edge of time window cases
        start = t[0]-self.adj_spacing
        end = t[0]
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

    def sample_signals(self, data=None, update_sampled_time=False, sample_rate=None, points_per_second=None, t=None):
        if points_per_second is None:
            points_per_second = self.points_per_second
        if sample_rate is None:
            sample_rate = self.system_params['adc_clock_freq']
        if t is None:
            t = self.t
        clock_ticks = int(points_per_second / sample_rate)
        sampled_data_list = []
        sampled_time_list = []
        for i in range(0, t.size, clock_ticks):
            if data is not None:
                sampled_data_list.append(data[i])
            sampled_time_list.append(t[i])
        if update_sampled_time:
            self.sampled_t = np.array(sampled_time_list)
            self.sampled_tf = np.linspace(-1/(2*self.adj_spacing*self.adc_clock_ticks),
                                        1/(2*self.adj_spacing*self.adc_clock_ticks),
                                        len(sampled_time_list),
                                        endpoint=False)
        sampled_data = np.array(sampled_data_list)

        return sampled_data
    
    def initialize(self, system_params=None):
        if system_params is not None:
            self.set_system_params(system_params)
        self.set_real_time()
        self.set_LO(t=self.real_t)
        self.rising_zero_crossings = self.generate_rising_zero_crossings(t=self.real_t)
        self.wb_nyquist_rate = int ( self.system_params['wbf_cut_freq'] * 2 )
        self.t = self.sample_signals(data=self.real_t,
                                     points_per_second=self.real_points_per_second,
                                     sample_rate=self.wb_nyquist_rate,
                                     t=self.real_t)
        self.tf = self.sample_signals(data=self.real_tf,
                                     points_per_second=self.real_points_per_second,
                                     sample_rate=self.wb_nyquist_rate,
                                     t=self.real_tf)
        self.num_time_points = self.t.size
        self.points_per_second = int( self.num_time_points / self.total_time )
        self.adc_clock_ticks = int(self.points_per_second / self.system_params['adc_clock_freq'])
        self.K_band = round( self.num_time_points*self.system_params['adc_clock_freq']/(self.points_per_second) )
        self.Real_K_band = None
        self.Zones = int( self.num_time_points/self.K_band )
        self.Real_Zones = None
        if self.dictionary_params['type'] == 'real':
            self.Real_Zones = 2 * self.Zones
            self.Real_K_band = round(self.K_band / 2)
    
    def simulate_system(self, input_signal=None, wave_params=None, file_path=None, t=None):
        if t is None:
            t = self.real_t

        if input_signal is None:
            print("No input signal provided. Creating Input Signal")
            if wave_params is None and file_path is None:
                print("No wave parameters provided. Adding new wave parameters")
                add_wave_param = True
                wave_params = []
                while add_wave_param:
                    amp = float(input("Enter amplitude: "))
                    freq = float(input("Enter frequency: "))
                    phase = float(input("Enter phase: "))
                    wave_params.append({"amp": amp, "freq": freq, "phase": phase})
                    add_wave_param = input("Add another wave parameter? (y/n): ").lower() == 'y'
            if file_path is not None:
                input_signal, _ = self.create_input_signal(file_path=file_path, t=t)
            else:
                input_signal, _ = self.create_input_signal(wave_params=wave_params, t=t)
        rising_zero_crossings = self.generate_rising_zero_crossings(t=t)
        mixed_input = np.copy(input_signal*rising_zero_crossings)
        mixed_input_filtered = self.filter_signal(mixed_input, t=t)
        output = self.sample_signals(mixed_input_filtered)
        self.real_LO = self.LO
        if not self.time_params['save_real_time']:
            self.clear_real_time_sigs()

        LO = self.sample_signals(data=self.LO,
                                 points_per_second=self.real_points_per_second,
                                 sample_rate=self.wb_nyquist_rate,
                                 t=t)
        self.LO_sampled = self.sample_signals(LO, update_sampled_time=True)
        self.LO_modulation_sampled = self.sample_signals(self.LO_modulation)
        return output
    
    def clear_real_time_sigs(self):
        self.real_LO = None
        self.real_t = None
        self.real_tf = None

    def create_input_signal(self, wave_params=None, file_path=None, t=None):
        if t is None:
            t = self.real_t
        if ( file_path is not None ):
            print("Loading Wave Settings from file: ", file_path)
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
        
        x_input = np.zeros_like(t)
        n_non_zero_amps = 0

        for param in wave_params:
            if ( param['amp'] != 0 ):
                n_non_zero_amps += 1
            new_signal = param['amp']*sin(2*pi*param['freq']*(t+param['phase']))
            x_input = np.add(x_input,new_signal)

        if ( self.system_params['system_noise_level'] != 0 ):
            noise = np.random.normal(0, self.noise, x_input.size)
            x_input = x_input + noise

        return [ x_input, n_non_zero_amps ]

    def filter_signal(self, data, filter_params=None, t=None):
        if t is None:
            t = self.t
        if self.filter_params is None and filter_params is None:
            print("No filter parameters provided.")
            self.set_filter_params()
        if (self.filter_params['type'] == 'butter'):
            system_nyquist_rate = 1/self.adj_spacing
            self.sos = butter(self.filter_params['order'],
                         self.filter_params['cutoff_freq'],
                         fs=system_nyquist_rate, btype='lowpass',
                         analog=False,
                         output='sos')
            filtered_data = sosfilt(self.sos, data)
        elif (self.filter_params['type'] == 'integrate'):
            filtered_data = np.zeros_like(t, dtype=np.complex_)
            for i in range(t.size):
                t_window = np.copy(t[i:i+self.filter_params['window_size']])
                data_window = np.copy(data[i:i+self.filter_params['window_size']])
                filtered_data = trapezoid(data_window, t_window)
        else:
            print("Unsupported Filter Type")
            filtered_data = None

        return filtered_data

    def create_dict(self, filt_down=None, dictionary_params=None):
        if self.dictionary_params is None:
            self.set_dictionary_params(dictionary_params=dictionary_params)
        
        dft_init = dft(self.K_band)
        if self.dictionary_params['version'] == 'enhanced':
            if filt_down is not None:
                filt_freq_down = fft(filt_down)
            else:
                if self.filter_params['type'] == 'butter':
                    _, filt_freq_down = self.get_filter_frequency()
                else:
                    filt_freq_down = np.ones_like(self.sampled_t)
            dft_matrix = np.matmul(filt_freq_down * R_init, dft_init)

        elif self.dictionary_params['version'] == 'original':
            dft_matrix = np.copy(dft_init)
        else:
            print("Unknown dictionary version")
            dictionary = None
            return dictionary

        M_index = self.__create_M_pattern()
        R_init = np.eye(self.K_band)

        if self.dictionary_params['type'] == 'complex':
            R, S, PSI = self.__create_complex_dict(R_init=R_init,
                                                    M_index=M_index,
                                                    dft_matrix=dft_matrix)
        elif self.dictionary_params['type'] == 'real':
            R, S, PSI = self.__create_real_dict(R_init=R_init,
                                                M_index=M_index,
                                                dft_matrix=dft_matrix)
        else:
            print("Unknown dictionary type")
            dictionary = None

        dictionary = np.matmul(np.matmul(R,S),PSI)
        return dictionary
    
    def __create_M_pattern(self):
        M_index = []
        M_temp = [0,0]
        M_pattern = [0, 1]
        for i in range(0,int(self.Zones/2)):
            M_temp[0] = M_pattern[0] + i
            M_temp[1] = M_pattern[1] + i
            M_index = M_index + M_temp
        
        return M_index

    def __create_complex_dict(self, R_init, M_index, dft_matrix):
        idft_norm = np.transpose(np.conjugate(dft_matrix))/(self.Zones*self.K_band)
        R = np.copy(R_init)
        for i in (range(self.Zones-1)):
            R = np.hstack((R,R_init))
        R_row, R_col = R.shape 
        S = np.zeros((R_col,R_col),dtype='complex')
        PSI = np.zeros_like(S)
        index = 0 
        for i in range(0, R_col - self.K_band, self.K_band):
            if M_index[index] == 0:
                S[i:i+R_row,i:i+R_row] = np.copy(R_init)
            else:
                S[i:i+R_row,i:i+R_row] = np.copy(e**(M_index[index]*self.LO_modulation[index]*1j)*R_init)
            PSI[i:i+R_row,i:i+R_row] = np.copy(idft_norm)
            index += 1 
        return R, S, PSI
    
    def __create_real_dict(self, R_init, M_index, dft_matrix):
        idft_norm = np.transpose(np.conjugate(dft_matrix))/(2*self.Zones*self.K_band)

        R = np.copy(R_init)
        for i in (range(2*self.Zones-1)):
            R = np.hstack((R,R_init))
        R_row, R_col = R.shape

        S = np.zeros((R_col,R_col),dtype='complex')
        PSI = np.zeros((R_col, int(R_col/2)),dtype='complex')
        
        idft_split = np.hsplit(idft_norm,2)
        zero_fill = np.zeros_like(idft_split[0])
        U_idft = np.hstack((idft_split[0], zero_fill))
        L_idft = np.hstack((zero_fill, idft_split[1]))
        UL_idft = np.vstack((U_idft,L_idft))

        M_index_reverse = [i * -1 for i in M_index]
        M_index_reverse.reverse()
        double_M_index = (M_index + M_index_reverse).copy()

        LO_list = []
        for _ in (range(0,self.Zones)):
            LO_list.append(self.LO_modulation_sampled)
        LO_mod_concat = np.concatenate(LO_list)
        double_LO_modulation = np.concatenate((LO_mod_concat,LO_mod_concat))

        for index,i in enumerate(range(0, R_col, self.K_band)):
            LO_mod = double_LO_modulation[i:i+R_row]
            S[i:i+R_row,i:i+R_row] = np.diag(e**(double_M_index[index]*LO_mod*1j))

        for i in range (0, R_col, 2*self.K_band):
            PSI[i:i+2*self.K_band,int(i/2):int(i/2)+self.K_band] = np.copy(UL_idft)

        return R, S, PSI

    def recover_signal(self, dictionary, output,
                       file_path=None,
                       aux_file_path=None, 
                       num_tones=0, 
                       sigma=0.01, 
                       mode="mag_ang",
                       mlp_inv_mod=0.01):
        output_norm = np.linalg.norm(output)
        coef_real = 0
        coef_imag = 0
        match self.recovery_params['type']:
            case 'OMP':
                coef_real = orthogonal_mp(np.abs(dictionary),output/output_norm)
                # coef_imag = orthogonal_mp(np.imag(dictionary),y_sampled/y_sampled_norm,n_nonzero_coefs=(2*num_tones))
            case 'OMP_Custom':
                coef_real = OMP(dictionary,output/output_norm)[0]
                # coef_imag = orthogonal_mp(np.imag(dictionary),y_sampled/y_sampled_norm,n_nonzero_coefs=(2*num_tones))
            case 'SPGL1':
                # coef_real,resid,grad,info = spgl1(dictionary,y_sampled/y_sampled_norm)
                coef_real,_,_,_ = spgl1(dictionary,output/output_norm,sigma=sigma)
                #coef_real,resid_real,grad_real,info_real = spgl1(np.real(dictionary),y_sampled/y_sampled_norm)
                #coef_imag,resid_imag,grad_imag,info_imag = spgl1(np.imag(dictionary),y_sampled/y_sampled_norm)
            case 'MLP1':
                pseudo = np.linalg.pinv(mlp_inv_mod*dictionary)
                init_guess = np.dot(pseudo,output)
                coef_real, coef_imag = model_prediction(init_guess, file_path, mode, aux_file_path)
            case _:
                print("No recovery performed")
        coef = coef_real + coef_imag*1j
        return coef
    
    def set_system_params(self, system_params=None):
        if system_params is None:
            print("No system parameters provided. Adding new system parameters")
            adc_clock_freq = int(input("Enter ADC clock frequency: "))
            system_noise_level = float(input("Enter system noise level: "))
            add_processing_systems = True
            processing_systems = []
            while add_processing_systems:
                processing_systems.append(input("Enter processing systems: "))
                add_processing_systems = input("Add another processing system? (y/n): ").lower() == 'y'
            self.system_params = {
                "adc_clock_freq": adc_clock_freq,
                "processing_systems": processing_systems,
                "system_noise_level": system_noise_level,
            }
        else:
            self.system_params = system_params

    def set_time_params(self, time_params=None):
        if time_params is None:
            print("No time parameters provided. Adding new time parameters")
            start = float(input("Enter start time: "))
            stop = float(input("Enter stop time: "))
            spacing = float(input("Enter time spacing: "))
            self.time_params = {
                "start": start,
                "stop": stop,
                "spacing": spacing,
            }
        else:
            self.time_params = time_params

    def set_system_config_name(self, system_config_name=None):
        if system_config_name is None:
            print("No system configuration name provided. Adding new name")
            self.system_config_name = input("Enter system configuration name: ")
        else:
            self.system_config_name = system_config_name

    def set_recovery_params(self, recovery_params=None):
        if recovery_params is None:
            print("No Recovery parameters provided. Adding new recovery parameters")
            recovery_type = float(input("Enter recovery type (OMP/OMP_Custom/SPGL1/MLP1): "))
            add_recovery_mode = True
            recovery_modes = []
            total_recovery_modes = 0
            while add_recovery_mode and total_recovery_modes < 3:
                recovery_modes.append(input("Enter recovery mode (mag_ang, real_imag, complex): "))
                add_recovery_mode = input("Add another recovery mode? (y/n): ").lower() == 'y'
                total_recovery_modes += 1
            self.recovery_params = {
                "type": recovery_type,
                "modes": recovery_modes
            }
        else:
            self.recovery_params = recovery_params           

    def set_LO_params(self, LO_params=None):
        if LO_params is None:
            print("No LO_params provided. Adding new LO parameters")
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
        else:
            self.LO_params = LO_params

    def set_filter_params(self, filter_params=None):
        if filter_params is None:
            print("No filter parameters provided. Adding new filter parameters")
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
        else:
            self.filter_params = filter_params

    def set_dictionary_params(self, dictionary_params=None):
        if dictionary_params is None:
            print("No dictionary parameters provided. Adding new dictionary parameters")
            type = input("Enter dictionary type (real/complex): ")
            version = int(input("Enter dictionary version: "))
            self.dictionary_params = {
                "type": type,
                "version": version,
            }
        else:
            self.dictionary_params = dictionary_params

    def get_dictionary_params(self):
        return self.dictionary_params

    def get_recovery_params(self):
        return self.recovery_params

    def get_filter_frequency(self):
        if self.sos is not None:
            _, filt_freq = sosfreqz(self.sos, worN=self.t.size, whole=True)
            filt_freq_down = np.concatenate((filt_freq[:self.adc_clock_ticks],filt_freq[-self.adc_clock_ticks:]))
        else:
            if self.t is None:
                filt_freq = None
            else:
                filt_freq = np.zeros_like(self.t)

            if self.sampled_t is None:
                filt_freq_down = None
            else:
                filt_freq_down = np.zeros_like(self.sampled_t)

        return filt_freq, filt_freq_down
    
    def get_system_config_name(self):
        return self.system_config_name
    
    def get_filter_params(self):
        return self.filter
    
    def get_system_params(self):
        return self.system_params
    
    def get_time_params(self):
        return self.time_params
    
    def get_LO_params(self):
        return self.LO_params
    
    def get_ringing_zero_crossings(self):
        return self.rising_zero_crossings
    
    def get_real_LO(self):
        return self.real_LO

    def get_real_time(self):
        return self.real_t
    
    def get_real_frequncy_bins(self):
        return self.real_tf   

    def get_time(self):
        return self.t

    def get_frequncy_bins(self):
        return self.tf
    
    def get_sampled_time(self):
        return self.sampled_t
    
    def get_sampled_freq_bins(self):
        return self.sampled_tf
    
    def get_points_per_second(self):
        return self.points_per_second
    
    def get_adjusted_spacing(self):
        return self.adj_spacing
    
    def get_K_band(self):
        if self.Real_K_band is None:
            return self.K_band
        else:
            return self.Real_K_band

    def get_num_time_points(self):
        return self.num_time_points
    
    def get_LO(self):
        return self.LO
    
    def get_LO_modulation(self):
        return self.LO_modulation
    
    def get_sampled_LO(self):
        return self.LO_sampled
    
    def get_Zones(self):
        if self.Real_Zones is None:
            return self.Zones
        else:
            return self.Real_Zones
        
    def get_wb_nyquist_rate(self):
        return self.wb_nyquist_rate