from utility import load_settings
import numpy as np
from numpy import sin
from math import e, pi
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.integrate import trapezoid
from scipy.fftpack import fft
from scipy.linalg import dft

class NYFR:
    def __init__(self, 
                 filter_params=None,
                 system_params=None,
                 time_params=None,
                 system_config_name=None,
                 dictionary_params=None, 
                 LO_params=None, 
                 wave_params=None,
                 file_path=None, ) -> None:
        
        if file_path is not None:
            print("Loading Initial Settings from file: ", file_path)
            system_config = load_settings(file_path)
            self.system_config_name = system_config['system_config_name']
            self.system_params = system_config['system_params']
            self.time_params = system_config['time_params']
            self.filter_params = system_config['filter_params']
            self.LO_params = system_config['LO_params']
            self.dictionary_params = system_config['dictionary_params']
        
        else:
            self.system_config_name = system_config_name
            self.system_params = system_params
            self.time_params = time_params
            self.filter_params = filter_params
            self.LO_params = LO_params
            self.wave_params = wave_params
            self.dictionary_params = dictionary_params

        self.points_per_second = 0
        self.adj_spacing = 0
        self.num_time_points = 0
        self.adc_clock_ticks = 0
        self.K_band = 0
        self.Zones = 0
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

    def set_real_time(self, time_params=None):
        if time_params is not None or self.time_params is None:
            if self.time_params is None:
                print("No time parameters provided during initialization. Adding new time parameters")
            else:
                print("Adding new time parameters")
            self.set_time_params(time_params)

        if ( self.filter_params['type'] == 'integrate' ):
            self.time_params['start'] = self.time_params['start'] - self.filter_params['window_size'] * self.time_params['spacing']

        self.__set_points_per_second()

        self.adj_spacing = 1/self.points_per_second
        total_time = abs( self.time_params['start'] - self.time_params['stop'] )
        self.num_time_points = int ( total_time * self.points_per_second )
        self.t = np.linspace(self.time_params['start'], self.time_params['stop'], self.num_time_points, endpoint=False)
        self.tf = np.linspace(-1/(2*self.adj_spacing), 1/(2*self.adj_spacing), int(self.t.size), endpoint=False)
        self.adc_clock_ticks = int(self.points_per_second / self.system_params['adc_clock_freq'])
        self.K_band = round( self.num_time_points*self.adj_spacing*self.system_params['adc_clock_freq'] )
        self.Zones = int( self.num_time_points/self.K_band )

    def __set_points_per_second(self):
        points_per_second = round(1/self.time_params['spacing'])
        #adjusting points_per_second to be evenly divisible by adc_clock_freq
        band = int( points_per_second / self.system_params['adc_clock_freq'] )
        band_remainder = int(points_per_second % self.system_params['adc_clock_freq'])
        if ( band_remainder != 0 ):
            points_per_second -= band_remainder
        # K_band must be an even number
        if ( band % 2 != 0 ):
            points_per_second += int( self.system_params['adc_clock_freq'] )

        self.points_per_second = points_per_second

    def set_LO(self):
        if self.LO_params is None:
            self.set_LO_params()

        self.LO_modulation = (self.LO_params['phase_delta']/self.LO_params['phase_freq'])* \
            sin(2*pi*self.LO_params['phase_freq']*self.t+self.LO_params['phase_offset'])
        
        self.LO = self.LO_params['amp']*sin(2*pi*self.LO_params['freq']*self.t+self.LO_params['phase']+self.LO_modulation)
    
    def generate_rising_zero_crossings(self):
        if self.LO is None:
            self.set_LO()
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

    def sample_signals(self, data=None, update_sampled_time=False):
        sampled_data_list = []
        sampled_time_list = []
        for i in range(0, self.t.size, self.adc_clock_ticks):
            if data is not None:
                sampled_data_list.append(data[i])
            sampled_time_list.append(self.t[i])
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
        self.set_LO()
        self.rising_zero_crossings = self.generate_rising_zero_crossings()
        self.LO_sampled = self.sample_signals(self.LO, update_sampled_time=True)
        self.LO_modulation_sampled = self.sample_signals(self.LO_modulation)
    
    def simulate_system(self, input_signal=None, wave_params=None, file_path=None):
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
                input_signal, _ = self.create_input_signal(file_path=file_path)
            else:
                input_signal, _ = self.create_input_signal(wave_params=wave_params)

        mixed_input = np.copy(input_signal*self.rising_zero_crossings)
        mixed_input_filtered = self.filter_signal(mixed_input)
        output = self.sample_signals(mixed_input_filtered)
        return output

    def create_input_signal(self, wave_params=None, file_path=None):
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
        
        x_input = np.zeros_like(self.t)
        n_non_zero_amps = 0

        for param in wave_params:
            if ( param['amp'] != 0 ):
                n_non_zero_amps += 1
            new_signal = param['amp']*sin(2*pi*param['freq']*(self.t+param['phase']))
            x_input = np.add(x_input,new_signal)

        if ( self.system_params['system_noise_level'] != 0 ):
            noise = np.random.normal(0, self.noise, x_input.size)
            x_input = x_input + noise

        return [ x_input, n_non_zero_amps ]

    def filter_signal(self, data, filter_params=None):
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
            filtered_data = np.zeros_like(self.t, dtype=np.complex_)
            for i in range(self.t.size):
                t_window = np.copy(self.t[i:i+self.filter_params['window_size']])
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
            M_temp[1] = -( M_pattern[1] + i )
            M_index = M_index + M_temp
        
        return M_index

    def __create_complex_dict(self, R_init=None, M_index=None, dft_matrix=None):
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
    
    def __create_real_dict(self, R_init=None, M_index=None, dft_matrix=None):
        idft_norm = np.transpose(np.conjugate(dft_matrix))/(2*self.Zones*self.K_band)

        R = np.copy(R_init)
        for i in (range(2*self.Zones-1)):
            R = np.hstack((R,R_init))
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
            if ( i >= int(R_col/2) ):
                PSI_1[i:i+2*self.K_band,int(i/2):int(i/2)+self.K_band] = np.copy(UL_idft_2)
            else:
                PSI_1[i:i+2*self.K_band,int(i/2):int(i/2)+self.K_band] = np.copy(UL_idft_1)

        return R, S, PSI

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
            self.filter_params = dictionary_params

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
    
    def get_real_time(self):
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
        return self.K_band
    
    def get_num_time_points(self):
        return self.num_time_points
    
    def get_LO(self):
        return self.LO
    
    def get_LO_modulation(self):
        return self.LO_modulation
    
    def get_sampled_LO(self):
        return self.LO_sampled