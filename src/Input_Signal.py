from utility import load_settings
import numpy as np

class Input_Signal:
    def __init__(self,
                 time_params=None,
                 adc_params=None,
                 env_params=None,
                 wave_params=None,
                 input_config_name=None,
                 config_file_path=None) -> None:
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_time_params(time_params)
            self.set_adc_params(adc_params)
            self.set_env_params(env_params)
            self.set_wave_params(wave_params)
            if ( time_params is None and
                    adc_params is None and
                    env_params is None and
                    wave_params is None):
                input_config_name = "Default_Input_Config"
            self.set_input_config_name(input_config_name)

        self.effects = None
        self.analog = self._create_analog()
        self.input_signal = self._create_input_signal()

    # -------------------------------
    # Setters
    # -------------------------------
    
    def set_config_from_file(self, config_file_path):
        print("Loading input configuration from file: ", config_file_path)
        input_config = load_settings(config_file_path)
        time_params = input_config.get('time_params', None)
        adc_params = input_config.get('adc_params', None)
        env_params = input_config.get('env_params', None)
        wave_params = input_config.get('wave_params', None)
        input_config_name = input_config.get('system_config_name', None)
        
        self.set_time_params(time_params)
        self.set_env_params(env_params)
        self.set_adc_params(adc_params)
        self.set_wave_params(wave_params)
        if ( time_params is None and
                adc_params is None and
                env_params is None and
                wave_params is None):
            input_config_name = "Default_Input_Config"
        self.set_input_config_name(input_config_name)
        
    def set_input_config_name(self, input_config_name=None):
        if input_config_name is None:
            input_config_name = "Input_Config_1"
        self.input_config_name = input_config_name

    def set_time_params(self, time_params=None):
        if time_params is None:
            time_params = {
                'time_range': (0, 1),
                'sim_freq': 1000000
            }
        self.time_params = time_params

    def set_adc_params(self, adc_params=None):
        if adc_params is None:
            adc_params = {
                "adc_samp_freq": 100,
                "allow_clipping": True,
                "v_ref_range": (0, 1),
                "num_bits": 8,
                "thermal_noise_std_dev": 0.0,
                "non_linearity_mode": False,
                "alpha": 0.0,
                "threshold": 1.0,
                "jitter_std": 0.0,
                "acquisition_time_constant": 0.0,
                "hold_noise_std": 0.0
            }           
        self.adc_params = adc_params
        
    def set_env_params(self, env_params=None):
        if env_params is None:
            env_params = {
                "noise_level": 0.0,
                "attenuation": 1.0,
                "doppler": 0.0,
                "delay": 0.0,
                "num_echoes": 0,
                "max_delay": 0.0,
                "max_doppler": 0.0,
                "phase_inversion_prob": 0.0
            }
        self.env_params = env_params

    def set_wave_params(self, wave_params=None):
        if wave_params is None:
            wave_params = {
                "num_waves": 0,
                "freq_range": (100, 1000),
                "amp_range": (0.1, 1.0),
                "phase_random": True,
                "waves": [
                    {"amp": 0.5,
                    "freq": 4,
                    "phase": 0}
                ]
            }
        self.wave_params = wave_params
        
    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def _create_analog(self):
        analog = {}
        points_per_second = round(self.time_params['sim_freq'])
        # Adjust to be evenly divisible by adc_clock_freq
        band = int(points_per_second / self.adc_params['adc_samp_freq'])
        band_remainder = int(points_per_second % self.adc_params['adc_samp_freq'])
        if band_remainder != 0:
            points_per_second -= band_remainder
        # K_band must be even
        if band % 2 != 0:
            points_per_second += int(self.adc_params['adc_samp_freq'])
        time_range = tuple(self.time_params.get('time_range', (0, 1)))
        analog['points_per_second'] = points_per_second
        analog['adj_spacing'] = 1 / points_per_second
        analog['total_time'] = abs(time_range[1] - time_range[0])
        analog['num_points'] = int(analog['total_time'] * points_per_second)
        analog['time'] = np.linspace(time_range[0],
                                  time_range[1],
                                  analog['num_points'],
                                  endpoint=False)
        analog['frequency'] = np.linspace(-points_per_second / 2,
                                   points_per_second / 2,
                                   int(analog['time'].size),
                                   endpoint=False)
        return analog

    def _create_input_signal(self):
        """
        Generate composite signal with environmental effects and random wave generation.
        """
         
        # Override defaults with values inside wave_params if they exist
        num_waves = self.wave_params.get('num_waves', 1)
        freq_range = tuple(self.wave_params.get('freq_range', (100, 1000)))
        amp_range = tuple(self.wave_params.get('amp_range', (0.1, 1.0)))
        phase_random = self.wave_params.get('phase_random', True)
        allow_clipping = self.adc_params.get('allow_clipping', True)
        v_ref_range = tuple(self.adc_params.get('v_ref_range', (0, 1)))

        # === Wave parameter setup ===
        if 'waves' not in self.wave_params or not self.wave_params['waves']:
            amps = np.random.uniform(amp_range[0], amp_range[1], num_waves)
            freqs = np.random.uniform(freq_range[0], freq_range[1], num_waves)
            if phase_random:
                t_shift = np.random.uniform(0, 1/freqs)  # seconds
                phases = 2 * np.pi * freqs * t_shift
            else:
                phases = np.zeros(num_waves)
            # Save generated wave dictionaries into wave_params
            self.wave_params['waves'] = [
                {"amp": float(amps[i]), "freq": float(freqs[i]), "phase": float(phases[i])}
                for i in range(num_waves)
            ]
        waves = self.wave_params['waves']

        input_signal = self._generate_signal(waves)
            # --- Adjust to midpoint ---
        midpoint = (v_ref_range[1] + v_ref_range[0]) / 2
        input_signal = input_signal + midpoint  # shift signal to midpoint

        # --- Scale signal ---
        if not allow_clipping:
            # Prevent clipping by scaling amplitude
            max_abs_val = max(abs(input_signal.min() - midpoint), abs(input_signal.max() - midpoint))
            max_allowed_amp = (v_ref_range[1] + v_ref_range[0]) / 2
            if max_abs_val > 0:
                scale = max_allowed_amp / max_abs_val
                input_signal = (input_signal - midpoint) * scale + midpoint
                
        return input_signal

    def _generate_signal(self, waves):
        self.effects = {}
        amps = np.array([wave['amp'] for wave in waves])
        freqs = np.array([wave['freq'] for wave in waves])
        phases = np.array([wave['phase'] for wave in waves])
            # === Apply system-level Doppler and delay ===
        doppler = self.env_params.get('doppler', 0.0)
        freqs = freqs * (1 + doppler)

        delay = self.env_params.get('delay', 0.0)
        self.analog['time'] = self.analog['time'] + delay

        # === Generate base composite signal ===
        signals = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * self.analog['time'] + phases[:, None])
        signal = np.sum(signals, axis=0)

        # === Environmental attenuation ===
        attenuation = self.env_params.get('attenuation', 1.0)
        signal *= attenuation

        # === Add Gaussian noise ===
        noise_level = self.env_params.get('noise_level', 0.0)
        self.effects['noise'] = 0.0
        if noise_level > 0:
            self.effects['noise'] = np.random.normal(0, noise_level, signal.shape)
            signal += self.effects['noise']

        # === Multipath reflections (with Doppler + phase inversion) ===
        num_echoes = self.env_params.get('num_echoes', 0)
        self.effects['delay'] = []
        self.effects['echo_att'] = []
        self.effects['local_doppler'] = []
        self.effects['phase_inversion'] = []
        if num_echoes > 0:
            dt = self.analog['time'][1] - self.analog['time'][0]
            max_delay = self.env_params.get('max_delay', 0.01)
            max_doppler = self.env_params.get('max_doppler', 0.002)
            phase_inversion_prob = self.env_params.get('phase_inversion_prob', 0.5)

            for _ in range(num_echoes):
                # Random propagation delay
                phase_inversion = False
                delay = np.random.uniform(0, max_delay)
                shift = int(delay / dt)

                # Random attenuation
                echo_att = np.random.uniform(0.2, 0.8)

                # Random Doppler shift
                local_doppler = 1 + np.random.uniform(-max_doppler, max_doppler)
                real_time_echo = self.analog['time'] * local_doppler
                echo = np.interp(self.analog['time'], real_time_echo, signal, left=0, right=0) * echo_att

                # Optional phase inversion
                if np.random.rand() < phase_inversion_prob:
                    phase_inversion = True
                    echo = -echo

                # Apply time delay
                echo = np.roll(echo, shift)
                echo[:shift] = 0
                signal += echo
                self.effects['delay'].append(delay)
                self.effects['echo_att'].append(echo_att)
                self.effects['local_doppler'].append(local_doppler)
                self.effects['phase_inversion'] = phase_inversion

        return signal

    # -------------------------------
    # Getters
    # -------------------------------

    def get_input_signal(self):
        return self.input_signal
    
    def get_input_config_name(self):
        return self.input_config_name
    
    def get_analog_signals(self):
        return self.analog
    
    def get_analog_signal_params(self):
        exclude = ["time", "frequency"]
        analog_signal_params = {k: v for k, v in self.analog.items() if k not in exclude}
        return analog_signal_params
    
    def get_analog_time(self):
        return self.analog['time']
    
    def get_analog_frequency(self):
        return self.analog['frequency']
    
    def get_time_params(self):
        return self.time_params
    
    def get_adc_params(self):
        return self.adc_params
    
    def get_env_params(self):
        return self.env_params
    
    def get_wave_params(self):
        return self.wave_params
    
    def get_effects(self):
        return self.effects