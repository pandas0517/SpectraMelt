from .utils import load_config_from_json, get_logger
import numpy as np

class InputSignal:
    def __init__(self,
                 input_params=None,
                 time_params=None,
                 adc_params=None,
                 env_params=None,
                 wave_params=None,
                 log_params=None,
                 config_name=None,
                 config_file_path=None) -> None:
        if config_file_path is not None:
            input_params = load_config_from_json(config_file_path)
        elif input_params is None:
            input_params = {}
            input_params['time_params'] = time_params
            input_params['adc_params'] = adc_params
            input_params['env_params'] =env_params
            input_params['wave_params'] = wave_params
            input_params['config_name'] = config_name
            input_params['log_params'] = log_params
        
        self.set_input_params(input_params)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")
                
        self.effects = None
        self.analog = self.create_analog()
        self.input_signal = self.create_input_signal()

    # -------------------------------
    # Setters
    # -------------------------------
    def set_input_params(self, input_params=None):
        if input_params is None:
            input_params = {}
        
        time_params = input_params.get('time_params', None)
        adc_params = input_params.get('adc_params', None)
        env_params = input_params.get('env_params', None)
        wave_params = input_params.get('wave_params', None)
        log_params = input_params.get('log_params', None)
        config_name = input_params.get('config_name', None)
        if ( time_params is None and
                adc_params is None and
                env_params is None and
                wave_params is None ):
            config_name = "Default_Input_Config"
        else:
            config_name = input_params.get('config_name', "Input_Config_1")
        
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
                
        self.set_time_params(time_params)
        self.set_adc_params(adc_params)
        self.set_env_params(env_params)
        self.set_wave_params(wave_params)
        self.set_config_name(config_name)
        
    def set_config_name(self, config_name):
        self.config_name = config_name
        
    def set_log_params(self, log_params=None):
        if log_params is None:
            log_params = {
                "enabled": True,
                "log_file": None,
                "level": "INFO",
                "console": True
            }
        self.log_params = log_params

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
                "adc_samp_freq": None,
                "v_ref_range": (0, 1),
                "allow_clipping": False
            }           
        self.adc_params = adc_params
        
    def set_env_params(self, env_params=None):
        if env_params is None:
            env_params = {
                "store_internal_sigs": True,
                "noise_level": 0.0,
                "attenuation": 1.0,
                "doppler": 0.0,
                "delay": 0.0,
                "num_echoes": 0,
                "max_delay": 0.0,
                "max_doppler": 0.0,
                "seed": None
            }
        self.rng = np.random.default_rng(env_params.get('seed', None))
        self.env_params = env_params

    def set_wave_params(self, wave_params=None):
        if wave_params is None:
            wave_params = {
                "num_waves": 1,
                "freq_range": [100, 1000],
                "amp_range": [0.1, 1.0],
                "phase_range": [0, 1],
                "waves": [
                    {"amp": 1,
                    "freq": 50,
                    "phase": 0},
                    {"amp": 1,
                    "freq": 500,
                    "phase": 0},
                    {"amp": 1,
                    "freq": 1500,
                    "phase": 0}                    
                ]
            }
        wave_params['phase_range'] = tuple(wave_params['phase_range'])
        wave_params["freq_range"] = tuple(wave_params["freq_range"])
        wave_params["amp_range"] = tuple(wave_params["amp_range"])
        self.wave_params = wave_params
        
    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def create_analog(self):
        analog = {}
        sim_freq = self.time_params.get('sim_freq', 1000000)
        adc_samp_freq = self.adc_params.get('adc_samp_freq', None)
        time_range = tuple(self.time_params.get('time_range', (0, 1)))
        
        points_per_second = round(sim_freq)
        
        # Adjust to be evenly divisible by adc_clock_freq
        if adc_samp_freq is not None:
            band = int(points_per_second / adc_samp_freq)
            band_remainder = int(points_per_second % adc_samp_freq)
            if band_remainder != 0:
                points_per_second -= band_remainder
            # K_band must be even
            if band % 2 != 0:
                points_per_second += int(adc_samp_freq)
                
        analog['sim_freq'] = points_per_second
        analog['adj_spacing'] = 1 / points_per_second
        analog['total_time'] = abs(time_range[1] - time_range[0])
        analog['num_points'] = int(round(analog['total_time'] * points_per_second))
        analog['time'] = np.linspace(time_range[0],
                                  time_range[1],
                                  analog['num_points'],
                                  endpoint=False)
        analog['frequency'] = np.linspace(-points_per_second / 2,
                                   points_per_second / 2,
                                   int(analog['time'].size),
                                   endpoint=False)
        return analog

    def create_input_signal(self):
        """
        Generate composite signal with environmental effects and random wave generation.
        """
         
        # Override defaults with values inside wave_params if they exist
        waves = self.wave_params.get('waves', [])
        num_waves = self.wave_params.get('num_waves', 1)
        freq_range = self.wave_params.get('freq_range', (100, 1000))
        amp_range = self.wave_params.get('amp_range', (0.1, 1.0))
        phase_range = self.wave_params.get('phase_range', (0, 1))
        allow_clipping = self.adc_params.get('allow_clipping', False)
        v_ref_range = tuple(self.adc_params.get('v_ref_range', (0, 1)))

        # === Wave parameter setup ===
        if not waves:
            amps = self.rng.uniform(amp_range[0], amp_range[1], num_waves)
            freqs = self.rng.uniform(freq_range[0], freq_range[1], num_waves)
            if phase_range:
                t_shift = self.rng.uniform(phase_range[0]/freqs, phase_range[1]/freqs)  # seconds
                phases = 2 * np.pi * freqs * t_shift
            else:
                phases = np.zeros(num_waves)
            # Save generated wave dictionaries into waves
            self.wave_params['waves'] = [
                {"amp": float(amps[i]), "freq": float(freqs[i]), "phase": float(phases[i])}
                for i in range(num_waves)
            ]
        else:
            self.wave_params['num_waves'] = len(self.wave_params['waves'])

        input_signal = self._generate_signal(self.wave_params['waves'])
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
        store_internal_sigs = self.env_params.get('store_internal_sigs', True)
        effects = {
            "noise": 0.0,
            "delay": [],
            "echo_att": [],
            "local_doppler": [],
            "phase_inversion": []
        }
        amps = np.array([wave['amp'] for wave in waves])
        freqs = np.array([wave['freq'] for wave in waves])
        phases = np.array([wave['phase'] for wave in waves])
        real_time = self.analog.get('time')
        
        # --- Set Evironment Nonidealities
        doppler = self.env_params.get('doppler', 0.0)
        delay = self.env_params.get('delay', 0.0)
        attenuation = self.env_params.get('attenuation', 1.0)
        noise_level = self.env_params.get('noise_level', 0.0)
        num_echoes = self.env_params.get('num_echoes', 0)
        max_delay = self.env_params.get('max_delay', 0.01)
        max_doppler = self.env_params.get('max_doppler', 0.002)
        phase_inversion_prob = self.env_params.get('phase_inversion_prob', 0.5)
        
        # === Apply system-level Doppler and delay ===
        freqs = freqs * (1 + doppler)
        real_time = real_time + delay

        # === Generate base composite signal ===
        signals = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * real_time + phases[:, None])
        signal = np.sum(signals, axis=0)

        # === Environmental attenuation ===
        signal *= attenuation

        # === Add Gaussian noise ===
        if noise_level > 0:
            effects['noise'] = self.rng.normal(0, noise_level, signal.shape)
            signal += effects['noise']

        # === Multipath reflections (with Doppler + phase inversion) ===
        if num_echoes > 0:
            dt = real_time[1] - real_time[0]


            for _ in range(num_echoes):
                # Random propagation delay
                phase_inversion = False
                delay = self.rng.uniform(0, max_delay)
                shift = int(delay / dt)

                # Random attenuation
                echo_att = self.rng.uniform(0.2, 0.8)

                # Random Doppler shift
                local_doppler = 1 + self.rng.uniform(-max_doppler, max_doppler)
                real_time_echo = real_time * local_doppler
                echo = np.interp(real_time, real_time_echo, signal, left=0, right=0) * echo_att

                # Optional phase inversion
                if self.rng.random() < phase_inversion_prob:
                    phase_inversion = True
                    echo = -echo

                # Apply time delay
                echo = np.roll(echo, shift)
                echo[:shift] = 0
                signal += echo
                effects['delay'].append(delay)
                effects['echo_att'].append(echo_att)
                effects['local_doppler'].append(local_doppler)
                effects['phase_inversion'].append(phase_inversion)
                
        if store_internal_sigs:
            self.logger.info("Storing Environmental Effects")
            self.effects = effects
            
        return signal

    # -------------------------------
    # Getters
    # -------------------------------

    def get_input_signal(self):
        return self.input_signal
    
    def get_config_name(self):
        return self.config_name
    
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
    
    def get_input_params(self):
        input_params = {
            "time_params": self.time_params,
            "adc_params": self.adc_params,
            "env_params": self.env_params,
            "wave_params": self.wave_params
        }
        return input_params       
    
    def get_time_params(self):
        return self.time_params
    
    def get_adc_params(self):
        return self.adc_params
    
    def get_env_params(self):
        return self.env_params
    
    def get_wave_params(self):
        return self.wave_params
    
    def get_log_params(self):
        return self.log_params
    
    def get_effects(self):
        return self.effects