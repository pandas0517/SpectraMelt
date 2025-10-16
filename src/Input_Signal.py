from utility import load_settings
import numpy as np

class Input_Signal:
    def __init__(self,
                 time_params=None,
                 system_params=None,
                 wave_params=None,
                 input_config_name=None,
                 config_file_path=None) -> None:
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_time_params(time_params)
            self.set_system_params(system_params)
            self.set_wave_params(wave_params)
            self.set_input_config_name(input_config_name)

        self.analog = None
        self.input_signal = None
        self.effects = None

    # -------------------------------
    # Setters
    # -------------------------------
    def set_config_from_file(self, config_file_path):
        print("Loading configuration from file: ", config_file_path)
        input_config = load_settings(config_file_path)
        self.set_input_config_name(input_config.get('input_config_name', None))
        self.set_time_params(input_config.get('time_params', None))
        self.set_system_params(input_config.get('system_params', None))
        self.set_wave_params(input_config.get('wave_params', None))
        
    def set_input_config_name(self, input_config_name=None):
        if input_config_name is None:
            input_config_name = "Input_Config_1"
        self.input_config_name = input_config_name

    def set_time_params(self, time_params=None):
        if time_params is None:
            time_params = {
                'start': 0,
                'stop': 1,
                'sim_freq': 1000000
            }
        self.time_params = time_params

    def set_system_params(self, system_params=None):
        if system_params is None:
            system_params = {
                "adc_samp_freq": 100,
                "allow_clipping": True,
                "v_ref_range": (0, 1),
                "system_noise_level": 0,
                "attenuation": 1,
                "doppler": 0,
                "delay": 0,
                "num_echoes": 0,
                "max_delay": 0,
                "max_doppler": 0,
                "phase_inversion_prob": 0
            }           
        self.system_params = system_params

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
    def create_analog(self):
        self.analog = {}
        points_per_second = round(self.time_params['sim_freq'])
        # Adjust to be evenly divisible by adc_clock_freq
        band = int(points_per_second / self.system_params['adc_samp_freq'])
        band_remainder = int(points_per_second % self.system_params['adc_samp_freq'])
        if band_remainder != 0:
            points_per_second -= band_remainder
        # K_band must be even
        if band % 2 != 0:
            points_per_second += int(self.system_params['adc_samp_freq'])

        self.analog['points_per_second'] = points_per_second
        self.analog['adj_spacing'] = 1 / points_per_second
        self.analog['total_time'] = abs(self.time_params['start'] - self.time_params['stop'])
        self.analog['num_points'] = int(self.analog['total_time'] * points_per_second)
        self.analog['time'] = np.linspace(self.time_params['start'],
                                  self.time_params['stop'],
                                  self.analog['num_points'],
                                  endpoint=False)
        self.analog['frequency'] = np.linspace(-points_per_second / 2,
                                   points_per_second / 2,
                                   int(self.analog['time'].size),
                                   endpoint=False)

    def create_input_signal(self):
        """
        Generate composite signal with environmental effects and random wave generation.
        """
         
        # Override defaults with values inside wave_params if they exist
        num_waves = self.wave_params.get('num_waves', 1)
        freq_range = tuple(self.wave_params.get('freq_range', (100, 1000)))
        amp_range = tuple(self.wave_params.get('amp_range', (0.1, 1.0)))
        phase_random = self.wave_params.get('phase_random', True)
        allow_clipping = self.system_params.get('allow_clipping', True)
        v_ref_range = tuple(self.system_params.get('v_ref_range', (0, 1)))

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

        self.generate_signal(waves)
            # --- Adjust to midpoint ---
        midpoint = (v_ref_range[1] + v_ref_range[0]) / 2
        self.input_signal = self.input_signal + midpoint  # shift signal to midpoint

        # --- Scale or clip signal ---
        if not allow_clipping:
            # Prevent clipping by scaling amplitude
            max_abs_val = max(abs(self.input_signal.min() - midpoint), abs(self.input_signal.max() - midpoint))
            max_allowed_amp = (v_ref_range[1] + v_ref_range[0]) / 2
            if max_abs_val > 0:
                scale = max_allowed_amp / max_abs_val
                self.input_signal = (self.input_signal - midpoint) * scale + midpoint
        else:
            # Allow clipping: just clip values beyond v_ref_min/v_ref_max
            self.input_signal = np.clip(self.input_signal, v_ref_range[0], v_ref_range[1])

    def generate_signal(self, waves):
        self.effects = {}
        amps = np.array([wave['amp'] for wave in waves])
        freqs = np.array([wave['freq'] for wave in waves])
        phases = np.array([wave['phase'] for wave in waves])
            # === Apply system-level Doppler and delay ===
        doppler = self.system_params.get('doppler', 0.0)
        freqs = freqs * (1 + doppler)

        delay = self.system_params.get('delay', 0.0)
        self.analog['time'] = self.analog['time'] + delay

        # === Generate base composite signal ===
        signals = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * self.analog['time'] + phases[:, None])
        signal = np.sum(signals, axis=0)

        # === Environmental attenuation ===
        attenuation = self.system_params.get('attenuation', 1.0)
        signal *= attenuation

        # === Add Gaussian noise ===
        noise_level = self.system_params.get('system_noise_level', 0.0)
        self.effects['noise'] = 0.0
        if noise_level > 0:
            self.effects['noise'] = np.random.normal(0, noise_level, signal.shape)
            signal += self.effects['noise']

        # === Multipath reflections (with Doppler + phase inversion) ===
        num_echoes = self.system_params.get('num_echoes', 0)
        self.effects['delay'] = []
        self.effects['echo_att'] = []
        self.effects['local_doppler'] = []
        self.effects['phase_inversion'] = []
        if num_echoes > 0:
            dt = self.analog['time'][1] - self.analog['time'][0]
            max_delay = self.system_params.get('max_delay', 0.01)
            max_doppler = self.system_params.get('max_doppler', 0.002)
            phase_inversion_prob = self.system_params.get('phase_inversion_prob', 0.5)

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

        self.input_signal = signal

    # -------------------------------
    # Getters
    # -------------------------------

    def get_input_signal(self):
        return self.input_signal
    
    def get_input_config_name(self):
        return self.input_config_name
    
    def get_analog_signals(self):
        return self.analog
    
    def get_time_params(self):
        return self.time_params
    
    def get_system_params(self):
        return self.system_params
    
    def get_wave_params(self):
        return self.wave_params
    
    def get_effects(self):
        return self.effects