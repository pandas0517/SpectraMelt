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
            self.create_real_time()

    # -------------------------------
    # Parameter setters
    # -------------------------------
    def set_config_from_file(self, config_file_path):
        print("Loading configuration from file: ", config_file_path)
        input_config = load_settings(config_file_path)
        self.set_input_config_name(input_config.get('input_config_name', None))
        self.set_time_params(input_config.get('time_params', None))
        self.set_system_params(input_config.get('system_params', None))
        self.set_wave_params(input_config.get('wave_params', None))
        self.create_real_time()
        
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
                "v_ref_range": (0, 5),
                "system_noise_level": 0,
                "attenuation": 0,
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
                    {"amp": 1,
                    "freq": 1,
                    "phase": 0}
                ]
            }
        self.wave_params = wave_params
        
    # -------------------------------
    # Core functional methods
    # -------------------------------
    def create_real_time(self):
        points_per_second = round(self.time_params['sim_freq'])
        # Adjust to be evenly divisible by adc_clock_freq
        band = int(points_per_second / self.system_params['adc_samp_freq'])
        band_remainder = int(points_per_second % self.system_params['adc_samp_freq'])
        if band_remainder != 0:
            points_per_second -= band_remainder
        # K_band must be even
        if band % 2 != 0:
            points_per_second += int(self.system_params['adc_samp_freq'])

        self.real_points_per_second = points_per_second
        self.adj_spacing = 1 / self.real_points_per_second
        self.total_time = abs(self.time_params['start'] - self.time_params['stop'])
        self.num_real_time_points = int(self.total_time * self.real_points_per_second)
        self.real_time = np.linspace(self.time_params['start'],
                                  self.time_params['stop'],
                                  self.num_real_time_points,
                                  endpoint=False)
        self.real_frequency = np.linspace(-self.real_points_per_second / 2,
                                   self.real_points_per_second / 2,
                                   int(self.real_time.size),
                                   endpoint=False)

    def create_input_signal(self):
        """
        Generate composite signal with environmental effects and random wave generation.

        Parameters
        ----------
        real_time : np.ndarray
            Time vector.
        system_params : dict
            Dictionary of system/environmental parameters.
        wave_params : list[dict], optional
            Explicitly provided wave definitions.
        num_waves : int
            Number of sine waves to generate (ignored if wave_params provided).
        freq_range : tuple(float, float)
            (min_freq, max_freq) range for random frequency generation.
        amp_range : tuple(float, float)
            (min_amp, max_amp) range for random amplitude generation.
        phase_random : bool
            If True, assign random phases in [0, 2π].

        Returns
        -------
        input_signal : np.ndarray
            The generated composite signal with environmental effects.
        """
         
        # Override defaults with values inside wave_params if they exist
        num_waves = self.wave_params.get('num_waves', 1)
        freq_range = tuple(self.wave_params.get('freq_range', (100, 1000)))
        amp_range = tuple(self.wave_params.get('amp_range', (0.1, 1.0)))
        phase_random = self.wave_params.get('phase_random', True)
        waves = self.wave_params.get('waves',[])
        
        # === Wave parameter setup ===
        if not waves:
            amps = np.random.uniform(amp_range[0], amp_range[1], num_waves)
            freqs = np.random.uniform(freq_range[0], freq_range[1], num_waves)
            if phase_random:
                t_shift = np.random.uniform(0, 1/freqs)  # seconds
                phases = 2 * np.pi * freqs * t_shift
            else:
                phases = np.zeros(num_waves)
        else:
            amps = np.array([p['amp'] for p in wave_params])
            freqs = np.array([p['freq'] for p in wave_params])
            phases = np.array([p['phase'] for p in wave_params])

        # === Apply system-level Doppler and delay ===
        doppler = system_params.get('doppler', 0.0)
        freqs = freqs * (1 + doppler)

        delay = system_params.get('delay', 0.0)
        real_time_eff = real_time + delay

        # === Generate base composite signal ===
        waves = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * real_time_eff + phases[:, None])
        input_signal = np.sum(waves, axis=0)

        # === Environmental attenuation ===
        attenuation = system_params.get('attenuation', 1.0)
        input_signal *= attenuation

        # === Add Gaussian noise ===
        noise_level = system_params.get('system_noise_level', 0.0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, input_signal.shape)
            input_signal += noise

        # === Multipath reflections (with Doppler + phase inversion) ===
        num_echoes = system_params.get('num_echoes', 0)
        if num_echoes > 0:
            dt = real_time[1] - real_time[0]
            max_delay = system_params.get('max_delay', 0.01)
            max_doppler = system_params.get('max_doppler', 0.002)
            phase_inversion_prob = system_params.get('phase_inversion_prob', 0.5)

            for _ in range(num_echoes):
                # Random propagation delay
                delay = np.random.uniform(0, max_delay)
                shift = int(delay / dt)

                # Random attenuation
                echo_att = np.random.uniform(0.2, 0.8)

                # Random Doppler shift
                local_doppler = 1 + np.random.uniform(-max_doppler, max_doppler)
                real_time_echo = real_time * local_doppler
                echo = np.interp(real_time, real_time_echo, input_signal, left=0, right=0) * echo_att

                # Optional phase inversion
                if np.random.rand() < phase_inversion_prob:
                    echo = -echo

                # Apply time delay
                echo = np.roll(echo, shift)
                echo[:shift] = 0
                input_signal += echo

        return input_signal

def generate_signal(self, amps, freqs, phases):
            # === Apply system-level Doppler and delay ===
        doppler = self.system_params.get('doppler', 0.0)
        freqs = freqs * (1 + doppler)

        delay = self.system_params.get('delay', 0.0)
        real_time_eff = self.real_time + delay

        # === Generate base composite signal ===
        waves = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * real_time_eff + phases[:, None])
        input_signal = np.sum(waves, axis=0)

        # === Environmental attenuation ===
        attenuation = self.system_params.get('attenuation', 1.0)
        input_signal *= attenuation

        # === Add Gaussian noise ===
        noise_level = self.system_params.get('system_noise_level', 0.0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, input_signal.shape)
            input_signal += noise

        # === Multipath reflections (with Doppler + phase inversion) ===
        num_echoes = self.system_params.get('num_echoes', 0)
        if num_echoes > 0:
            dt = self.real_time[1] - self.real_time[0]
            max_delay = self.system_params.get('max_delay', 0.01)
            max_doppler = self.system_params.get('max_doppler', 0.002)
            phase_inversion_prob = self.system_params.get('phase_inversion_prob', 0.5)

            for _ in range(num_echoes):
                # Random propagation delay
                delay = np.random.uniform(0, max_delay)
                shift = int(delay / dt)

                # Random attenuation
                echo_att = np.random.uniform(0.2, 0.8)

                # Random Doppler shift
                local_doppler = 1 + np.random.uniform(-max_doppler, max_doppler)
                real_time_echo = self.real_time * local_doppler
                echo = np.interp(self.real_time, real_time_echo, input_signal, left=0, right=0) * echo_att

                # Optional phase inversion
                if np.random.rand() < phase_inversion_prob:
                    echo = -echo

                # Apply time delay
                echo = np.roll(echo, shift)
                echo[:shift] = 0
                input_signal += echo

        return input_signal