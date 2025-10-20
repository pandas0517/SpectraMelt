import numpy as np
from utility import load_settings

class LocalOscillator:
    """
    Represents a realistic local oscillator (LO) with optional modulation, noise,
    frequency drift, and harmonic distortion.
    """

    def __init__(self,
                 real_time=None,
                 lo_params=None,
                 time_params=None,
                 adc_params=None,
                 lo_config_name=None,
                 config_file_path=None) -> None:
        """
        Parameters
        ----------
        lo_params : dict
            {
                'freq': carrier frequency (Hz),
                'amp': amplitude,
                'phase': initial phase (radians),
                'mod_enable': bool,                     # enable phase modulation
                'phase_freq': modulation frequency (Hz),
                'phase_delta': phase deviation (radians),
                'phase_offset': phase offset (radians),

                'phase_noise_std': float,                # std dev of random phase noise (radians)
                'amp_noise_std': float,                  # std dev of amplitude noise (fraction of amplitude)
                'freq_drift_ppm': float,                 # frequency drift in parts per million
                'harmonic_distortion': float             # fractional 2nd harmonic amplitude
            }
        t : np.ndarray
            Time vector for signal generation.
        """
        if config_file_path is not None:
            self.set_config_from_file(config_file_path)
        else:
            self.set_lo_params(lo_params)
            self.set_adc_params(adc_params)
            self.set_time_params(time_params)
            if ( time_params is None and
                adc_params is None and
                lo_params is None):
                lo_config_name = "Default_LO_Config"
            self.set_lo_config_name(lo_config_name)
        if real_time is None:
            real_time = self._create_analog().get('time')
        self.real_time = real_time
        self.signal = self.generate_signal()
 
    # -------------------------------
    # Setters
    # -------------------------------
       
    def set_config_from_file(self, config_file_path=None):
        print("Loading Local Oscillator configuration from file: ", config_file_path)
        lo_config = load_settings(config_file_path)
        lo_params = lo_config.get('lo_params', None)
        time_params = lo_config.get('time_params', None)
        adc_params = lo_config.get('adc_params', None)
        lo_config_name = lo_config.get('config_name', None)

        self.set_lo_params(lo_params)
        self.set_time_params(time_params)
        self.set_adc_params(adc_params)
        self.set_lo_config_name(lo_config_name)
        
    def set_lo_config_name(self, lo_config_name=None):
        if lo_config_name is None:
            lo_config_name = "LO_Config_1"
        self.lo_config_name = lo_config_name        
        
    def set_lo_params(self, lo_params=None):
        if lo_params is None:
            lo_params = {
                "amp": 1,
                "freq": 100,
                "phase": 0,
                "mod_enabled": False,
                "phase_delta": 0.08,
                "phase_freq": 0.1,
                "phase_offset": 0.0,
                "phase_noise_std": 0.0,
                "amp_noise_std": 0.0,
                "freq_drift_ppm": 0.0,
                "harmonic_distortion": 0.0,
                "seed": None
            }
        self.rng = np.random.default_rng(lo_params.get('seed', None))           
        self.lo_params = lo_params
        
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

    def generate_signal(self, real_time=None) -> np.ndarray:
        if real_time is None:
            real_time = self.real_time
    
        # --- Local Oscillator Base Signal ---
        f0 = self.lo_params.get('freq', 100)
        A = self.lo_params.get('amp', 1)
        phase = self.lo_params.get('phase', 0)
        pre_phase = phase
        
        # --- Local Oscillator Modulation Parameters ---
        mod_enabled = self.lo_params.get('mod_enabled', False)
        phase_delta = self.lo_params.get('phase_delta', 0.08)
        phase_freq = self.lo_params.get('phase_freq', 0.1)
        phase_offset = self.lo_params.get('phase_offset', 0.0)
        
        # --- Set Local Oscillator Nonidealities ---
        freq_drift_ppm = self.lo_params.get('freq_drift_ppm', 0.0)
        phase_noise_std = self.lo_params.get('phase_noise_std', 0.0)
        amp_noise_std = self.lo_params.get('amp_noise_std', 0.0)
        harmonic_distortion = self.lo_params.get('harmonic_distortion', 0.0)
        
        # --- Frequency drift (ppm → fractional offset) ---
        freq_drift = f0 * (freq_drift_ppm * 1e-6)

        # --- Compute pre-start time for edge case handling ---
        dt = real_time[1] - real_time[0]
        pre_start_time = real_time[0] - dt

        # --- Optional phase modulation ---
        if mod_enabled:
            phase_mod = (phase_delta / phase_freq) * \
                        np.sin(2 * np.pi * phase_freq * real_time + phase_offset)
            pre_phase_mod = (phase_delta / phase_freq) * \
                            np.sin(2 * np.pi * phase_freq * pre_start_time + phase_offset)
        else:
            phase_mod = 0.0
            pre_phase_mod = 0.0

        # --- Phase noise (white, Gaussian) ---
        phase_noise = self.rng.normal(0, phase_noise_std, len(real_time))
        pre_phase_noise = self.rng.normal(0, phase_noise_std)  # single value

        # --- Amplitude noise (white, Gaussian) ---
        amp_noise = self.rng.normal(0, amp_noise_std, len(real_time))
        pre_amp_noise = self.rng.normal(0, amp_noise_std)
        amp = A * (1 + amp_noise)
        pre_amp = A * (1 + pre_amp_noise)

        # --- Instantaneous phase (includes drift, modulation, and noise) ---
        phase = 2 * np.pi * (f0 + freq_drift) * real_time + \
                phase + phase_mod + phase_noise

        pre_phase = 2 * np.pi * (f0 + freq_drift) * pre_start_time + \
                    pre_phase + pre_phase_mod + pre_phase_noise

        # --- Fundamental signal ---
        lo = amp * np.sin(phase)
        pre_start_lo = pre_amp * np.sin(pre_phase)

        # --- Add 2nd harmonic distortion ---
        harmonic_amp = harmonic_distortion * A
        if harmonic_amp != 0:
            lo += harmonic_amp * np.sin(2 * phase)
            pre_start_lo += harmonic_amp * np.sin(2 * pre_phase)

        # --- Store pre-start LO value for later use (e.g., zero-cross detection) ---
        self.pre_start_lo = pre_start_lo

        return lo

    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_lo_config_name(self):
        return self.lo_config_name
    
    def get_lo_signal(self):
        return self.signal
    
    def get_real_time(self):
        return self.real_time
    
    def get_lo_params(self):
        return self.lo_params
    
    def get_time_params(self):
        return self.time_params
    
    def get_adc_params(self):
        return self.adc_params
    
    def get_pre_start_lo(self):
        return self.pre_start_lo