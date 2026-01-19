from .utils import (
    load_config_from_json,
    get_logger,
)
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AnalogData:
    sim_freq: float  | None = None         # points per second
    adj_spacing: float | None = None       # spacing between points (1/fs)
    total_time: float | None = None        # duration of the signal
    num_points: int | None = None          # number of samples
    time: np.ndarray | None = None         # time vector
    frequency: np.ndarray | None = None    # frequency vector


class Analog:
    def __init__(self,
                 all_params=None,
                 time_params=None,
                 log_params=None,
                 config_name=None,
                 config_file_path=None) -> None:
        if config_file_path is not None:
            all_params = load_config_from_json(config_file_path)
        elif all_params is None:
            all_params = {
                "time_params": time_params,
                "config_name": config_name,
                "log_params": log_params
            }
        
        self.set_all_params(all_params)
        
        if config_file_path is not None and self.logger is not None:
            self.logger.info(f"Loaded {self.__class__.__name__} configuration from file: {config_file_path}")

    # -------------------------------
    # Setters
    # -------------------------------
    
    def set_all_params(self, all_params=None):
        if all_params is None:
            all_params = {}
        
        time_params = all_params.get('time_params', None)
        log_params = all_params.get('log_params', None)
        config_name = all_params.get('config_name', None)
        if time_params is None:
            config_name = "Default_Analog_Config"
        else:
            config_name = all_params.get('config_name', "Analog_Config_1")
        
        self.set_log_params(log_params)    
        self.logger = None
        logging_enabled = self.log_params.get('enabled', True)
        if logging_enabled:
            log_file = self.log_params.get('log_file', None)
            level = self.log_params.get('level', "INFO")
            console = self.log_params.get('console', True)
            self.logger = get_logger(self.__class__.__name__, log_file, level, console)
       
        self.set_time_params(time_params)
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
                'time_range': [0, 1],
                "adc_samp_freq": None,
                'sim_freq': 1000000
            }
            
        time_params['time_range'] = tuple(time_params['time_range'])
        self.time_params = time_params
        
    # -------------------------------
    # Core functional methods
    # -------------------------------
    
    def create_analog(self) -> AnalogData:
        sim_freq = self.time_params.get('sim_freq', 1000000)
        adc_samp_freq = self.time_params.get('adc_samp_freq', None)
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
                
        total_time = abs(time_range[1] - time_range[0])
        num_points = int(round(total_time * points_per_second))
        time = np.linspace(time_range[0], time_range[1], num_points, endpoint=False)
        frequency = np.linspace(-points_per_second/2, points_per_second/2, num_points, endpoint=False)
        
        return AnalogData(
            sim_freq=points_per_second,
            adj_spacing=1 / points_per_second,
            total_time=total_time,
            num_points=num_points,
            time=time,
            frequency=frequency
        )

    # -------------------------------
    # Getters
    # -------------------------------
    
    def get_config_name(self):
        return self.config_name
           
    
    def get_time_params(self):
        return self.time_params
    
    
    def get_log_params(self):
        return self.log_params
    

    def get_all_params(self):
        input_params = {
            "config_name": self.config_name,
            "time_params": self.time_params,
            "log_params": self.log_params
        }
        return input_params