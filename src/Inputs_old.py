from utility import load_settings
import numpy as np
from numpy import sin
from math import e, pi

def set_real_time(self, time_params=None):
    if time_params is not None or self.time_params is None:
        if self.time_params is None:
            print("No time parameters provided during initialization. Adding new time parameters")
        else:
            print("Adding new time parameters")
        self.set_time_params(time_params)

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
    self.adj_spacing = 1/self.real_points_per_second
    self.total_time = abs( self.time_params['start'] - self.time_params['stop'] )
    self.num_real_time_points = int ( self.total_time * self.real_points_per_second )
    self.real_t = np.linspace(self.time_params['start'], self.time_params['stop'], self.num_real_time_points, endpoint=False)
    self.real_tf = np.linspace(-self.real_points_per_second/2, self.real_points_per_second/2, int(self.real_t.size), endpoint=False)

def clear_real_time_sigs(self):
    self.LO = None
    self.real_t = None
    self.real_tf = None
    self.filt_freq = None

def set_LO(self, t=None):
    if t is None:
        t = self.real_t
    if self.LO_params is None:
        self.set_LO_params()

    self.LO_modulation = (self.LO_params['phase_delta']/self.LO_params['phase_freq'])* \
        sin(2*pi*self.LO_params['phase_freq']*t+self.LO_params['phase_offset'])
    
    self.LO = self.LO_params['amp']*sin(2*pi*self.LO_params['freq']*t+self.LO_params['phase']+self.LO_modulation)

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