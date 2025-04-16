from utility import get_all_file_paths, get_all_sub_dirs
from utility import load_settings, get_file_sub_dirs
from utility import delete_lines_with_string
from NYFR import NYFR
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import os
import time
import random
from itertools import combinations
import pickle

class NYFR_Test_Harness:
    def __init__(self,
                 filenames=None,
                 directories=None,
                 input_set_params=None, 
                 filenames_json=None,
                 directories_json=None,
                 input_set_json=None,
                 system_conf_json=None,
                 nyfr=None) -> None:
        if filenames_json is not None:
            filenames = load_settings(filenames_json)         
        
        if directories_json is not None:
            directories = load_settings(directories_json)

        if input_set_json is not None:
            input_set_params = load_settings(input_set_json)

        if system_conf_json is not None:
            nyfr = NYFR(file_path=system_conf_json)
            nyfr.initialize()

        self.__set_init(nyfr=nyfr,
                        filenames=filenames,
                        directories=directories,
                        input_set_params=input_set_params)

    def create_sets(self, nyfr=None, filenames=None, directories=None, input_set_params=None):
        self.__set_init(nyfr, filenames, directories, input_set_params)
        if self.__needs_init(include_set_params=True):
            print("NYFR Test Harness not properly initialized.  Please re-initialize object")
            return
        system_params = self.nyfr.get_system_params()
        wbf_cut_freq = system_params['wbf_cut_freq']
        LO_params = self.nyfr.get_LO_params()
        for noise_level, _ in self.input_set_params["noise_levels"]:
            for phase_shift, _ in self.input_set_params["phase_shifts"]:
                for input_tones, _ in self.input_set_params["input_tones"]:
                    input_file_path = os.path.join(self.input_dir,
                                                   noise_level,
                                                   phase_shift,
                                                   self.input_tones[input_tones]['sigs']) # e.g. 1_2_tone_sigs.npy
                    input_list_path = os.path.join(self.input_dir,
                                                   noise_level,
                                                   phase_shift,
                                                   self.input_tones[input_tones]['list'])
                    for f_mod, f_mod_value in self.input_set_params["f_mods"]:
                        LO_params['phase_freq'] = f_mod_value
                        for f_delta, f_delta_value in self.input_set_params["f_deltas"]:
                            LO_params['phase_delta'] = round(f_delta_value * f_mod_value, 2)
                            self.nyfr.set_LO_params(LO_params=LO_params)
                            output_file_path = os.path.join(self.output_dir,
                                                           noise_level,
                                                           phase_shift,
                                                           f_mod,
                                                           f_delta,
                                                           self.input_tones[input_tones]['sigs'])
                            dictionary_file_path = os.path.join(self.dictionary_dir[self.nyfr.get_dictionary_params()['version']],
                                                                f_mod,
                                                                f_delta,
                                                                self.dictionary_file['name'])
                            output_list = []
                            input_list = []
                            wave_param_list = []
                            output_file_exists = os.path.isfile(output_file_path)
                            input_list_exists = os.path.isfile(input_list_path)
                            if not output_file_exists:
                                if input_list_exists:
                                    with open(input_list_path, 'rb') as file:
                                        input_freq_tot_list = pickle.load(file)
                                else:
                                    input_freq_tot_list = self.__get_frequency_list(input_tones, wbf_cut_freq)

                                for input_freqs in input_freq_tot_list:
                                    if input_list_exists:
                                        wave_params = input_freqs[0]
                                        noise = input_freqs[1]
                                    else:
                                        wave_params, noise = self.__update_wave_system(input_freqs,phase_shift,noise_level)
                                        wave_param_list.append((wave_params, noise))
                                    system_params['system_noise_level'] = noise
                                    self.nyfr.set_system_params(system_params=system_params)
                                    analog_input, _ = self.nyfr.create_input_signal(wave_params=wave_params)
                                    output_list.append( self.nyfr.simulate_system(input_signal=analog_input) )
                                    if ( not os.path.isfile(input_file_path) ):
                                        input_list.append(self.nyfr.sample_signals(data=analog_input, sample_rate=self.nyfr.get_wb_nyquist_rate()))
                                    if ( not os.path.isfile(dictionary_file_path) ):
                                        dictionary = self.nyfr.create_dict()
                                        np.save(dictionary_file_path, dictionary) # save the dictionary for this f_mod/f_delta combo
                                if wave_param_list:
                                    # Save the wave parameters if they were generated
                                    with open(input_list_path, 'wb') as file:
                                        pickle.dump(wave_param_list, file)
                                if input_list:
                                    # Save the input set if it was generated
                                    input_set = np.array(input_list)
                                    np.save(input_file_path, input_set)
                                output_set = np.array(output_list)
                                np.save(output_file_path, output_set)

    def __get_frequency_list(self, num_of_sigs, wbf_cut_freq):
        pos_bins = list(range(1, wbf_cut_freq))
        input_freq_tot_list = []
        match num_of_sigs:
            case '1_2':
                    input_freq_list_1 = list(combinations(pos_bins,1))
                    input_freq_list_2 = list(combinations(pos_bins, 2))
                    random.shuffle(input_freq_list_2)
                    input_freq_list_2 = input_freq_list_2[0:self.input_set_params['tot_num_freq_combos'] - len(input_freq_list_1)]
                    input_freq_tot_list = input_freq_list_1 + input_freq_list_2
                    random.shuffle(input_freq_tot_list)
            case '3':
                input_freq_tot_list += list(combinations(pos_bins, 3))
                random.shuffle(input_freq_tot_list)
                input_freq_tot_list = input_freq_tot_list[0:self.input_set_params['tot_num_freq_combos']]
            case '4':
                input_freq_tot_list = [(random.randint(1, wbf_cut_freq),
                                        random.randint(1, wbf_cut_freq),
                                        random.randint(1, wbf_cut_freq),
                                        random.randint(1, wbf_cut_freq)) for _ in range(self.input_set_params['tot_num_freq_combos'])]
            case '5':
                input_freq_tot_list = [(random.randint(1, wbf_cut_freq),
                                        random.randint(1, wbf_cut_freq),
                                        random.randint(1, wbf_cut_freq),
                                        random.randint(1, wbf_cut_freq),
                                        random.randint(1, wbf_cut_freq)) for _ in range(self.input_set_params['tot_num_freq_combos'])]
            case _:
                print("Unsupported right now")
        return input_freq_tot_list
    
    def __update_wave_system(self, input_freqs, phase_shift, noise_level):
        wave_params = []
        for input_freq in input_freqs:
            wave_param = {
                "amp": random.uniform(self.input_set_params['amp_min'], self.input_set_params['amp_max']),
                "freq": input_freq,
                "phase": 0
            }
            if phase_shift == "high_phase_shift":
                wave_param['phase'] = random.uniform(self.input_set_params['high_phase_min'], self.input_set_params['high_phase_max']) / input_freq
            if phase_shift == "low_phase_shift":
                wave_param['phase'] = random.uniform(self.input_set_params['low_phase_min'], self.input_set_params['low_phase_max']) / input_freq
            wave_params.append(wave_param)

        noise = 0
        if noise_level == "high_noise":
            noise = random.uniform(self.input_set_params['high_noise_min'], self.input_set_params['high_noise_max'])
        elif noise_level == "low_noise":
            noise = random.uniform(self.input_set_params['low_noise_min'], self.input_set_params['low_noise_max'])
        return wave_params, noise

    def create_dictionaries(self, nyfr=None):
        if nyfr is not None:
            self.nyfr = nyfr
        LO_params = self.nyfr.get_LO_params()
        dictionary_params = self.nyfr.get_dictionary_params()
        if LO_params == None or self.nyfr.get_K_band() == None or dictionary_params == None:
            print("NYFR not initialized.  Please re-initialize object")
        else:
            f_mod_list = [[0.1, "f_mod_0_1"], [0.2, "f_mod_0_2"], [0.25, "f_mod_0_25"], [0.5, "f_mod_0_5"]]
            f_delta_list = [[0.1, "f_delta_0_1"], [0.8, "f_delta_0_8"], [1.2, "f_delta_1_2"], [10, "f_delta_9_9"]]
            for f_mod in f_mod_list:
                LO_params['phase_freq'] = f_mod[0]
                mod_dir = f_mod[1]
                for f_delta in f_delta_list:
                    LO_params['phase_delta'] = round(f_delta[0] * f_mod[0], 2)
                    nyfr.set_LO_params(LO_params=LO_params)
                    dictionary_file_path = os.path.join(self.dictionary_dir[dictionary_params['version']],
                                                        mod_dir,
                                                        f_delta[1],
                                                        self.dictionary_file['name'])
                    dictionary = nyfr.create_dict()
                    np.save(dictionary_file_path, dictionary)

    def create_input_sets(self, nyfr=None, filenames=None, directories=None, input_set_params=None):
        _ = self.__set_init(nyfr, filenames, directories, input_set_params=input_set_params)
        system_params = self.nyfr.get_system_params()
        wbf_cut_freq = system_params['wbf_cut_freq']
        for num_of_sigs, tone_sigs_file_name in self.input_tones.items():
                input_freq_tot_list = self.__get_frequency_list(num_of_sigs=num_of_sigs,
                                                               wbf_cut_freq=wbf_cut_freq)
                input_sub_dirs = get_all_sub_dirs(self.input_dir)
                for sub_dir in input_sub_dirs:
                    input_file_path = os.path.join(sub_dir, tone_sigs_file_name)
                    noise_level, phase_shift, _ = get_file_sub_dirs(input_file_path)
                    input_list = []
                    for input_freqs in input_freq_tot_list:
                        wave_params, noise = self.__update_wave_system(input_freqs=input_freqs,
                                                                       phase_shift=phase_shift,
                                                                       noise_level=noise_level)
                        system_params['system_noise_level'] = noise
                        self.nyfr.set_system_params(system_params=system_params)
                        input, _ = self.nyfr.create_input_signal(wave_params=wave_params)
                        input_list.append(input)
                    input_set = np.array(input_list)
                    np.save(input_file_path, input_set)

    def create_output_sets(self, nyfr=None, filenames=None, directories=None):
        mod_delta_table = self.__set_init(nyfr, filenames, directories)
        f_mod_list = ["f_mod_0_1", "f_mod_0_2", "f_mod_0_25", "f_mod_0_5"]
        f_delta_list = ["f_delta_0_1", "f_delta_0_8", "f_delta_1_2", "f_delta_9_9"]
        input_file_paths = get_all_file_paths(self.input_dir)
        time_file_path = os.path.join(self.time_dir, self.time_file)
        LO_params = self.nyfr.get_LO_params()
        output_list = []
        t = np.load(time_file_path)
        for input_file_path in input_file_paths:
            input_set = np.load(input_file_path)
            noise_level, phase_shift, file_name = get_file_sub_dirs(input_file_path)
            output_sub_path = self.output_dir + noise_level + "\\" + phase_shift + "\\"
            for f_mod in f_mod_list:
                LO_params['phase_freq'] = mod_delta_table[f_mod]
                for f_delta in f_delta_list:
                    LO_params['phase_delta'] = round(mod_delta_table[f_delta] * mod_delta_table[f_mod], 2)
                    self.nyfr.set_LO_params(LO_params=LO_params)
                    output_sub_dir = output_sub_path + f_mod + "\\" + f_delta + "\\"
                    output_file_path = (os.path.join(output_sub_dir, file_name))
                    if ( os.path.isfile(output_file_path) ):
                        os.remove( output_file_path )
                    for input in input_set:
                        output_list.append( self.nyfr.simulate_system(input_signal=input) )

                    output_set = np.array(output_list)
                    np.save(output_file_path, output_set)
                    output_list.clear()

    def batch_recover(self, nyfr=None, filenames=None, directories=None, recovery_set_size=100, get_recovery_time=False):
        self.__set_init(nyfr=nyfr, filenames=filenames, directories=directories)
        if self.__needs_init():
            print("NYFR Test Harness not properly initialized.  Please re-initialize object")
            return
        system_params = self.nyfr.get_system_params()
        dictionary_params = self.nyfr.get_dictionary_params()
        recovery_params = self.nyfr.get_recovery_params()
        
        recovery_list = []
        for mode in recovery_params['modes']:
            recovery_base_path = self.recovery_dir[dictionary_params['version']][recovery_params['type']]
            if ( recovery_params['type'] == 'MLP1' ):
                recovery_base_path = os.path.join(recovery_base_path,
                                                  mode)
            if ( mode == 'real_imag' ):
                mlp_models_base_path = self.mlp_models_dir[dictionary_params['version']]['real']
                mlp_models_base_path_aux = self.mlp_models_dir[dictionary_params['version']]['imag']
            elif ( mode == 'mag_ang' ):
                mlp_models_base_path = self.mlp_models_dir[dictionary_params['version']]['mag']
                mlp_models_base_path_aux = self.mlp_models_dir[dictionary_params['version']]['ang']
            elif ( mode == 'complex' ):
                mlp_models_base_path = self.mlp_models_dir[dictionary_params['version']]['complex']
                mlp_models_base_path_aux = None
            elif ( mode == 'active_zones' ):
                mlp_models_base_path = self.mlp_models_dir[dictionary_params['version']]['active_zones']
                mlp_models_base_path_aux = None
            for processing_system in system_params['processing_systems']:
                for noise_level, _ in self.input_set_params["noise_levels"]:
                    for phase_shift, _ in self.input_set_params["phase_shifts"]:
                        for input_tones, _ in self.input_set_params["input_tones"]:
                            if ( mode == 'real_imag' ):
                                recovery_log_file_path = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['real'][processing_system])
                                recovery_log_file_path_aux = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['imag'][processing_system])
                            elif ( mode == 'mag_ang' ):
                                recovery_log_file_path = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['mag'][processing_system])
                                recovery_log_file_path_aux = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['ang'][processing_system])
                            elif ( mode == 'complex' ):
                                recovery_log_file_path = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode]['complex'][processing_system])
                                recovery_log_file_path_aux = None
                            elif ( mode == 'active_zones' ):
                                recovery_log_file_path = os.path.join(self.recovery_dir[dictionary_params['version']][recovery_params['type']],
                                                                    mode,
                                                                    self.recovery_file[mode][processing_system])
                                recovery_log_file_path_aux = None
                            for f_mod, _ in self.input_set_params["f_mods"]:
                                for f_delta, _ in self.input_set_params["f_deltas"]:
                                    output_file_path = os.path.join(self.output_dir,
                                                                    noise_level,
                                                                    phase_shift,
                                                                    f_mod,
                                                                    f_delta,
                                                                    self.input_tones[input_tones]['sigs'])
                                    mlp_model_file_path = os.path.join(mlp_models_base_path,
                                                                       noise_level,
                                                                       phase_shift,
                                                                       f_mod,
                                                                       f_delta,
                                                                       self.mlp_models_file['name'])
                                    mlp_model_aux_file_path = None
                                    if mlp_models_base_path_aux is not None:
                                        mlp_model_aux_file_path = os.path.join(mlp_models_base_path_aux,
                                                                               noise_level,
                                                                               phase_shift,
                                                                               f_mod,
                                                                               f_delta,
                                                                               self.mlp_models_file['name'])
                                    dictionary_file_path = os.path.join(self.dictionary_dir[dictionary_params['version']],
                                                                        f_mod,
                                                                        f_delta,
                                                                        self.dictionary_file['name'])
                                    recovery_file_path = os.path.join(recovery_base_path,
                                                                      noise_level,
                                                                      phase_shift,
                                                                      f_mod,
                                                                      f_delta,
                                                                      self.recovery_file['name'])
                                    found_string_in_file = False
                                    found_string_in_file_aux = False
                                    with open(recovery_log_file_path, "r") as recovery_log:
                                        for line in recovery_log:
                                            if output_file_path in line:
                                                found_string_in_file = True
                                                break
                                    if recovery_log_file_path_aux is not None:
                                        with open(recovery_log_file_path_aux, "r") as recovery_log:
                                            for line in recovery_log:
                                                if output_file_path in line:
                                                    found_string_in_file_aux = True
                                                    break                    
                                    if ( found_string_in_file ):
                                        output_set = np.load(output_file_path)
                                        dictionary = np.load(dictionary_file_path)
                                        if ( not os.path.isfile(recovery_file_path )):
                                            if get_recovery_time:
                                                ave_recovery_time = 0
                                                start_time = time.perf_counter()
                                            for idx in range(recovery_set_size):
                                                pass
                                                recovered_signal = self.nyfr.recover_signal(dictionary,
                                                                                            output_set[idx],
                                                                                            file_path=mlp_model_file_path,
                                                                                            aux_file_path=mlp_model_aux_file_path,
                                                                                            mode=mode)
                                                recovery_list.append(recovered_signal)
                                            if get_recovery_time:
                                                end_time = time.perf_counter()
                                                ave_recovery_time = ( end_time - start_time ) / recovery_set_size
                                            recovery_set = np.array(recovery_list)
                                            np.save(recovery_file_path, recovery_set)
                                            recovery_list = []
                                        delete_lines_with_string(recovery_log_file_path, output_file_path)

    def create_dfs(self, nyfr=None, filenames=None, directories=None):
        self.__set_init(nyfr=nyfr, filename=filenames, directories=directories)
        if self.__needs_init():
            print("NYFR Test Harness not properly initialized.  Please re-initialize object")
            return
        dictionary_params = self.nyfr.get_dictionary_params()
        recovery_params = self.nyfr.get_recovery_params()
        input_df_file_path = os.path.join(self.df_dir, self.input_df_file)
        output_df_file_path = os.path.join(self.df_dir, self.output_df_file)
        recovery_df_file_path = os.path.join(self.df_dir, self.recovery_file['df'])

        input_df = pd.DataFrame({'file_name': pd.Series(dtype='str'),
                    'noise_level': pd.Series(dtype='str'),
                    'phase_shift': pd.Series(dtype='str'),
                    'num_tones': pd.Series(dtype='int'),
                    'tone_frequencies': pd.Series(dtype='int')})
        output_df = pd.DataFrame({'file_name': pd.Series(dtype='str'),
                    'noise_level': pd.Series(dtype='str'),
                    'phase_shift': pd.Series(dtype='str'),
                    'f_mod': pd.Series(dtype='float'),
                    'f_delta': pd.Series(dtype='float')})
        recovery_df = pd.DataFrame({'file_name': pd.Series(dtype='str'),
                    'noise_level': pd.Series(dtype='str'),
                    'phase_shift': pd.Series(dtype='str'),
                    'f_mod': pd.Series(dtype='float'),
                    'f_delta': pd.Series(dtype='float')})
        for mode in recovery_params['modes']:
            recovery_base_path = self.recovery_dir[dictionary_params['version']][recovery_params['type']]
            if ( recovery_params['type'] == 'MLP1' ):
                recovery_base_path = os.path.join(recovery_base_path,
                                                  mode)
            for noise_level, _ in self.input_set_params["noise_levels"]:
                for phase_shift, _ in self.input_set_params["phase_shifts"]:
                    for input_tones, num_tones in self.input_set_params["input_tones"]:
                        input_df.loc[len(input_df)] = [input_tones, noise_level, phase_shift, num_tones, 0]
                        for f_mod, f_mod_value in self.input_set_params["f_mods"]:
                            for f_delta, f_delta_value in self.input_set_params["f_deltas"]:
                                output_file_path = os.path.join(self.output_dir,
                                                                noise_level,
                                                                phase_shift,
                                                                f_mod,
                                                                f_delta,
                                                                self.input_tones[input_tones]['sigs'])
                                recovery_file_path = os.path.join(recovery_base_path,
                                                                  noise_level,
                                                                  phase_shift,
                                                                  f_mod,
                                                                  f_delta,
                                                                  self.recovery_file['name'])       
                                output_df.loc[len(output_df)] = [output_file_path, noise_level, phase_shift, f_mod_value, f_delta_value]
                                recovery_df.loc[len(recovery_df)] = [recovery_file_path, noise_level, phase_shift, f_mod_value, f_delta_value]

            input_df.to_pickle(input_df_file_path)
            output_df.to_pickle(output_df_file_path)
            recovery_df.to_pickle(recovery_df_file_path)
              
    def set_recovery_df(self, nyfr=None, filenames=None, directories=None, input_set_params=None):
        self.__set_init(nyfr=nyfr, filename=filenames, directories=directories, input_set_params=input_set_params)
        if self.__needs_init(include_set_params=True):
            print("NYFR Test Harness not properly initialized.  Please re-initialize object")
            return
        dictionary_params = self.nyfr.get_dictionary_params()
        recovery_params = self.nyfr.get_recovery_params()

        add_columns = False
        
        recovery_df_path = os.path.join(self.df_dir, self.recovery_file['df'])
        if os.path.exists(recovery_df_path):
            recovery_df = pd.read_pickle(recovery_df_path)
            for mode in recovery_params['modes']:
                recovery_base_path = self.recovery_dir[dictionary_params['version']][recovery_params['type']]
                if ( recovery_params['type'] == 'MLP1' ):
                    recovery_base_path = os.path.join(recovery_base_path,
                                                    mode)
                for noise_level, _ in self.input_set_params["noise_levels"]:
                    for phase_shift, _ in self.input_set_params["phase_shifts"]:
                        for input_tones, _ in self.input_set_params["input_tones"]:
                            input_list_path = os.path.join(self.input_dir,
                                                           noise_level,
                                                           phase_shift,
                                                           self.input_tones[input_tones]['list'])
                            for f_mod, f_mod_value in self.input_set_params["f_mods"]:
                                for f_delta, f_delta_value in self.input_set_params["f_deltas"]:
                                    recovery_file_path = os.path.join(recovery_base_path,
                                                                      noise_level,
                                                                      phase_shift,
                                                                      f_mod,
                                                                      f_delta,
                                                                      self.recovery_file['name'])
                                    recovery_sig_set = np.load(recovery_file_path)
                                    recovery_sig_set_size = 0
                                    if add_columns:
                                        recovery_sig_set_size = recovery_sig_set.shape[0]
                                        add_columns = False
                                        for stats in range(recovery_sig_set_size):
                                            min_spur_mag = "min_spur_mag_" + str(stats)
                                            recovery_df[min_spur_mag] = 0.0
                                            min_rec_mag = "min_rec_mag_" + str(stats)
                                            recovery_df[min_rec_mag] = 0.0
                                            pass
                                        recovery_df.to_pickle(recovery_df_path)

                                    current_recovery_row = recovery_df.index[(recovery_df['file_name']==self.input_tones[input_tones]['sigs']) &
                                                (recovery_df['noise_level']==noise_level) &
                                                (recovery_df['phase_shift']==phase_shift) &
                                                (recovery_df['f_mod']==f_mod_value) &
                                                (recovery_df['f_delta']==f_delta_value) &
                                                (recovery_df['dictionary_type']==dictionary_params['type']) &
                                                (recovery_df['recovery_method']==recovery_params['type'])]
                                    recovery_df = self.__update_recovery_df(recovery_df,                         
                                                                            recovery_sig_set,
                                                                            input_list_path,
                                                                            recovery_params["mag_thresh"],
                                                                            current_recovery_row,
                                                                            self.input_set_params["amp_min"])     
                        recovery_df.to_pickle(recovery_df_path)

    def __update_recovery_df(self, 
                             recovery_df,
                             recovery_sig_set,
                             input_list_path,
                             recovery_mag_thresh,
                             current_recovery_row,
                             input_tone_thresh):
        input_sig_params = pd.read_pickle(input_list_path)
        system_params = self.nyfr.get_system_params()
        orig_system_noise_level = system_params["system_noise_level"]
        system_params["system_noise_level"] = 0
        self.nyfr.set_system_params(system_params=system_params)

        if len(input_sig_params) != recovery_sig_set.shape[0]:
            print("Input signal set size does not match recovery signal set size")
            return recovery_df
        
        for idx,rec_sig in enumerate(recovery_sig_set):
            meta_data = {
                'num_rec_freq': {
                    'col_name': "num_rec_freq_" + str(idx),
                    'value': 0
                },
                'num_spur_freq': {
                    'col_name': "num_spur_freq_" + str(idx),
                    'value': 0
                },
                'ave_rec_mag_err': {
                    'col_name': "ave_rec_mag_err_" + str(idx),
                    'value': 0
                },
                'total_input_tones': {
                    'col_name': "total_input_tones_" + str(idx),
                    'value': 0
                },
                'rec_tone_thresh': {
                    'col_name': "rec_tone_thresh_" + str(idx),
                    'value': 0
                },
                'ave_rec_mag': {
                    'col_name': "ave_rec_mag_" + str(idx),
                    'value': 0
                },
                'max_rec_mag': {
                    'col_name': "max_rec_mag_" + str(idx),
                    'value': 0
                },
                'min_rec_mag': {
                    'col_name': "min_rec_mag_" + str(idx),
                    'value': 0
                },
                'ave_spur_mag': {
                    'col_name': "ave_spur_mag_" + str(idx),
                    'value': 0
                },
                'max_spur_mag': {
                    'col_name': "max_spur_mag_" + str(idx),
                    'value': 0
                },
                'min_spur_mag': {
                    'col_name': "min_spur_mag_" + str(idx),
                    'value': 0
                }
            }
            input_sig_param, _ = input_sig_params[idx]
            analog_input, _ = self.nyfr.create_input_signal(wave_params=input_sig_param)
            input_sig = self.nyfr.sample_signals(data=analog_input, sample_rate=self.nyfr.get_wb_nyquist_rate())
            input_sig_xf = fft(input_sig)
            input_sig_tones = np.where(abs(input_sig_xf) > input_tone_thresh)[0]
            input_tone_mag = np.abs(input_sig_xf)
            rec_sig_tones = np.where(abs(rec_sig) > recovery_mag_thresh)[0]
            mask = np.in1d(rec_sig_tones,input_sig_tones)
            recovered_freq = np.where(mask)[0]
            spur_freq = np.where(~mask)[0]
            recovered_tones = rec_sig_tones[recovered_freq]
            spur_tones = rec_sig_tones[spur_freq]
            rec_mag = abs(rec_sig[recovered_tones])
            spur_mag = abs(rec_sig[spur_tones])
            meta_data['num_rec_freq']['value'] = recovered_freq.size
            meta_data['num_spur_freq']['value'] = spur_freq.size
            meta_data['total_input_tones']['value'] = input_sig_tones.size
            meta_data['rec_tone_thresh']['value'] = recovery_mag_thresh
            if ( recovered_freq.size == 0 ):
                meta_data['ave_rec_mag_err']['value'] = -1
                meta_data['ave_rec_mag']['value'] = -1
                meta_data['max_rec_mag']['value'] = -1
                meta_data['min_rec_mag']['value'] = -1
            else:
                meta_data['ave_rec_mag_err']['value'] = abs( np.average(input_tone_mag) - np.average(rec_mag) )
                meta_data['ave_rec_mag']['value'] = np.average(rec_mag)
                meta_data['max_rec_mag']['value'] = np.max(rec_mag)
                meta_data['min_rec_mag']['value'] = np.min(rec_mag)
            if ( spur_freq.size == 0 ):
                meta_data['ave_spur_mag']['value'] = -1
                meta_data['max_spur_mag']['value'] = -1
                meta_data['min_spur_mag']['value'] = -1
            else:
                meta_data['ave_spur_mag']['value'] = np.average(spur_mag)
                meta_data['max_spur_mag']['value'] = np.max(spur_mag)
                meta_data['min_spur_mag']['value'] = np.min(spur_mag)
            pass
            for data in meta_data:
                recovery_df.at[current_recovery_row[0], meta_data[data]['col_name']] = meta_data[data]['value']
        system_params["system_noise_level"] = orig_system_noise_level
        self.nyfr.set_system_params(system_params=system_params)
        return recovery_df

    def set_nyfr(self, nyfr):
        self.nyfr = nyfr

    def set_filenames(self, filenames=None):
        if filenames is None:
            filenames = {
                "input_df": "input_df.pkl",
                "output_df": "output_df.pkl",
                "input_tones": {
                    "1_2": {
                        "sigs": "1_2_tone_sigs.npy",
                        "list": "1_2_tone_list.pkl"
                    },
                    "3": {
                        "sigs": "3_tone_sigs.npy", 
                        "list": "3_tone_list.pkl"
                    },
                    "4": {
                        "sigs": "4_tone_sigs.npy", 
                        "list": "4_tone_list.pkl"
                    },
                    "5": {
                        "sigs": "5_tone_sigs.npy", 
                        "list": "5_tone_list.pkl"
                    }
                },
                "dictionary": {
                    "name": "dictionary.npy"
                },
                "time": {
                    "name": "time.npy",
                    "frequency": "complex_tf.npy",
                    "sampled_freq": "complex_tf_sampled.npy"
                },
                "recovery":{
                    "df": "recovery_df.pkl",
                    "name": "recovery_list.txt",
                    "mag_ang": {
                        "mag": {
                            "system1": "recovery_list_system1_mag.txt",
                            "system2": "recovery_list_system2_mag.txt"
                        },
                        "ang": {
                            "system1": "recovery_list_system1_ang.txt",
                            "system2": "recovery_list_system2_ang.txt"
                        }
                    },
                    "active_zones": {
                        "system1": "recovery_list_system1_active_zones.txt",
                        "system2": "recovery_list_system2_active_zones.txt"
                    }
                },
                "mlp_models": {
                    "name" : "mlp_model_file.keras",
                    "log": {
                        "name": "input_list.txt",
                        "system1": "input_list_system1.txt",
                        "system2": "input_list_system2.txt"
                    }
                }
            }

        self.dictionary_file = filenames['dictionary']
        self.time_file = filenames['time']
        self.recovery_file = filenames['recovery']
        self.input_df_file = filenames['input_df']
        self.output_df_file = filenames['output_df']
        self.mlp_models_file = filenames['mlp_models']
        self.input_tones = filenames['input_tones']

    def set_input_set_params(self, input_set_params=None, input_set_json=None):
        if input_set_json is not None:
            input_set_params = load_settings(input_set_json)
        elif input_set_params is None:
            input_set_params = {
                "input_config_name": "Input_Config_1",
                "tot_num_freq_combos": 40000,
                "amp_min": 0.5,
                "amp_max": 1,
                "noise_levels": [["no_noise", []], ["low_noise", [0.2, 0.3]], ["high_noise", [2, 3]]],
                "phase_shifts": [["no_phase_shift", []], ["low_phase_shift",[0.05,0.25]], ["high_phase_shift", [0.4,0.75]]],
                "f_mods": [["f_mod_0_1", 0.1], ["f_mod_0_2", 0.2], ["f_mod_0_25", 0.25], ["f_mod_0_5", 0.5]],
                "f_deltas": [["f_delta_0_1", 0.1], ["f_delta_0_8", 0.8], ["f_delta_1_2", 1.2], ["f_delta_9_9", 10.0]],
                "input_tones": [["1_2", [1,2]], ["3", [3]], ["4", [4]], ["5", [5]]]     
            }
        self.input_set_params = input_set_params

    def set_directories(self, directories=None):
        if directories is None:
            directories = {
                "system_config_name": "System_Config_1",
                "input_config_name": "Input_Config_1",
                "time": ["test_sets", "System_Config_1", "Internal", "System", "Not_Sampled"],
                "time_sampled": ["test_sets", "System_Config_1", "Internal", "System", "Sampled"],
                "input": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "Inputs"],
                "output": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "Outputs"],
                "train": ["test_sets", "System_Config_1", "Model_Inputs", "train"],
                "test": ["test_sets", "System_Config_1", "Model_Inputs", "test"],
                "fft": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "Model_Inputs", "fft"],
                "active_zones": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "Model_Inputs", "active_zones"],
                "df": ["F:\\", "test_sets"],
                "dictionary": {
                    "enhanced": ["F:\\", "test_sets", "System_Config_1", "Dictionaries", "enhanced"],
                    "original": ["F:\\", "test_sets", "System_Config_1", "Dictionaries", "original"]
                },
                "recovery": {
                    "enhanced": {
                    },
                    "original": {
                        "OMP_Custom": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "Recovery", "OMP_Custom", "original"],
                        "OMP": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "Recovery", "OMP", "original"],
                        "MLP1": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "Recovery", "MLP1", "original"],
                        "SPGL1": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "Recovery", "SPGL1", "original"]
                    }
                },
                "mlp_models": {
                    "enhanced": {
                        "real": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "enhanced", "real"],
                        "imag": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "enhanced", "imag"],
                        "mag": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "enhanced", "mag"],
                        "ang": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "enhanced", "ang"],
                        "complex": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "enhanced", "complex"],
                        "active_zones": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "enhanced", "active_zones"]
                    },
                    "original": {
                        "real": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "original", "real"],
                        "imag": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "original", "imag"],
                        "mag": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "original", "mag"],
                        "ang": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "original", "ang"],
                        "complex": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "original", "complex"],
                        "active_zones": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "MLP_Models", "original", "active_zones"]
                    }
                },
                "premultiply": {
                    "enhanced": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "PreMultiply", "enhanced"],
                    "original": ["F:\\", "test_sets", "System_Config_1", "Input_Config_1", "PreMultiply", "original"]
                }
            }

        self.system_config_name = directories['system_config_name']
        self.input_dir = os.path.join(*directories['input'])
        self.output_dir = os.path.join(*directories['output'])
        self.fft_dir = os.path.join(*directories['fft'])
        self.time_dir = os.path.join(*directories['time'])
        self.time_sampled_dir = os.path.join(*directories['time_sampled'])
        self.df_dir = os.path.join(*directories['df'])
        self.active_zones_dir = os.path.join(*directories['active_zones'])

        recovery = directories['recovery']
        for type in recovery:
            if recovery[type]:
                for algorithm in recovery[type]:
                    recovery[type][algorithm] = os.path.join(*recovery[type][algorithm])
        self.recovery_dir = recovery

        mlp_models = directories['mlp_models']
        for type in mlp_models:
            if mlp_models[type]:
                for mode in mlp_models[type]:
                    mlp_models[type][mode] = os.path.join(*mlp_models[type][mode])
        self.mlp_models_dir = mlp_models

        premultiply = directories['premultiply']
        for type in premultiply:
            premultiply[type] = os.path.join(*premultiply[type])
        self.premultiply_dir = premultiply

        dictionary = directories['dictionary']
        for type in dictionary:
            dictionary[type] = os.path.join(*dictionary[type])
        self.dictionary_dir = dictionary

    def __set_init(self, nyfr=None, filenames=None, directories=None, input_set_params=None):
        if filenames is not None:
            self.set_filenames(filenames=filenames)

        if directories is not None:
            self.set_directories(directories=directories)

        if input_set_params is not None:
            self.set_input_set_params(input_set_params=input_set_params)

        if nyfr is not None:
            self.nyfr = nyfr

    def __needs_init(self, include_set_params=False):
        needs_init = False
        filenames = self.get_filenames()
        directories = self.get_directories()
        input_set_params = self.get_input_set_params()
        if not filenames:
            needs_init = True

        if not directories:
            needs_init = True

        if include_set_params:
            if not input_set_params:
                needs_init = True

        if self.nyfr is None:
            needs_init = True

        return needs_init

    def get_nyfr(self):
        return self.nyfr

    def get_filenames(self):
        filenames = {}
        filenames['dictionary'] = self.dictionary_file
        filenames['time'] = self.time_file
        filenames['recovery'] = self.recovery_file
        filenames['input_df'] = self.input_df_file
        filenames['output_df'] = self.output_df_file
        filenames['mlp_models'] = self.mlp_models_file
        filenames['input_tones'] = self.input_tones
        return filenames
    
    def get_directories(self):
        directories = {}
        directories['system_config_name'] = self.system_config_name
        directories['input'] = self.input_dir
        directories['output'] = self.output_dir
        directories['fft'] = self.fft_dir
        directories['time'] = self.time_dir
        directories['time_sampled'] = self.time_sampled_dir
        directories['dictionary'] = self.dictionary_dir
        directories['recovery'] = self.recovery_dir
        directories['active_zones'] = self.active_zones_dir
        directories['mlp_models'] = self.mlp_models_dir
        directories['premultiply'] = self.premultiply_dir
        return directories

    def get_input_set_params(self):
        return self.input_set_params

    def get_recovery_params(self):
        return self.nyfr.get_recovery_params()

    def get_dictionary_params(self):
        return self.nyfr.get_dictionary_params()
    
    def get_real_time(self):
        return self.nyfr.get_real_time()
    
    def get_frequncy_bins(self):
        return self.nyfr.get_frequncy_bins()
    
    def get_sampled_freq_bins(self):
        return self.nyfr.get_sampled_freq_bins()

    def get_system_params(self):
        return self.nyfr.get_system_params()