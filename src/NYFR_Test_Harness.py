from utility import get_all_file_paths, get_all_sub_dirs
from utility import load_settings, get_file_sub_dirs
from utility import delete_lines_with_string
import numpy as np
import pandas as pd
from numpy import sin
from scipy.fftpack import fft
import os
import time
import random
from itertools import combinations

class NYFR_Test_Harness:
    def __init__(self,
                 filenames=None,
                 directories=None,
                 input_set_params=None, 
                 filenames_json=None,
                 directories_json=None,
                 input_set_json=None,
                 nyfr=None) -> None:
        if filenames_json is not None:
            filenames = load_settings(filenames_json)         
        
        if directories_json is not None:
            directories = load_settings(directories_json)

        if input_set_json is not None:
            input_set_params = load_settings(input_set_json)

        self.set_filenames(filenames=filenames)
        self.set_input_set_params(input_set_params=input_set_params)
        self.set_directories(directories=directories)
        self.nyfr = nyfr

    def set_filenames(self, filenames=None):
        if filenames is None:
            print("No file names provided. Adding new file names")
            input_df_filename = input("Enter input data frame file name: ")
            output_df_filename = input("Enter output data frame file name: ")
            recovery_df_filename = input("Enter recovery data frame file name: ")
            dictionary_base_name = input("Enter dictionary file base name: ")
            time_base_name = input("Enter time file base name: ")
            frequency_base_name = input("Enter frequency file base name: ")
            sampled_frequency_base_name = input("Enter sampled frequency file base name: ")
            recovery_base_name = input("Enter recovery file base name: ")
            mlp_model_file_name = input("Enter MLP model file name: ")
            mlp_log_file_name = input("Enter MLP log file name: ")

            add_recovery_mode = True
            recovery_modes = []
            total_recovery_modes = 0
            while add_recovery_mode and total_recovery_modes < 3:
                recovery_modes.append(input("Enter recovery mode (mag_ang, real_imag, complex): "))
                add_recovery_mode = input("Add another recovery mode? (y/n): ").lower() == 'y'
                total_recovery_modes += 1

            add_processing_systems = True
            processing_systems = []
            while add_processing_systems:
                processing_systems.append(input("Enter processing systems: "))
                add_processing_systems = input("Add another processing system? (y/n): ").lower() == 'y'

            add_input_tone_file_name = True
            input_tones = {}
            while add_input_tone_file_name:
                num_input_tones = input("Enter number of input tones (1_2/3/4/5): ")
                input_tones[num_input_tones] = num_input_tones + "_tone_sigs.npy"
                add_input_tone_file_name = input("Add another input signal filename? (y/n): ").lower() == 'y'

            mlp_log_name = os.path.splitext(mlp_log_file_name)[0]
            mlp_log_extension = os.path.splitext(mlp_log_file_name)[1]
            mlp_models = {
                "name": mlp_model_file_name,
                "log": {
                    "name": mlp_log_file_name
                }
            }
            for processing_system in processing_systems:
                mlp_models["log"][processing_system] = mlp_log_name + "_" + processing_system + mlp_log_extension

            recovery_file = {
                "df": recovery_df_filename,
                "name": recovery_base_name
            }
            for recovery_mode in recovery_modes:
                recovery_file[recovery_mode] = {}
                sub_modes = []
                if recovery_mode == 'mag_ang':
                    sub_modes = [ 'mag', 'ang' ]
                elif recovery_mode == 'real_imag':
                    sub_modes = [ 'real', 'imag' ]

                if sub_modes == []:
                    for processing_system in processing_systems:
                        recovery_file[recovery_mode][processing_system] = "recovery_list_" + processing_system + "_complex.txt"
                else:
                    for sub_mode in sub_modes:
                        recovery_file[recovery_mode][sub_mode][processing_system] = "recovery_list_" + processing_system + "_" + sub_mode + ".txt"

            self.dictionary_file = {
                "name": dictionary_base_name
            }
            self.time_file = {
                "name": time_base_name,
                "frequency": frequency_base_name,
                "sample_freq": sampled_frequency_base_name
            }
            self.recovery_file = recovery_file           
            self.input_df_file = input_df_filename
            self.output_df_file = output_df_filename
            self.mlp_models_file = mlp_models
            self.input_tones = input_tones
        else:
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
                "tot_num_freq_combos": 79800,
                "amp_min": 0.5,
                "amp_max": 1,
                "high_phase_max": 0.75,
                "high_phase_min": 0.4,
                "low_phase_max": 0.25,
                "low_phase_min": 0.05,
                "high_noise_max": 3,
                "high_noise_min": 2,
                "low_noise_max": 0.3,
                "low_noise_min": 0.2           
            }
        self.input_set_params = input_set_params

    def set_directories(self, directories=None):
        if directories is None:
            print("No directory names provided.  Adding new directory names")
            system_config_name = input("Enter system configuration name: ")
            input_dir = input("Enter input directory name: ")
            output_dir = input("Enter output directory name: ")
            fft_dir = input("Enter frequency file base name: ")
            time_dir = input("Enter time directory: ")
            time_sampled_dir = input("Enter sampled time directory: ")
            df_dir = input("Enter data frame directory: ")

            dictionary_versions = []
            total_dictionary_versions = 0
            add_dictionary_version = True
            while add_dictionary_version and total_dictionary_versions < 2:
                dictionary_versions.append(input("Add dictionary version (enhanced/original): "))
                add_dictionary_version = input("Add another dictionary version? (y/n): ").lower() == 'y'
                total_recovery_modes += 1
            
            recovery_types = []
            total_recovery_types = 0
            add_recovery_types = True
            while add_recovery_types and total_recovery_types < 4:
                recovery_types.append(input("Add recovery type (OMP_Custom/OMP/MLP1/SPGL1): "))
                add_recovery_types = input("Add another dictionary version? (y/n): ").lower() == 'y'
                total_recovery_types += 1

            mlp_model_types = [ "real", "imag", "mag", "ang", "complex"]
            self.system_config_name = system_config_name
            self.input_dir = "test_sets\\" + system_config_name + "\\" + self.input_set_params['input_config_name'] + "\\" + input_dir + "\\" 
            self.output_dir = "test_sets\\" + system_config_name + "\\" + self.input_set_params['input_config_name'] + "\\" + output_dir + "\\"
            self.fft_dir = fft_dir
            self.time_dir = time_dir
            self.time_sampled_dir = time_sampled_dir
            self.df_dir = df_dir
            recovery = {}
            dictionary = {}
            mlp_models = {}
            for dictionary_version in dictionary_versions:
                dictionary[dictionary_version] = "test_sets\\" \
                    + self.system_config_name + "\\Internal\\" + dictionary_version + "\\Dictionary\\"
                for mlp_model_type in mlp_model_types:
                    mlp_models[dictionary_version][mlp_model_type] = "F:\\test_sets" \
                        + self.system_config_name + "\\MLP_Models\\" + dictionary_version + "\\" + mlp_model_type + "\\"
                for recovery_type in recovery_types:
                    recovery[dictionary_version][recovery_type] = "test_sets\\" \
                        + self.system_config_name + "\\Recovery\\" + dictionary_version + "\\" + recovery_type + "\\"
            self.mlp_models_dir = mlp_models
            self.dictionary_dir = dictionary
            self.recovery_dir = recovery
        else:
            self.system_config_name = directories['system_config_name']
            self.input_dir = directories['input']
            self.output_dir = directories['output']
            self.fft_dir = directories['fft']
            self.time_dir = directories['time']
            self.time_sampled_dir = directories['time_sampled']
            self.dictionary_dir = directories['dictionary']
            self.recovery_dir = directories['recovery']
            self.df_dir = directories['df']
            self.mlp_models_dir = directories['mlp_models']

    def get_filenames(self):
        filenames = {}
        filenames['dictionary'] = self.dictionary_file
        filenames['time'] = self.time_file
        filenames['recovery'] = self.recovery_file
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
        return directories

    def get_input_set_params(self):
        return self.input_set_params

    def __set_init(self, nyfr, filenames, directories, input_set_params=None):
        mod_delta_table = None
        need_init = False
        if self.recovery_file is None:
            if filenames is not None:
                self.set_filenames(filenames=filenames)
            else:
                print("File names need to be initialized")
                need_init = True

        if self.recovery_dir is None:
            if directories is not None:
                self.set_directories(directories=directories)
            else:
                print("File names need to be initialized")
                need_init = True

        if self.set_input_set_params is None:
            if input_set_params is not None:
                self.set_input_set_params(input_set_params=input_set_params)
            else:
                print("Input set params need to be initialized")
                need_init = True
        
        if self.nyfr is None:
            if nyfr is not None:
                self.nyfr = nyfr
            else:
                print("The NYFR needs to be initialized")
                need_init = True

        if not need_init:
            mod_delta_table = {
                "f_mod_0_1": 0.1,
                "f_mod_0_2": 0.2,
                "f_mod_0_25": 0.25,
                "f_mod_0_5": 0.5,
                "f_delta_0_1": 0.1,
                "f_delta_0_8": 0.8,
                "f_delta_1_2": 1.2,
                "f_delta_9_9": 10,                        
            }
        return mod_delta_table

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
                    delta_dir = f_delta[1]
                    dictionary_base_path = self.dictionary_dir[dictionary_params['version']] + "\\" + mod_dir + "\\" + delta_dir + "\\"
                    dictionary_file_path = os.path.join(dictionary_base_path, self.dictionary_file['name'])
                    dictionary = nyfr.create_dict()
                    np.save(dictionary_file_path, dictionary)

    def create_input_sets(self, nyfr=None, filenames=None, directories=None, input_set_params=None):
        _ = self.__set_init(nyfr, filenames, directories, input_set_params=input_set_params)
        system_params = self.nyfr.get_system_params()
        wbf_cut_freq = system_params['adc_clock_freq'] * system_params['wbf_cut_mod']
        pos_bins = list(range(1, wbf_cut_freq))
        for num_of_sigs, tone_sigs_file_name in self.input_tones.items():
                input_freq_tot_list = []
                match num_of_sigs:
                    case '1_2':
                        for total_active_sig in range(1,2):
                            input_freq_tot_list += list(combinations(pos_bins, total_active_sig))
                            random.shuffle(input_freq_tot_list)
                    case '3':
                        input_freq_tot_list += list(combinations(pos_bins, 3))
                        random.shuffle(input_freq_tot_list)
                        input_freq_tot_list = input_freq_tot_list[0:input_set_params['tot_num_freq_combos']]
                    case '4':
                        input_freq_tot_list = [(random.randint(1, wbf_cut_freq),
                                               random.randint(1, wbf_cut_freq),
                                               random.randint(1, wbf_cut_freq),
                                               random.randint(1, wbf_cut_freq)) for _ in range(input_set_params['tot_num_freq_combos'])]
                    case '5':
                        input_freq_tot_list = [(random.randint(1, wbf_cut_freq),
                                               random.randint(1, wbf_cut_freq),
                                               random.randint(1, wbf_cut_freq),
                                               random.randint(1, wbf_cut_freq),
                                               random.randint(1, wbf_cut_freq)) for _ in range(input_set_params['tot_num_freq_combos'])]
                    case _:
                        print("Unsupported right now")
                        break

                input_sub_dirs = get_all_sub_dirs(self.input_dir)
                for sub_dir in input_sub_dirs:
                    input_file_path = os.path.join(sub_dir, tone_sigs_file_name)
                    noise_level, phase_shift, _ = get_file_sub_dirs(input_file_path)
                    input_list = []
                    for input_freqs in input_freq_tot_list:
                        wave_params = []
                        for input_freq in input_freqs:
                            wave_param = {
                                "amp": random.uniform(input_set_params['amp_min'], input_set_params['amp_max']),
                                "freq": input_freq,
                                "phase": 0
                            }
                            if phase_shift == "high_phase_shift":
                                wave_param['phase'] = random.uniform(input_set_params['high_phase_min'], input_set_params['high_phase_max']) / input_freq
                            if phase_shift == "low_phase_shift":
                                wave_param['phase'] = random.uniform(input_set_params['low_phase_min'], input_set_params['low_phase_max']) / input_freq
                            wave_params.append(wave_param)

                        noise = 0
                        if noise_level == "high_noise":
                            noise = random.uniform(input_set_params['high_noise_min'], input_set_params['high_noise_max'])
                        elif noise_level == "low_noise":
                            noise = random.uniform(input_set_params['low_noise_min'], input_set_params['low_noise_max'])
                        system_params['system_noise_level'] = noise
                        nyfr.set_system_params(system_params=system_params)
                        input, _ = nyfr.create_input_signal(wave_params=wave_params)
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
                    nyfr.set_LO_params(LO_params=LO_params)
                    output_sub_dir = output_sub_path + f_mod + "\\" + f_delta + "\\"
                    output_file_path = (os.path.join(output_sub_dir, file_name))
                    if ( os.path.isfile(output_file_path) ):
                        os.remove( output_file_path )
                    for input in input_set:
                        output_list.append( nyfr.simulate_system(input_signal=input) )

                    output_set = np.array(output_list)
                    np.save(output_file_path, output_set)
                    output_list.clear()

    def batch_recover(self, nyfr=None, filenames=None, directories=None, recovery_set_size=100, get_recovery_time=False):
        _ = self.__set_init(nyfr, filenames, directories)
        system_params = self.nyfr.get_system_params()
        dictionary_params = self.nyfr.get_dicionary_params()
        recovery_params = self.nyfr.get_recovery_params()

        input_file_paths = get_all_file_paths(self.input_dir)
        recovery_set = np.zeros((recovery_set_size, nyfr.get_num_time_points()), dtype=np.complex128)
        for mode in recovery_params['modes']:
            recovery_base_path = self.recovery_dir[dictionary_params['version']][recovery_params['type']]
            if ( recovery_params['type'] == 'MLP1' ):
                recovery_base_path = recovery_base_path + mode + "\\"

            if ( mode == 'real_imag' ):
                mlp_models_base_path = self.mlp_models[dictionary_params['version']]['real']
                mlp_models_base_path_aux = self.mlp_models[dictionary_params['version']]['imag']
            elif ( mode == 'mag_ang' ):
                mlp_models_base_path = self.mlp_models[dictionary_params['version']]['mag']
                mlp_models_base_path_aux = self.mlp_models[dictionary_params['version']]['ang']
            elif ( mode == 'complex' ):
                mlp_models_base_path = self.mlp_models[dictionary_params['version']]['complex']
                mlp_models_base_path_aux = None
            
            for input_file_path in input_file_paths:
                noise_level, phase_shift, file_name = get_file_sub_dirs(input_file_path)
                output_sub_path = self.output_dir + noise_level + "\\" + phase_shift + "\\"
                recovery_sub_path = recovery_base_path + noise_level + "\\" + phase_shift + "\\"
                mlp_models_sub_path = mlp_models_base_path + noise_level + "\\" + phase_shift + "\\"
                if mlp_models_base_path_aux is not None:
                    mlp_models_sub_path_aux = mlp_models_base_path_aux + noise_level + "\\" + phase_shift + "\\"
                output_file_sub_dirs = get_all_sub_dirs(output_sub_path)
                for processing_system in system_params['processing_systems']:
                    if ( mode == 'real_imag' ):
                        recovery_log_file_path = self.recovery_dir + self.recovery_file[mode]['real']
                        recovery_log_file_path_aux = self.recovery_dir + self.recovery_file[mode]['imag']
                    elif ( mode == 'mag_ang' ):
                        recovery_log_file_path = self.recovery_dir + self.recovery_file[mode]['mag']
                        recovery_log_file_path_aux = self.recovery_dir + self.recovery_file[mode]['ang']
                    elif ( mode == 'complex' ):
                        recovery_log_file_path = self.recovery_dir + self.recovery_file[mode]['complex']
                        recovery_log_file_path_aux = None

                    for sub_dir in output_file_sub_dirs:
                        output_file_path = os.path.join(sub_dir, file_name)
                        f_mod, f_delta, _ = get_file_sub_dirs(output_file_path)
                        mlp_model_dir = mlp_models_sub_path + f_mod + "\\" + f_delta + "\\"
                        if mlp_models_base_path_aux is not None:
                            mlp_model_aux_dir = mlp_models_sub_path_aux + f_mod + "\\" + f_delta + "\\"
                        recovery_dir = recovery_sub_path + f_mod + "\\" + f_delta + "\\"
                        mlp_model_file_path = os.path.join(mlp_model_dir, self.mlp_models_file['name'])
                        if mlp_models_base_path_aux is not None:
                            mlp_model_aux_file_path = os.path.join(mlp_model_aux_dir, self.mlp_models_file['name'])
                        else:
                            mlp_model_aux_file_path = None

                        dictionary_file_path = os.path.join(self.dictionary_dir[dictionary_params['version']], self.dictionary_file['name'])
                        recovery_file_path = os.path.join(recovery_dir, file_name)
                        found_string_in_file = False
                        found_string_in_file_aux = False
                        with open(recovery_log_file_path, "r") as recovery_log:
                            for line in recovery_log:
                                if output_file_path in line:
                                    found_string_in_file = True
                                    break
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
                                    recovered_signal = nyfr.recover_signal(dictionary,
                                                                           output_set[idx],
                                                                           file_path=mlp_model_file_path,
                                                                           aux_file_path=mlp_model_aux_file_path,
                                                                           mode=mode)
                                    recovery_set[idx] = recovered_signal
                                if get_recovery_time:
                                    end_time = time.perf_counter()
                                    ave_recovery_time = ( end_time - start_time ) / recovery_set_size
                                np.save(recovery_file_path, recovery_set)
                                recovery_set.fill(0)
                            delete_lines_with_string(recovery_log_file_path, output_file_path)

    def create_dfs(self, nyfr=None, filenames=None, directories=None):
        mod_delta_table = self.__set_init(nyfr, filenames, directories)
        dictionary_params = self.nyfr.get_dicionary_params()
        recovery_params = self.nyfr.get_recovery_params()

        if mod_delta_table is not None:
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
            input_file_paths = get_all_file_paths(self.input_dir)
            for input_file_path in input_file_paths:
                noise_level, phase_shift, file_name = get_file_sub_dirs(input_file_path)
                match file_name:
                    case '1_2_tone_sigs.npy':
                        num_tones = [1,2]
                    case '3_tone_sigs.npy':
                        num_tones = [3]
                    case '4_tone_sigs.npy':
                        num_tones = [4]
                    case '5_tone_sigs.npy':
                        num_tones = [5]
                    case _:
                        num_tones = []
                input_df.loc[len(input_df)] = [file_name, noise_level, phase_shift, num_tones, 0]
                output_base_dir = self.output_dir + noise_level + "\\" + phase_shift + "\\"
                recovery_base_dir = self.recovery_dir[dictionary_params['version']][recovery_params['type']] + noise_level + "\\" + phase_shift + "\\"
                output_file_sub_dirs = get_all_sub_dirs(output_base_dir)
                for sub_dir in output_file_sub_dirs:
                    output_file_path = os.path.join(sub_dir, file_name)
                    f_mod, f_delta, _ = get_file_sub_dirs(output_file_path)
                    recovery_dir = recovery_base_dir + f_mod + "\\" + f_delta + "\\"
                    recovery_file_path = os.path.join(recovery_dir, file_name)
                    output_df.loc[len(output_df)] = [output_file_path, noise_level, phase_shift, mod_delta_table[f_mod], mod_delta_table[f_delta]]
                    recovery_df.loc[len(recovery_df)] = [recovery_file_path, noise_level, phase_shift, mod_delta_table[f_mod], mod_delta_table[f_delta]]
                    pass
            input_df.to_pickle(self.input_df_file)
            output_df.to_pickle(self.output_df_file)
            recovery_df.to_pickle(self.recovery_file['df'])
              
    def set_recovery_df(self, nyfr, filenames=None, directories=None):
        mod_delta_table = self.__set_init(nyfr, filenames, directories)
        dictionary_params = self.nyfr.get_dicionary_params()
        recovery_params = self.nyfr.get_recovery_params()
        if mod_delta_table is not None:
            add_columns = 0
            recovery_sig_set_size = 0
            input_tone_thresh = 600
            recovery_mag_thresh = 2
            recovery_df_path = os.path.join(self.recovery_dir['df'], self.recovery_file['df'])
            if os.path.exists(recovery_df_path):
                recovery_df = pd.read_pickle(recovery_df_path)
                input_file_paths = get_all_file_paths(self.input_dir)
                for input_file_path in input_file_paths:
                    file_name, noise_level, phase_shift = get_file_sub_dirs(input_file_path)
                    recovery_sub_path = self.recovery_dir[dictionary_params['version']] + noise_level + "\\" + phase_shift + "\\"
                    input_sig_set = np.load(input_file_path)
                    recovery_file_sub_dirs = get_all_sub_dirs(recovery_sub_path)
                    for sub_dir in recovery_file_sub_dirs:
                        recovery_file_path = os.path.join(sub_dir, file_name)
                        _, f_mod, f_delta = get_file_sub_dirs(recovery_file_path)
                        recovery_sig_set = np.load(recovery_file_path)
                        if ( add_columns == 1 ):
                            recovery_sig_set_size = recovery_sig_set.shape[0]
                            add_columns = 0
                            for stats in range(recovery_sig_set_size):
                                min_spur_mag = "min_spur_mag_" + str(stats)
                                recovery_df[min_spur_mag] = 0.0
                                min_rec_mag = "min_rec_mag_" + str(stats)
                                recovery_df[min_rec_mag] = 0.0
                                pass
                            recovery_df.to_pickle(recovery_df_path)

                        current_recovery_row = recovery_df.index[(recovery_df['file_name']==file_name) &
                                    (recovery_df['noise_level']==noise_level) &
                                    (recovery_df['phase_shift']==phase_shift) &
                                    (recovery_df['f_mod']==mod_delta_table[f_mod]) &
                                    (recovery_df['f_delta']==mod_delta_table[f_delta]) &
                                    (recovery_df['dictionary_type']==dictionary_params['type']) &
                                    (recovery_df['recovery_method']==recovery_params['type'])]
                        recovery_df = self.__update_recovery_df(recovery_df,                         
                                                                    recovery_sig_set,
                                                                    input_sig_set,
                                                                    input_tone_thresh,
                                                                    recovery_mag_thresh,
                                                                    current_recovery_row)     
            recovery_df.to_pickle(recovery_df_path)

    def __update_recovery_df(recovery_df,
                                  recovery_sig_set,
                                  input_sig_set,
                                  input_tone_thresh,
                                  recovery_mag_thresh,
                                  current_recovery_row):
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
            input_sig_xf = fft(input_sig_set[idx])
            input_sig_tones = np.where(abs(input_sig_xf) > input_tone_thresh)[0]
            input_tone_mag = abs(input_sig_xf[input_sig_tones])
            rec_sig_tones = np.where(abs(rec_sig) > recovery_mag_thresh)[0]
            mask = np.isin(rec_sig_tones,input_sig_tones)
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
            
            return recovery_df