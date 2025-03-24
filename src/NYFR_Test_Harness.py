from utility import get_all_file_paths, get_all_sub_dirs
from utility import load_settings, get_file_sub_dirs
import numpy as np
import pandas as pd
from numpy import sin
from scipy.fftpack import fft
import os


class NYFR_Test_Harness:
    def __init__(self,
                 filenames=None,
                 directories=None,
                 filenames_json=None,
                 directories_json=None) -> None:
        if filenames_json is not None:
            filenames = load_settings(filenames_json)         
        
        if directories_json is not None:
            directories = load_settings(directories_json)

        self.set_filenames(filenames=filenames)
        self.set_directories(directories=directories)

    def set_filenames(self, filenames=None):
        if filenames is None:
            print("No file names provided. Adding new file names")
            dictionary_base_name = input("Enter dictionary file base name: ")
            time_base_name = input("Enter time file base name: ")
            frequency_base_name = input("Enter frequency file base name: ")
            sampled_frequency_base_name = input("Enter sampled frequency file base name: ")
            recovery_base_name = input("Enter recovery file base name: ")
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

            recovery_file = {
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

        else:
            self.dictionary_file = filenames['dictionary']
            self.time_file = filenames['time']
            self.recovery_file = filenames['recovery']

    def set_directories(self, directories=None):
        if directories is None:
            print("No directory names provided.  Adding new directory names")
            system_config_name = input("Enter system configuration name: ")
            input_dir = input("Enter input directory: ")
            output_dir = input("Enter output directory: ")
            fft_dir = input("Enter frequency file base name: ")
            time_dir = input("Enter time directory: ")
            time_sampled_dir = input("Enter sampled time directory: ")
            
            dictionary_versions = []
            total_dictionary_versions = 0
            add_dictionary_version = True
            while add_dictionary_version and total_dictionary_versions < 2:
                dictionary_versions.append(input("Add dictionary version (enhanced/original): "))
                add_dictionary_version = input("Add another dictionary version? (y/n): ").lower() == 'y'
                total_recovery_modes += 1
            dictionary_versions = []
            
            recovery_types = []
            total_recovery_types = 0
            add_recovery_types = True
            while add_recovery_types and total_recovery_types < 4:
                recovery_types.append(input("Add recovery type (OMP_Custom/OMP/MLP1/SPGL1): "))
                add_recovery_types = input("Add another dictionary version? (y/n): ").lower() == 'y'
                total_recovery_types += 1

            self.system_config_name = system_config_name
            self.input_dir = input_dir
            self.output_dir = output_dir
            self.fft_dir = fft_dir
            self.time_dir = time_dir
            self.time_sampled_dir = time_sampled_dir
            
            recovery = {}
            dictionary = {}
            for dictionary_version in dictionary_versions:
                dictionary[dictionary_version] = "test_sets\\" + self.system_config_name + "\\Internal\\" + dictionary_version + "\\Dictionary\\"
                for recovery_type in recovery_types:
                    recovery[dictionary_version][recovery_type] = "test_sets\\" + self.system_config_name + "\\Recovery\\" + dictionary_version + "\\" + recovery_type + "\\"
            
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

    def create_dictionaries(self, nyfr):
        LO_params = nyfr.get_LO_params()
        dictionary_params = nyfr.get_dictionary_params()
        if LO_params == None or nyfr.get_K_band() == None or dictionary_params == None:
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

    def __update_df_for_analysis(recovery_df,
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
            
    def analyze_dfs(self, nyfr, filenames=None, directories=None):
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

        dictionary_params = nyfr.get_dictionary_params()
        if dictionary_params is None:
            print("NYFR dictionary not initialized")
            need_init = True

        recovery_params = nyfr.get_recovery_params()
        if recovery_params is None:
            print("NYFR dictionary parameters not initialized")
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
                        recovery_df = self.__update_df_for_analysis(recovery_df,                         
                                                                    recovery_sig_set,
                                                                    input_sig_set,
                                                                    input_tone_thresh,
                                                                    recovery_mag_thresh,
                                                                    current_recovery_row)     
            recovery_df.to_pickle(recovery_df_path)

    def meta_input_output(system_params):
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
        input_df_filename = "test_sets\\input_df.pkl"
        output_df_filename = "test_sets\\output_df.pkl"
        recovery_df_filename = "test_sets\\recovery_df.pkl"
        # pickle_1 = pd.read_pickle(recovery_df_filename)
        # pickle_2 = pickle_1.copy()
        # pickle_1['dictionary_type'] = "original"
        # pickle_2['dictionary_type'] = "enhanced"
        # pickle_3 = pd.concat([pickle_1, pickle_2], ignore_index=True)
        # pickle_4 = pickle_3.copy()
        # pickle_3['recovery_method'] = "c_omp"
        # pickle_4['recovery_method'] = "spgl1"
        # pickle_5 = pd.concat([pickle_3, pickle_4], ignore_index=True)
        # pickle_5.to_pickle(recovery_df_filename)
        # system_params['recovery'] = 'omp'
        input_directory = "test_sets\\System_Config_1_Inputs\\"
        input_file_paths = get_all_file_paths(input_directory)
        output_base_directory_name = "System_Config_1_Outputs"
        recovery_base_directory_names = {
            'c_omp': 'System_Config_1_OMP_Custom_Recovery',
            'o_omp': 'System_Config_1_OMP_Recovery',
            'mlp1': 'System_Config_1_MLP1_Recovery',
            'spgl1': 'System_Config_1_SPGL_Recovery'
        }
        dictionary_base_directory = "test_sets\\System_Config_1_Internal\\Dictionary\\original\\"
        # dictionary_base_directory = "test_sets\\System_Config_1_Internal\\Dictionary\\"
        dictionary_file_list = get_all_file_paths(dictionary_base_directory)
        recovery_set_size = 100
        for input_set_file in input_file_paths:
            input_path = Path(input_set_file)
            input_path_len = len(input_path.parts)
            input_file_name = input_path.parts[input_path_len - 1]
            input_phase_shift = input_path.parts[input_path_len - 2]
            input_noise_level = input_path.parts[input_path_len - 3]
            match input_path.parts[4]:
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
            input_df.loc[len(input_df)] = [input_file_name, input_noise_level, input_phase_shift, num_tones, 0]
            input_path_list = input_set_file.split("\\")
            output_file_name = input_path_list.pop()
            output_base_path_list = input_path_list.copy()
            recovery_base_path_list = input_path_list.copy()
            output_base_path_list[1] = output_base_directory_name
            recovery_base_path_list[1] = recovery_base_directory_names[system_params['recovery']]
            output_base_dir = os.path.join(*output_base_path_list)
            output_file_sub_dirs = get_all_sub_dirs(output_base_dir)
            recovery_base_dir = "E:\\" + os.path.join(*recovery_base_path_list)
            recovery_file_sub_dirs = get_all_sub_dirs(recovery_base_dir)
            for index, sub_dir in enumerate(recovery_file_sub_dirs):
                output_file_path = os.path.join(output_file_sub_dirs[index % len(output_file_sub_dirs)], output_file_name)
                #output_sig_set = np.load(output_file_path)
                dictionary_file_name =  dictionary_file_list[index]
                #dictionary = np.load(dictionary_file_name)
                recovery_file_path = os.path.join(sub_dir, output_file_name)
                
                output_path = Path(output_file_path)
                output_file_name = output_path.parts[6]
                output_noise_level = output_path.parts[2]
                output_phase_shift = output_path.parts[3]
            
                match output_path.parts[4]:
                    case 'f_mod_0_1':
                        f_mod = 0.1
                    case 'f_mod_0_2':
                        f_mod = 0.2
                    case 'f_mod_0_25':
                        f_mod = 0.25
                    case 'f_mod_0_5':
                        f_mod = 0.5
                    case _:
                        f_mod = 0

                match output_path.parts[5]:
                    case 'f_delta_0_1':
                        f_delta = 0.1
                    case 'f_delta_0_8':
                        f_delta = 0.8
                    case 'f_delta_1_2':
                        f_delta = 1.2
                    case 'f_delta_9_9':
                        f_delta = 10
                    case _:
                        f_delta = 0

                output_df.loc[len(output_df)] = [output_file_name, output_noise_level, output_phase_shift, f_mod, f_delta]
                recovery_df.loc[len(recovery_df)] = [output_file_name, output_noise_level, output_phase_shift, f_mod, f_delta]
                pass
        input_df.to_pickle(input_df_filename)
        output_df.to_pickle(output_df_filename)
        recovery_df.to_pickle(recovery_df_filename)

    def batch_recover(system_params):
        # modes = [ 'real_imag', 'mag_ang', 'complex' ]
        modes = [ 'mag_ang' ]
        # processing_systems = [ 'daddo' ]
        processing_systems = [ 'bedroom' ]
        # mlp_model_file_name = "input_list.txt"
        recovery_log_file_name = {
            'mag_ang': {
                'mag':{
                    'bedroom': "recovery_list_bedroom_mag.txt",
                    'daddo': "recovery_list_daddo_mag.txt"
                },
                'ang':{
                    'bedroom': "recovery_list_bedroom_ang.txt",
                    'daddo': "recovery_list_daddo_ang.txt"
                },
            }
        }
        # recovery_log_file_name = "recovery_list_Bedroom_PC.txt"
        recovery_dic_type = 'original'
        directory_list = {
            'input': "test_sets\\System_Config_1_Inputs\\",
            'fft': "test_sets\\System_Config_1_Model_Inputs\\FFT\\",
            'train': "test_sets\\System_Config_1_Model_Inputs\\train\\",
            'test': "test_sets\\System_Config_1_Model_Inputs\\test\\",
            'output': "test_sets\\System_Config_1_Outputs\\",
            'mlp_models': {
                'file_name' : "mlp_model_file.keras",
                'enhanced': {
                    'real': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\real\\",
                    'imag': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\imag\\",
                    'mag': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\mag\\",
                    'ang': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\ang\\",
                    'complex': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\complex\\"
                },
                'original': {
                    'real': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\real\\",
                    'imag': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\imag\\",
                    'mag': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\mag\\",
                    'ang': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\ang\\",
                    'complex': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\complex\\"
                }
            },
            'dictionary': {
                'enhanced': "test_sets\\System_Config_1_Internal\\Dictionary\\enhanced\\",
                'original': "test_sets\\System_Config_1_Internal\\Dictionary\\original\\"
            },
            'recovery': {
                'enhanced': {
                    'c_omp': "test_sets\\System_Config_1_Recovery\\OMP_Custom\\enhanced\\",
                    'o_omp': "test_sets\\System_Config_1_Recovery\\OMP\\enhanced\\",
                    'mlp1': "test_sets\\System_Config_1_Recovery\\MLP1\\enhanced\\",
                    'spgl1': "test_sets\\System_Config_1_Recovery\\SPGL\\enhanced\\"
                },
                'original': {
                    'c_omp': "test_sets\\System_Config_1_Recovery\\OMP_Custom\\original\\",
                    'o_omp': "test_sets\\System_Config_1_Recovery\\OMP\\original\\",
                    'mlp1': "test_sets\\System_Config_1_Recovery\\MLP1\\original\\",
                    'spgl1': "test_sets\\System_Config_1_Recovery\\SPGL\\original\\"
                }
            }
        }
        input_file_paths = get_all_file_paths(directory_list['input'])
        dictionary_file_list = get_all_file_paths(directory_list['dictionary'][recovery_dic_type])
        test_sig_set = np.load(input_file_paths[0])
        recovery_set_size = 100
        recovery_sig_set = np.zeros((recovery_set_size,test_sig_set.shape[1]), dtype=np.complex128)
        use_complex = False
        for mode in modes:
            recovery_base_path = directory_list['recovery'][recovery_dic_type][system_params['recovery']]
            if ( system_params['recovery'] == 'mlp1' ):
                recovery_base_path = recovery_base_path + mode + "\\"

            if ( mode == 'real_imag' ):
                mlp_models_base_path = directory_list['mlp_models'][recovery_dic_type]['real']
                mlp_models_base_path_aux = directory_list['mlp_models'][recovery_dic_type]['imag']
            elif ( mode == 'mag_ang' ):
                mlp_models_base_path = directory_list['mlp_models'][recovery_dic_type]['mag']
                mlp_models_base_path_aux = directory_list['mlp_models'][recovery_dic_type]['ang']
            elif ( mode == 'complex' ):
                mlp_models_base_path = directory_list['mlp_models'][recovery_dic_type]['complex']
                use_complex = True
            
            for input_set_file in input_file_paths:
                input_path = Path(input_set_file)
                input_path_len = len(input_path.parts)
                input_file_name = input_path.parts[input_path_len - 1]
                input_phase_shift = input_path.parts[input_path_len - 2]
                input_noise_level = input_path.parts[input_path_len - 3]
                output_sub_path = directory_list['output'] + input_noise_level + "\\" + input_phase_shift + "\\"
                recovery_sub_path = recovery_base_path + input_noise_level + "\\" + input_phase_shift + "\\"
                mlp_models_sub_path = mlp_models_base_path + input_noise_level + "\\" + input_phase_shift + "\\"
                if ( not use_complex ):
                    mlp_models_sub_path_aux = mlp_models_base_path_aux + input_noise_level + "\\" + input_phase_shift + "\\"
                # input_sig_set = np.load(input_set_file)
                # test_sig_set = np.load(input_set_file)
                # recovery_sig_set = np.zeros_like(test_sig_set)
                # input_path_list = input_set_file.split("\\")
                # output_file_name = input_path_list.pop()
                # output_base_path_list = input_path_list.copy()
                # recovery_base_path_list = input_path_list.copy()
                # output_base_path_list[1] = output_base_directory_name
                # recovery_base_path_list[1] = recovery_base_directory_names[system_params['recovery']]
                # output_base_dir = os.path.join(*output_base_path_list)
                # output_file_sub_dirs = get_all_sub_dirs(output_base_dir)
                # recovery_base_dir = "E:\\" + os.path.join(*recovery_base_path_list)
                output_file_sub_dirs = get_all_sub_dirs(output_sub_path)
                recovery_file_sub_dirs = get_all_sub_dirs(recovery_sub_path)
                mlp_models_sub_dirs = get_all_sub_dirs(mlp_models_sub_path)
                for proc in processing_systems:
                    recovery_log_file_path_mag = directory_list['recovery'][recovery_dic_type][system_params['recovery']] \
                        + mode + "\\" + recovery_log_file_name[mode]['mag'][proc]
                    recovery_log_file_path_ang = directory_list['recovery'][recovery_dic_type][system_params['recovery']] \
                        + mode + "\\" + recovery_log_file_name[mode]['ang'][proc]
                    if ( not use_complex ):
                        mlp_models_sub_dirs_aux = get_all_sub_dirs(mlp_models_sub_path_aux)

                    for index, sub_dir in enumerate(recovery_file_sub_dirs):
                        output_file_path = os.path.join(output_file_sub_dirs[index], input_file_name)
                        mlp_model_file_path = os.path.join(mlp_models_sub_dirs[index], directory_list['mlp_models']['file_name'])
                        found_string_in_file_mag = False
                        found_string_in_file_ang = False
                        with open(recovery_log_file_path_mag, "r") as recovery_log:
                            for line in recovery_log:
                                if output_file_path in line:
                                    found_string_in_file_mag = True
                                    break
                        with open(recovery_log_file_path_ang, "r") as recovery_log:
                            for line in recovery_log:
                                if output_file_path in line:
                                    found_string_in_file_ang = True
                                    break                    
                        # if ( os.path.isfile(mlp_model_file_path) ):
                        # if ( found_string_in_file_mag and found_string_in_file_ang ):
                        if ( found_string_in_file_mag ):
                            if ( not use_complex ):
                                mlp_model_file_path_aux = os.path.join(mlp_models_sub_dirs_aux[index], directory_list['mlp_models']['file_name'])
                            output_sig_set = np.load(output_file_path)
                            dictionary_file_name =  dictionary_file_list[index]
                            dictionary = np.load(dictionary_file_name)
                            recovery_file_path = os.path.join(sub_dir, input_file_name)
                            if ( not os.path.isfile(recovery_file_path )):
                            # if ( os.path.isfile(recovery_file_path )):
                                # for idx, output_sig in enumerate(output_sig_set):
                                # start_time = time.perf_counter()
                                for idx in range(recovery_set_size):
                                    if ( use_complex ):
                                        recovered_signal = recover_signal(dictionary, output_sig_set[idx], system_params, mode, mlp_model_file_path)
                                    else:
                                        recovered_signal = recover_signal(dictionary, output_sig_set[idx], system_params, mode, mlp_model_file_path, mlp_model_file_path_aux)
                                    recovery_sig_set[idx] = recovered_signal
                                # end_time = time.perf_counter()
                                # ave_recovery_time = ( end_time - start_time ) / recovery_set_size
                                np.save(recovery_file_path, recovery_sig_set)
                                recovery_sig_set.fill(0)
                        #else:
                            #os.remove( recovery_file_path )
                    pass
    def create_nyfr_output(system_params, filter_params, LO_params):
        input_directory = "test_sets\\System_Config_1_Inputs\\"
        output_base_dir = "test_sets\\System_Config_1_Outputs\\"
        f_mod_list = [[0.1, "f_mod_0_1"], [0.2, "f_mod_0_2"], [0.25, "f_mod_0_25"], [0.5, "f_mod_0_5"]]
        f_delta_list = [[0.1, "f_delta_0_1"], [0.8, "f_delta_0_8"], [1.2, "f_delta_1_2"], [10, "f_delta_9_9"]]
        all_file_paths = get_all_file_paths(input_directory)
        time_file_path = "test_sets\\System_Config_1_Internal\\System\\Not_Sampled\\time.npy"
        encoded_test_list = []
        t = np.load(time_file_path)
        for input_set_file in all_file_paths:
            test_sig_set = np.load(input_set_file)
            input_path = Path(input_set_file)
            input_path_len = len(input_path.parts)
            input_file_name = input_path.parts[input_path_len - 1]
            input_phase_shift = input_path.parts[input_path_len - 2]
            input_noise_level = input_path.parts[input_path_len - 3]
            output_sub_path = output_base_dir + input_noise_level + "\\" + input_phase_shift + "\\"
            for f_mod in f_mod_list:
                LO_params['phase_freq'] = f_mod[0]
                for f_delta in f_delta_list:
                    LO_params['phase_delta'] = round(f_delta[0] * f_mod[0], 2)
                    output_file_dir = output_sub_path + f_mod[1] + "\\" + f_delta[1] + "\\"
                    output_file_path = (os.path.join(output_file_dir, input_file_name))
                    if ( os.path.isfile(output_file_path) ):
                        os.remove( output_file_path )
                    for test_data in test_sig_set:
                        LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t, LO_params, system_params)
                        y_mixed = np.copy(test_data*rising_zero_crossings)
                        y_filtered, filt_freq, filt_freq_down = filter_signal(y_mixed, t, filter_params, system_params)
                        y_sampled, LO_mod_sampled, t_sampled, tf_sampled, filt_sampled, downsample_train = downsample(y_filtered, LO_mod, t, system_params, rising_zero_crossings, filt_freq)
                        encoded_test_list.append( y_sampled )
                    encoded_test_set = np.array(encoded_test_list)
                    np.save(output_file_path, encoded_test_set)
                    encoded_test_list.clear()
    
    def create_test_set(dictionary, t, system_params, wave_params, filter_params, LO_params):
        enc_dim = (dictionary.shape)[0]
        signal_dim = (dictionary.shape)[1]
        wbf_cut_freq = system_params['adc_clock_freq'] * system_params['wbf_cut_mod']
        # num_of_pos_bins = int(signal_dim / 2)
        # num_of_sig = 2
        num_of_pos_sig = 1
        # for wave_param in wave_params:
        #     if wave_param['amp']!= 0:
        #         num_of_pos_sig += 1
        # num_of_tot_sig = int(num_of_pos_sig * 2)
        pos_bins = list(range(1, wbf_cut_freq))
        # sig_array = np.zeros((signal_dim,1,1))
        test_freq_tot_list = []
        min_num_of_pos_sig = 1

        # tot_num_freq_combos = 0
        for total_active_sig in range(min_num_of_pos_sig,num_of_pos_sig + 1):
            pass
            # test_freq_tot_list.append(list(combinations(pos_bins, total_active_sig)))
            test_freq_tot_list += list(combinations(pos_bins, total_active_sig))
            # tot_num_freq_combos += len(test_freq_tot_list[total_active_sig - 1])
        random.shuffle(test_freq_tot_list)

        split_list_len = 79800
        # test_freq_tot_list = test_freq_tot_list[0:split_list_len]
        # test_freq_tot_list = [(random.randint(1, wbf_cut_freq), random.randint(1, wbf_cut_freq), random.randint(1, wbf_cut_freq), random.randint(1, wbf_cut_freq)) for _ in range(split_list_len)]
        # test_freq_tot_list = [(random.randint(1, wbf_cut_freq), random.randint(1, wbf_cut_freq), random.randint(1, wbf_cut_freq), random.randint(1, wbf_cut_freq), random.randint(1, wbf_cut_freq)) for _ in range(split_list_len)]

        # tot_num_freq_combos = len(test_freq_tot_list)
        tot_num_freq_combos = split_list_len
        test_sig_set = np.zeros((tot_num_freq_combos,signal_dim), dtype='complex128')
        test_sig_set_time = np.zeros((tot_num_freq_combos,signal_dim), dtype='float64')
        count = 0
        # for test_freq_list in test_freq_tot_list:
        for test_freq in test_freq_tot_list:
            test_wave_params = []
            for freq in test_freq:
                period = 1 / freq
                # test_wave_param = {'amp': 0.1,
                test_wave_param = {'amp': random.uniform(0.5, 1),
                                'freq': freq,
                                # high shift: 0.4 - 0.75
                                # low shift: 0.05 - 0.25
                                # 'phase': random.uniform(0.05, 0.25) * period}
                                'phase': 0}
                # high noise: 2 - 3
                # low noise: 0.2 - 0.3
                system_params['noise'] = random.uniform(2, 3)
                # system_params['noise'] = 0
                # test_wave_param['amp'] = 0.1
                # test_wave_param['freq'] = freq
                test_wave_params.append(test_wave_param)
            x, t_test, num_tones = multi_tone_sine_wave(system_params, test_wave_params, filter_params)
            xf = fft(x)
            test_sig_set[count,:] = xf
            test_sig_set_time[count,:] = x
            test_wave_params.clear()
            count += 1
        # np.save("test_sets/System_Config_1_Inputs/time.npy", t_test)
        # np.save("test_sets/System_Config_1_Inputs/high_noise/no_phase_shift/5_tone_sigs.npy", test_sig_set_time)

        encoded_test_set = np.zeros((tot_num_freq_combos,enc_dim),dtype=np.complex128)
        # LO_mod_test_set = np.zeros((tot_num_freq_combos,signal_dim),dtype=np.float64)
        # dic_test_set = np.zeros((tot_num_freq_combos,signal_dim),dtype=np.complex128)
        # decoder_test_set = np.zeros((tot_num_freq_combos,signal_dim),dtype=np.complex128)
        # decoded_mag_model = tf.keras.models.load_model('models/decoder_mag_model_rnd_3_sig.keras')
        # decoded_ang_model = tf.keras.models.load_model('models/decoder_ang_model_rnd_3_sig.keras')
        pseudo_inv = np.linalg.pinv(dictionary)
        input_directory = "test_sets\\System_Config_1_Inputs\\"
        output_base_directory_name = "System_Config_1_Outputs"
        f_mod_list = [0.1, 0.2, 0.25, 0.5]
        f_delta_list = [0.1, 0.8, 1.2, 10]
        all_file_paths = get_all_file_paths(input_directory)
        # all_file_names = get_all_file_names(input_directory)
        
        # t_test_filename = all_file_names.pop(0)
        # t_test = np.load(all_file_paths.pop(0))
        for input_set_file in all_file_paths:
            test_sig_set = np.load(input_set_file)
            input_path_list = input_set_file.split("\\")
            output_file_name = input_path_list.pop()
            output_base_path_list = input_path_list.copy()
            output_base_path_list[1] = output_base_directory_name
            output_base_dir = os.path.join(*output_base_path_list)
            output_file_dirs = get_all_sub_dirs(output_base_dir)
            for num_mod, f_mod in enumerate(f_mod_list):
                LO_params['phase_freq'] = f_mod
                num_mod_adj = num_mod * len(f_delta_list)
                for num_delta, f_delta in enumerate(f_delta_list):
                    LO_params['phase_delta'] = round(f_delta * f_mod, 2)
                    output_file_dir = output_file_dirs[num_delta + num_mod_adj]
                    # LO_output_file_path = (os.path.join(output_file_dir, "LO_mod_" + output_file_name))
                    output_file_path = (os.path.join(output_file_dir, output_file_name))
                    for idx, test_data in enumerate(test_sig_set):
                        LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t_test, LO_params, system_params)
                        y_mixed = np.copy(test_data*rising_zero_crossings)
                        y_filtered, filt_freq, filt_freq_down = filter_signal(y_mixed, t, filter_params, system_params)
                        y_sampled, LO_mod_sampled, t_sampled, tf_sampled, filt_sampled, downsample_train = downsample(y_filtered, LO_mod, t, system_params, rising_zero_crossings, filt_freq)
                        encoded_test_set[idx,:] = y_sampled
                        # LO_mod_test_set[idx,:] = LO_mod
                    # print(input_set_file, "   ", output_file_path)
                    np.save(output_file_path, encoded_test_set)
                    # np.save(LO_output_file_path, LO_mod_test_set)
            # pass

        for input_set_file in all_file_paths:
            test_sig_set = np.load(input_set_file)
            for idx, test_data in enumerate(test_sig_set):
                LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t_test, LO_params, system_params)
                y_mixed = np.copy(ifft(test_data)*rising_zero_crossings)
                y_filtered, filt_freq, filt_freq_down = filter_signal(y_mixed, t, filter_params, system_params)
                y_sampled, LO_mod_sampled, t_sampled, tf_sampled, filt_sampled, downsample_train = downsample(y_filtered, LO_mod, t, system_params, rising_zero_crossings, filt_freq)
                encoded_test_set[idx,:] = y_sampled
            np.save("test_sets/System_Config_1_Outputs/high_noise/high_phase_shift/1_2_tone_sigs.npy", encoded_test_set)
            pass
        # test_sig_set = test_sig_set[np.random.permutation(tot_num_freq_combos),:]
        for idx, test_data in enumerate(test_sig_set):
            LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t, LO_params, system_params)
            y_mixed = np.copy(ifft(test_data)*rising_zero_crossings)
            y_filtered, filt_freq, filt_freq_down = filter_signal(y_mixed, t, filter_params, system_params)
            y_sampled, LO_mod_sampled, t_sampled, tf_sampled, filt_sampled, downsample_train = downsample(y_filtered, LO_mod, t, system_params, rising_zero_crossings, filt_freq)
            encoded_test_set[idx,:] = y_sampled
            y_sampled_mag = np.abs(y_sampled)
            y_sampled_ang = np.angle(y_sampled)
            y_sampled_mag = y_sampled_mag.reshape((1,y_sampled_mag.shape[0]))
            y_sampled_ang = y_sampled_ang.reshape((1,y_sampled_ang.shape[0]))
            coef_mag = np.transpose(decoded_mag_model.predict(y_sampled_mag))
            coef_ang = np.transpose(decoded_ang_model.predict(y_sampled_ang))
            dic_test_data = np.dot(pseudo_inv,y_sampled)
            decoder_test_data = coef_mag*(cos(coef_ang)+1j*sin(coef_ang))
            # dic_test_data = np.matmul(dictionary,test_data)
            dic_test_set[idx,:] = dic_test_data
            decoder_test_set[idx,:] = decoder_test_data.reshape((1,decoder_test_data.shape[0]))
            pass
        return test_sig_set, encoded_test_set, dic_test_set, decoder_test_set