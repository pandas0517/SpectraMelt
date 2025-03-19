'''
Created on Jul 11, 2024

@author: pete
'''
# class MyClass(object):
#     '''
#     classdocs
#     '''
#
#
#     def __init__(self, params):
#         '''
#         Constructor
#         '''
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft
# from sklearn.model_selection import train_test_split
import numpy as np
import random
from itertools import combinations
from signals import multi_tone_sine_wave
import keras
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras import regularizers
from keras import layers
from signals import generate_LO
from signals import filter_signal
from signals import downsample
from signals import create_nyfr_dict
from signals import recover_signal
import tensorflow as tf
from numpy import sin, cos
from pathlib import Path
import os
import time
from keras import losses

def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(losses.mean_squared_error(y_true, y_pred))

def reset_tensforflow_session():
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()

def get_all_file_names(directory):
    file_names = []
    for _, _, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    return file_names

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def get_all_sub_dirs(directory):
    deepest_sub_dirs = []
    for root, dirnames, _ in os.walk(directory):
        if not dirnames:
            deepest_sub_dirs.append(root)
    return deepest_sub_dirs

def create_dictionaries(system_params, LO_params, filter_params):
    input_directory = "test_sets\\System_Config_1_Inputs\\"
    # lo_directory = "test_sets\\System_Config_1_Internal\\LocalOscillator\\"
    dictionary_file_name = "dictionary.npy"
    system_directory = "test_sets\\System_Config_1_Internal\\System\\Not_Sampled\\"
    dictionary_directory = "test_sets\\System_Config_1_Internal\\Dictionary\\"
    f_mod_list = [[0.1, "f_mod_0_1"], [0.2, "f_mod_0_2"], [0.25, "f_mod_0_25"], [0.5, "f_mod_0_5"]]
    f_delta_list = [[0.1, "f_delta_0_1"], [0.8, "f_delta_0_8"], [1.2, "f_delta_1_2"], [10, "f_delta_9_9"]]
    recovery_dict = 'original'
    all_file_paths = get_all_file_paths(input_directory)
    # LO_file_paths = get_all_file_paths(lo_directory)
    # filt_freq_file_path = os.path.join(system_directory, "filt_freq.npy")  
    # filt_sampled_file_path = os.path.join(system_directory, "filt_sampled.npy") 
    t_test = np.load(os.path.join(system_directory, "time.npy"))
    test_sig_set = np.load(all_file_paths[0])
    test_data = test_sig_set[0]
    for f_mod in f_mod_list:
        LO_params['phase_freq'] = f_mod[0]
        mod_dir = f_mod[1]
        for f_delta in f_delta_list:
            LO_params['phase_delta'] = round(f_delta[0] * f_mod[0], 2)
            delta_dir = f_delta[1]
            # LO_mod_output_file_path = os.path.join(lo_directory, file_name_ext_list[num_delta + num_mod_adj] + "_LO_mod.npy")  
            # LO_mod_sampled_output_file_path = os.path.join(lo_directory, file_name_ext_list[num_delta + num_mod_adj] + "_LO_mod_sampled.npy")
            dictionary_base_path = dictionary_directory + recovery_dict + "\\" + mod_dir + "\\" + delta_dir + "\\"
            dictionary_file_path = os.path.join(dictionary_base_path, dictionary_file_name)
            LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t_test, LO_params, system_params)
            y_mixed = np.copy(test_data*rising_zero_crossings)
            y_filtered, filt_freq, filt_freq_down = filter_signal(y_mixed, t_test, filter_params, system_params)
            y_sampled, LO_mod_sampled, t_sampled, tf_sampled, filt_sampled, downsample_train = downsample(y_filtered, LO_mod, t_test, system_params, rising_zero_crossings, filt_freq)
            dictionary = create_nyfr_dict(t_test, LO_mod, LO_mod_sampled, filt_sampled, system_params, tf_sampled)
            # np.save(filt_freq_file_path, filt_freq)
            # np.save(filt_sampled_file_path, filt_sampled)
            # np.save(LO_mod_output_file_path, LO_mod)
            # np.save(LO_mod_sampled_output_file_path, LO_mod_sampled)
            np.save(dictionary_file_path, dictionary)

def analyze_dfs(system_params):
    recovery_dict = 'original'
    # recovery_dict = 'enhanced'
    recovery_info = {
        'input_dir': "test_sets\\System_Config_1_Inputs\\",
        'enhanced': {
            'dict_base': "test_sets\\System_Config_1_Internal\\Dictionary\\enhanced\\",
            'rec_base': {
                'c_omp': 'test_sets\\System_Config_1_Enhanced_Recovery\\System_Config_1_OMP_Custom_Recovery\\',
                'o_omp': 'test_sets\\System_Config_1_Enhanced_Recovery\\System_Config_1_OMP_Recovery\\',
                'mlp1': 'test_sets\\System_Config_1_Enhanced_Recovery\\System_Config_1_MLP1_Recovery\\',
                'spgl1': 'test_sets\\System_Config_1_Enhanced_Recovery\\System_Config_1_SPGL_Recovery\\'
                }
            },        
        'original': {
            'dict_base': "test_sets\\System_Config_1_Internal\\Dictionary\\original\\",
            'rec_base': {
                'c_omp': 'test_sets\\System_Config_1_Original_Recovery\\System_Config_1_OMP_Custom_Recovery\\',
                'o_omp': 'test_sets\\System_Config_1_Original_Recovery\\System_Config_1_OMP_Recovery\\',
                'mlp1': 'test_sets\\System_Config_1_Original_Recovery\\System_Config_1_MLP1_Recovery\\',
                'spgl1': 'test_sets\\System_Config_1_Original_Recovery\\System_Config_1_SPGL_Recovery\\'
                }
            },
    }
    # input_df_filename = "test_sets\\input_df.pkl"
    # input_df = pd.read_pickle(input_df_filename)
    # del input_df['num_tones']
    # del input_df['tone_frequencies']
    # tone_freqs = np.zeros((input_df.shape[0],2), dtype='int')
    # tone_freq = tone_freqs.tolist()
    recovery_df_filename = "test_sets\\recovery_df.pkl"
    recovery_df = pd.read_pickle(recovery_df_filename)
    # rec_sub_df = recovery_df[ recovery_df['recovery_method'] == "c_omp" ]
    # rec_sub_df['recovery_method'] = "mlp1"
    # new_df = pd.concat([recovery_df, rec_sub_df], axis=0, ignore_index=True)
    # new_df.to_pickle(recovery_df_filename)
    add_columns = 0
    recovery_sig_set_size = 0
    input_tone_thresh = 600
    recovery_mag_thresh = 2
    recovery_base_path = recovery_info[recovery_dict]['rec_base'][system_params['recovery']]
    input_file_paths = get_all_file_paths(recovery_info['input_dir'])
    for input_set_file in input_file_paths:
        input_path = Path(input_set_file)
        input_path_len = len(input_path.parts)
        input_file_name = input_path.parts[input_path_len - 1]
        input_phase_shift = input_path.parts[input_path_len - 2]
        input_noise_level = input_path.parts[input_path_len - 3]
        recovery_sub_path = recovery_base_path + input_noise_level + "\\" + input_phase_shift + "\\"
        input_sig_set = np.load(input_set_file)
        recovery_file_sub_dirs = get_all_sub_dirs(recovery_sub_path)
        for sub_dir in recovery_file_sub_dirs:
            recovery_file_path = os.path.join(sub_dir, input_file_name)
            recovery_sig_set = np.load(recovery_file_path)
            if ( add_columns == 1 ):
                recovery_sig_set_size = recovery_sig_set.shape[0]
                add_columns = 0
                for stats in range(recovery_sig_set_size):
                    # num_rec_freq = "num_rec_freq_" + str(stats)
                    # num_spur_freq = "num_spur_freq_" + str(stats)
                    # ave_rec_mag_err = "ave_rec_mag_err_" + str(stats)
                    # recovery_df[num_rec_freq] = 0
                    # recovery_df[num_spur_freq] = 0
                    # recovery_df[ave_rec_mag_err] = 0.0
                    # total_input_tones = "total_input_tones_" + str(stats)
                    # recovery_df[total_input_tones] = 0
                    # rec_tone_thresh_ = "rec_tone_thresh_" + str(stats)
                    # recovery_df[rec_tone_thresh_] = 0.0
                    # ave_rec_mag = "ave_rec_mag_" + str(stats)
                    # recovery_df[ave_rec_mag] = 0.0
                    # max_rec_mag = "max_rec_mag_" + str(stats)
                    # recovery_df[max_rec_mag] = 0.0
                    # ave_spur_mag = "ave_spur_mag_" + str(stats)
                    # recovery_df[ave_spur_mag] = 0.0
                    # max_spur_mag = "max_spur_mag_" + str(stats)
                    # recovery_df[max_spur_mag] = 0.0
                    min_spur_mag = "min_spur_mag_" + str(stats)
                    recovery_df[min_spur_mag] = 0.0
                    min_rec_mag = "min_rec_mag_" + str(stats)
                    recovery_df[min_rec_mag] = 0.0
                    pass
                recovery_df.to_pickle(recovery_df_filename)
                
            recovery_path = Path(recovery_file_path)

            match recovery_path.parts[5]:
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
            match recovery_path.parts[6]:
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
            pass
            current_recovery_row = recovery_df.index[(recovery_df['file_name']==input_file_name) &
                           (recovery_df['noise_level']==input_noise_level) &
                           (recovery_df['phase_shift']==input_phase_shift) &
                           (recovery_df['f_mod']==f_mod) &
                           (recovery_df['f_delta']==f_delta) &
                           (recovery_df['dictionary_type']==recovery_dict) &
                           (recovery_df['recovery_method']==system_params['recovery'])]
            
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
                # rec_tone_mag = abs(rec_sig[rec_sig_tones])
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
            pass
    recovery_df.to_pickle(recovery_df_filename)
    #     for idx, input_sig in enumerate(input_sig_set):
    #         signal_number = "tones_signal_" + str(idx)
    #         input_sig_xf = fft(input_sig)
    #         input_sig_xf_shifted = np.fft.fftshift(abs(input_sig_xf))
    #         indices = np.where(input_sig_xf_shifted>600)
    #         indices_shifted = indices[0] - (input_sig.size/2)
    #         tone_frequencies = (((indices_shifted[indices_shifted>0])/4).astype('int')).tolist()
    #         if (index == 0):
    #             input_df[signal_number] = tone_freq
    #             pass
    #         input_df.at[current_input_df[0], signal_number] = tone_frequencies
    #         pass
    #     pass
    # input_df.to_pickle(input_df_filename)

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
        # new_freq_list = list(combinations(pos_bins, total_active_sig))
        # test_freq_list = test_freq_list + new_freq_list
        # new_freq_list.clear()
    # x, t, num_tones = multi_tone_sine_wave(system_params, wave_params, filter_params)
    # sig_array = np.zeros((1,signal_dim))
    # for sig in range(1,num_of_pos_sig + 1):
    #     comb_pos_sig= np.array(list(combinations(bins, sig)))
    #     comb_neg_sig = np.copy(-1*comb_pos_sig)
    #     comb_sig = np.stack((comb_neg_sig, comb_pos_sig), axis=2)
    #     comb_sig_shift = np.copy(num_of_pos_bins + comb_sig)
    #     # sig_array_comb_n = np.zeros((signal_dim,1,comb_sig.shape[0]))
    #     sig_array_comb_n = np.zeros((comb_sig.shape[0],signal_dim))
    #     for n, comb in enumerate(comb_sig_shift):
    #         # sig_array_comb_n[comb, 0, n] = 1
    #         # sig_array_comb_n[n, comb] = 1
    #         sig_array_comb_n[n, comb] = num_of_pos_bins
    #     sig_array = np.concatenate((sig_array, sig_array_comb_n), axis=0)
    #     if (sig == 1):
    #         sig_array = np.delete(sig_array,(0), axis=0)
    # pass
    # # sig_array = np.delete(sig_array,0,2)
    # return sig_array
def create_mlp1_models(system_params):
    # modes = [ 'imag', 'mag', 'ang', 'complex']
    # modes = [ 'real', 'imag', 'mag', 'ang', 'complex']
    # modes = [ 'real']
    # modes = [ 'mag', 'ang' ]
    time_file_path = "test_sets\\System_Config_1_Internal\\System\\Not_Sampled\\time.npy"
    t_test = np.load(time_file_path)
    K_band = round(t_test.size*(system_params['spacing']*system_params['adc_clock_freq']))
    Zones = int(t_test.size/K_band)
    del t_test
    modes = [ 'mag' ]
    pre_omp = False
    total_num_sigs = 40000
    dictionary_file_name = "dictionary.npy"
    mlp_model_file_name = "mlp_model_file.keras"
    # mlp_model_file_name = "input_list.txt"
    # model_log_file_name = "input_list_Bedroom_PC.txt"
    train_test_split_percentage = 0.7
    learning_rate = 0.00001
    num_epochs = 200
    batch_sz = 128
    recovery_dic_type = 'original'
    # processing_systems = 'daddo'
    processing_systems = 'bedroom'
    # mlp_model_file_name = "input_list.txt"
    model_log_file_name = {
        'bedroom':"input_list_Bedroom_PC.txt",
        'daddo':"input_list_Daddo_PC.txt"
    }
    recovery_log_file_name = {
        'mag':{
            'rec_mode': 'mag_ang',
            'bedroom': "recovery_list_bedroom_mag.txt",
            'daddo': "recovery_list_daddo_mag.txt"
        },
        'ang':{
            'rec_mode': 'mag_ang',
            'bedroom': "recovery_list_bedroom_ang.txt",
            'daddo': "recovery_list_daddo_ang.txt"
        },
    }
    directory_list = {
        'input': "test_sets\\System_Config_1_Inputs\\",
        'fft': "test_sets\\System_Config_1_Model_Inputs\\FFT\\",
        'train': "test_sets\\System_Config_1_Model_Inputs\\train\\",
        'test': "test_sets\\System_Config_1_Model_Inputs\\test\\",
        'output': "test_sets\\System_Config_1_Outputs\\",
        'mlp_models': {
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
        'premultiply': {
            'enhanced': "Y:\\School_Stuff\\System_Config_1_PreMultiply\\enhanced\\",
            'original': "Y:\\School_Stuff\\System_Config_1_PreMultiply\\original\\"
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

    dictionary_file_base_dir = directory_list['dictionary'][recovery_dic_type]
    dictionary_file_sub_dirs = get_all_sub_dirs(dictionary_file_base_dir)
    input_file_paths = get_all_file_paths(directory_list['input'])
    use_fft_file = False
    save_fft_file = True
    use_premultiply = False
    save_premultiply = True

    for mode in modes:
        # train_input_file_paths = get_all_file_paths(directory_list['train'])
        # test_input_file_paths = get_all_file_paths(directory_list['test'])

        for id, input_set_file in enumerate(input_file_paths):
            input_path = Path(input_set_file)
            input_path_len = len(input_path.parts)
            input_file_name = input_path.parts[input_path_len - 1]
            input_phase_shift = input_path.parts[input_path_len - 2]
            input_noise_level = input_path.parts[input_path_len - 3]
            fft_file_sub_dir = directory_list['fft'] + input_noise_level + "\\" + input_phase_shift + "\\"
            fft_file_path = os.path.join(fft_file_sub_dir, input_file_name)
            
            if ( os.path.isfile(fft_file_path) ):
                input_sig_set = np.load(fft_file_path)
                use_fft_file = True
            else:
            # for i, train_input_file in enumerate(train_input_file_paths):
                use_fft_file = False
                input_sig_set_total = np.load(input_set_file)
                num_input_sigs_total = input_sig_set_total.shape[0]
                input_sigs_not_used = num_input_sigs_total - total_num_sigs
                # input_sig_set_split = np.vsplit(input_sig_set_total, 2)
                input_sig_set_split = np.vsplit(input_sig_set_total, [input_sigs_not_used])
                input_sig_set = np.copy(input_sig_set_split[1])
                del input_sig_set_split
                del input_sig_set_total
            # test_size = input_sig_set_test.shape[0]
            # train_size = input_sig_set_train.shape[0]
            fft_input_sig_set = []
            complex_input_sig_set = []

            for i, input_sig in enumerate(input_sig_set):
                if ( use_fft_file ):
                    input_sig_fft = input_sig
                else:
                    input_sig_fft = fft(input_sig)
                    if ( save_fft_file ):
                        fft_input_sig_set.append(input_sig_fft)
                # input_sig_fft_real = input_sig_fft.real
                # input_sig_fft_imag = input_sig_fft.imag
                if (mode == 'real'):
                    input_sig_set[i] = np.copy(input_sig_fft.real)
                elif (mode == 'imag'):
                    input_sig_set[i] = np.copy(input_sig_fft.imag)
                elif (mode == 'mag'):
                    input_sig_set[i] = np.abs(input_sig_fft)
                elif (mode == 'ang'):
                    input_sig_set[i] = np.angle(input_sig_fft)
                elif (mode == 'complex'):
                    input_sig_concat = np.concatenate((input_sig_fft.real, input_sig_fft.imag))
                    complex_input_sig_set.append(input_sig_concat)

            if ( not use_fft_file and save_fft_file ):
                fft_input_set = np.array(fft_input_sig_set)
                if ( not os.path.exists(fft_file_path) ):
                    np.save(fft_file_path, fft_input_set)
                del fft_input_set
                del fft_input_sig_set

            if ( mode == 'complex'):
                input_sig_set = np.array(complex_input_sig_set)
                del complex_input_sig_set
            elif ( use_fft_file ):
                input_sig_set = input_sig_set.astype(np.float64)

            model_input_size = input_sig_set.shape[1]
            num_input_sigs = input_sig_set.shape[0]
            train_size = int(num_input_sigs * train_test_split_percentage)
            test_size = num_input_sigs - train_size

            input_sig_set_train_test = np.vsplit(input_sig_set, [train_size])
            del input_sig_set
        
            output_sub_path = directory_list['output'] + input_noise_level + "\\" + input_phase_shift + "\\"
            mlp_model_sub_path = directory_list['mlp_models'][recovery_dic_type][mode] + input_noise_level + "\\" + input_phase_shift + "\\"
            model_log_file_path = directory_list['mlp_models'][recovery_dic_type][mode] + model_log_file_name[processing_systems]
            premultiply_sub_path = directory_list['premultiply'][recovery_dic_type] + input_noise_level + "\\" + input_phase_shift + "\\"
            recovery_log_file_path = directory_list['recovery'][recovery_dic_type][system_params['recovery']] \
                    + recovery_log_file_name[mode]['rec_mode'] + "\\" + recovery_log_file_name[mode][processing_systems]
            output_file_sub_dirs = get_all_sub_dirs(output_sub_path)
            premultiply_file_sub_dirs = get_all_sub_dirs(premultiply_sub_path)
            mlp_model_file_sub_dirs = get_all_sub_dirs(mlp_model_sub_path)
            # for index, sub_dir in enumerate(premultiply_file_sub_dirs):
            for index, sub_dir in enumerate(output_file_sub_dirs):
                output_file_path = os.path.join(sub_dir, input_file_name)
                found_string_in_file = False
                premultiply_file_path = os.path.join(premultiply_file_sub_dirs[index], input_file_name)
                with open(model_log_file_path, "r") as model_log:
                    for line in model_log:
                        # if premultiply_file_path in line:
                        if output_file_path in line:
                            found_string_in_file = True
                            break

                if not found_string_in_file:  
                    #del model_log
                    # del line
                    if ( os.path.isfile(premultiply_file_path) ):
                        output_sig_set = np.load(premultiply_file_path)

                        use_premultiply = True
                    else:
                        use_premultiply = False
                        dictionary_file_path = dictionary_file_sub_dirs[index] + "\\" + dictionary_file_name
                        dictionary = np.load(dictionary_file_path)
                        output_sig_set_total = np.load(output_file_path)
                        num_input_sigs_total = output_sig_set_total.shape[0]
                        output_sigs_not_used = num_input_sigs_total - total_num_sigs
                        output_sig_set_split = np.vsplit(output_sig_set_total, [output_sigs_not_used])
                        output_sig_set = output_sig_set_split[1]
                        del output_sig_set_total
                        del output_sig_set_split

                    premultiply_sig_set = []
                    premultiply_sig_set_train = np.zeros((train_size, model_input_size))
                    premultiply_sig_set_test = np.zeros((test_size, model_input_size))
                    complex_premultiply_sig_set_train = []
                    complex_premultiply_sig_set_test = []
                    # start_time = time.perf_counter()
                    for ij, output_signal in enumerate(output_sig_set):
                        if ( use_premultiply ):
                            premultiply_signal = np.copy(output_signal)
                        else:
                            if ( pre_omp ):
                                original_recovery = system_params['recovery']
                                system_params['recovery'] = 'c_omp'
                                premultiply_signal = recover_signal(0.2*dictionary, output_signal, system_params, mode)
                                system_params['recovery'] = original_recovery
                            else:
                                pseudo = np.linalg.pinv( 0.01 * dictionary)
                                premultiply_signal = np.dot(pseudo,output_signal)

                            premultiply_sig_set.append(premultiply_signal)

                        # premultiply_signal_real = premultiply_signal.real
                        # premultiply_signal_imag = premultiply_signal.imag

                        if ( ij < train_size ):
                            if ( mode == 'real'):
                                premultiply_sig_set_train[ij] = np.copy(premultiply_signal.real)
                            elif ( mode == 'imag'):
                                premultiply_sig_set_train[ij] = np.copy(premultiply_signal.imag)
                            elif ( mode == 'mag'):
                                premultiply_sig_set_train[ij] = np.abs(premultiply_signal)
                            elif ( mode == 'ang'):
                                premultiply_sig_set_train[ij] = np.angle(premultiply_signal)
                            elif ( mode == 'complex'):
                                premultiply_sig_concat = np.concatenate((premultiply_signal.real, premultiply_signal.imag))
                                complex_premultiply_sig_set_train.append(premultiply_sig_concat)
                        else:
                            if ( mode == 'real'):
                                premultiply_sig_set_test[ij - train_size] = np.copy(premultiply_signal.real)
                            elif ( mode == 'imag'):
                                premultiply_sig_set_test[ij - train_size] = np.copy(premultiply_signal.imag)
                            elif ( mode == 'mag'):
                                premultiply_sig_set_test[ij - train_size] = np.abs(premultiply_signal)
                            elif ( mode == 'ang'):
                                premultiply_sig_set_test[ij - train_size] = np.angle(premultiply_signal)
                            elif ( mode == 'complex'):
                                premultiply_sig_concat = np.concatenate((premultiply_signal.real, premultiply_signal.imag))
                                complex_premultiply_sig_set_test.append(premultiply_sig_concat)
                        # premultiply_sig_flattened[ij] = np.concatenate((premultiply_signal_real, premultiply_signal_imag))
                        pass

                    if ( not use_premultiply and save_premultiply ):
                        premultiply_sig_set_array = np.array(premultiply_sig_set)
                        np.save(premultiply_file_path, premultiply_sig_set_array)
                        del premultiply_sig_set_array
                        del premultiply_sig_set

                    if ( mode == 'complex'):
                        premultiply_sig_set_train = np.array(complex_premultiply_sig_set_train)
                        premultiply_sig_set_test = np.array(complex_premultiply_sig_set_test)
                        del complex_premultiply_sig_set_train
                        del complex_premultiply_sig_set_test

                    # input_file_name_without_extension = os.path.splitext(input_file_name)[0]
                    # mlp_model_file_path_ind = os.path.join(mlp_model_file_sub_dirs[index], input_file_name_without_extension + ".keras" )
                    mlp_model_file_path = os.path.join(mlp_model_file_sub_dirs[index], mlp_model_file_name)
                    if ( os.path.isfile( mlp_model_file_path )):
                        mlp_model = tf.keras.models.load_model(mlp_model_file_path)
                    else:
                        # if ( mode == 'ang' ):
                        #     loss_type = 'mean_squared_error'
                        # else:
                            # loss_type = 'mean_squared_error'
                            # loss_type = 'mean_absolute_error'
                        # loss_type = 'mean_squared_logarithmic_error'
                        # loss_type = 'Huber'
                        # loss_type = 'mean_absolute_error'
                        # loss_type = 'LogCosh'
                        loss_type = 'root_mean_squared_error'
                        mlp_model = keras.Sequential()
                        mlp_model.add(keras.Input(shape=(model_input_size,)))
                        mlp_model.add(layers.Reshape((Zones, K_band), input_shape=(model_input_size,)))
                        mlp_model.add(layers.Conv1D(filters=K_band,
                                                    kernel_size=10,
                                                    padding='same',
                                                    input_shape=(Zones,K_band),
                                                    # activity_regularizer=regularizers.l1(0.001),
                                                    name="mlp_model_layer_1"))
                        mlp_model.add(layers.Flatten())
                        # mlp_model.add(layers.Dense(4*model_input_size, name="mlp_model_layer_2"))
                        # mlp_model.add(layers.Dense(model_input_size, activity_regularizer=regularizers.l1(0.01), name="mlp_model_layer_2"))
                        # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_3"))
                        # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_4"))
                        mlp_model.add(layers.Dense(model_input_size,
                                                   activation='linear',
                                                #    activity_regularizer=regularizers.l2(0.001),
                                                   name="mlp_model_out"))
                        # mlp_model.add(layers.Activation('relu'))
                        mlp_opt = keras.optimizers.Adam(learning_rate=learning_rate)
                        # mlp_model.compile(optimizer=mlp_opt, loss=loss_type)
                        mlp_model.compile(optimizer=mlp_opt, loss=root_mean_squared_error)

                    early_stopping = EarlyStopping(monitor='val_loss',
                                                   min_delta=0.1,
                                                   patience=4,
                                                   verbose=1,
                                                   start_from_epoch=5,
                                                   restore_best_weights=True)
                    mlp_model.fit(premultiply_sig_set_train, input_sig_set_train_test[0],
                                    epochs=num_epochs,
                                    batch_size=batch_sz,
                                    shuffle=True,
                                    validation_data=(premultiply_sig_set_test, input_sig_set_train_test[1]),
                                    callbacks=[early_stopping])
                    mlp_model.save(mlp_model_file_path, overwrite=True)
                    # mlp_model.save(mlp_model_file_path_ind, overwrite=True)

                    with open(model_log_file_path, "a") as model_log:
                        model_log.write(output_file_path + "\n")
                    with open(recovery_log_file_path, "a") as recovery_log:
                        recovery_log.write(output_file_path + "\n")
                    reset_tensforflow_session()
            # input_sig_set_train = np.load(train_input_file)
            # input_sig_set_test = np.load(test_input_file_paths[i])
            # input_sig_set_combined = np.vstack((input_sig_set_test, input_sig_set_train))
            # input_sig_set_split = np.vsplit(input_sig_set_combined, 2)
            # input_sig_set =input_sig_set_split[1]
            # del input_sig_set_combined
            # del input_sig_set_split
            # del input_sig_set_train
            # del input_sig_set_test
            # input_sig_set_train = input_sig_set_train_test[0]

            # input_sig_set_test_original = input_sig_set_train_test[1]
            # del input_sig_set_train_test

            # input_file_name = input_path.parts[5]
            # input_noise_level = input_path.parts[3]
            # input_phase_shift = input_path.parts[4]
            # train_file_name = directory_list['train'] + input_noise_level + "\\" + input_phase_shift + "\\" + input_file_name
            # test_file_name = directory_list['test'] + input_noise_level + "\\" + input_phase_shift + "\\" + input_file_name       
            # input_sig_fft_flattened = np.zeros((num_input_sigs, model_input_size))
            # input_sig_set_train = np.zeros((train_size, model_input_size))
            # input_sig_set_test = np.zeros((test_size, model_input_size))
            # for i, input_signal in enumerate(input_sig_set):
            #     input_signal_fft = fft(input_signal)
            #     input_signal_fft_real = input_signal_fft.real
            #     input_signal_fft_imag = input_signal_fft.imag
            #     if ( i < test_size ):
            #         input_sig_set_test[i] = np.concatenate((input_signal_fft_real, input_signal_fft_imag))
            #     else:
            #         input_sig_set_train[i - test_size] = np.concatenate((input_signal_fft_real, input_signal_fft_imag))
            #     # input_sig_fft_flattened[i] = np.concatenate((input_signal_fft_real, input_signal_fft_imag))
            #     pass
            # # del input_sig_set
            # np.save(train_file_name, input_sig_set_train)
            # np.save(test_file_name, input_sig_set_test)
                    # premultiply_sig_set_total = np.load(premultiply_file_path)
                    # premultiply_sig_set_split = np.vsplit(premultiply_sig_set_total, 2)
                    #del premultiply_sig_set
                    # premultiply_sig_set = premultiply_sig_set_split[1]
                    # del premultiply_sig_set_total
                    # del premultiply_sig_set_split
                    # premultiply_sig_flattened = np.zeros((num_input_sigs, model_input_size))
                    # premultiply_sig_set_train = np.zeros((train_size, model_input_size))
                    # premultiply_sig_set_test = np.zeros((test_size, model_input_size))
                    # end_time = time.perf_counter()
                    # preMultiply_time = end_time - start_time
                    # ave_recovery_time = ( end_time - start_time ) / num_input_sigs
                    
                    # print(preMultiply_time)
                    
                    # premultiply_sig_set_train_real = np.zeros((train_size, model_input_size_half))
                    # premultiply_sig_set_test_real = np.zeros((test_size, model_input_size_half))
                    # premultiply_sig_set_train_imag = np.zeros((train_size, model_input_size_half))
                    # premultiply_sig_set_test_imag = np.zeros((test_size, model_input_size_half))
                    # premultiply_sig_set_train_mag = np.zeros((train_size, model_input_size_half))
                    # premultiply_sig_set_test_mag = np.zeros((test_size, model_input_size_half))
                    # premultiply_sig_set_train_ang = np.zeros((train_size, model_input_size_half))
                    # premultiply_sig_set_test_ang = np.zeros((test_size, model_input_size_half))
                    # for ij, premultiply_signal in enumerate(premultiply_sig_set):
                    #     premultiply_signal_real = premultiply_signal.real
                    #     premultiply_signal_imag = premultiply_signal.imag
                    #     premultiply_signal_mag = np.abs(premultiply_signal)
                    #     premultiply_signal_ang = np.angle(premultiply_signal)
                    #     if ( ij < test_size ):
                    #         premultiply_sig_set_test_real[ij] = np.concatenate((premultiply_signal_real, premultiply_signal_imag))
                    #         premultiply_sig_set_test_imag[ij] = np.concatenate((premultiply_signal_imag, premultiply_signal_imag))
                    #         premultiply_sig_set_test_mag[ij] = np.concatenate((premultiply_signal_mag, premultiply_signal_imag))
                    #         premultiply_sig_set_test_ang[ij] = np.concatenate((premultiply_signal_ang, premultiply_signal_imag))
                    #     else:
                    #         premultiply_sig_set_train_real[ij - test_size] = np.concatenate((premultiply_signal_real, premultiply_signal_imag))
                    #         premultiply_sig_set_train_imag[ij - test_size] = np.concatenate((premultiply_signal_imag, premultiply_signal_imag))
                    #         premultiply_sig_set_train_mag[ij - test_size] = np.concatenate((premultiply_signal_mag, premultiply_signal_imag))
                    #         premultiply_sig_set_train_ang[ij - test_size] = np.concatenate((premultiply_signal_ang, premultiply_signal_imag))
                        # premultiply_sig_flattened[ij] = np.concatenate((premultiply_signal_real, premultiply_signal_imag))
                        # pass

                    #del premultiply_sig_set
                    #del premultiply_signal_real
                    #del premultiply_signal_imag
                    # input_sig_set_train, input_sig_set_test, premultiply_sig_set_train, premultiply_sig_set_test = train_test_split(input_sig_fft_flattened, 
                    #                                                                                                                 premultiply_sig_flattened, 
                    #                                                                                                                 test_size=train_test_split_percentage)
                    # del input_sig_fft_flattened
                    # del premultiply_sig_flattened


                    # output_sig_set = np.load(output_file_path)
                    # dictionary_file_name =  dictionary_file_list[index]
                    # dictionary = np.load(dictionary_file_name)
                    # premultiply_sig_set = np.zeros((output_sig_set.shape[0], dictionary.shape[1]))
                    # pseudo_inv = np.linalg.pinv(dictionary)

                    # if ( os.path.isfile( mlp_model_file_path )):
                    #     mlp_model = tf.keras.models.load_model(mlp_model_file_path)
                    # else:
                    #     mlp_model = keras.Sequential()
                    #     mlp_model.add(keras.Input(shape=(model_input_size,)))
                    #     mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_1"))
                    #     # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_2"))
                    #     mlp_model.add(layers.Dense(model_input_size, name="mlp_model_out"))
                    #     mlp_opt = keras.optimizers.Adam(learning_rate=learning_rate)
                    #     mlp_model.compile(optimizer=mlp_opt, loss='mean_absolute_error')

                    #del mlp_model
                    #del premultiply_sig_set_test
                    #del premultiply_sig_set_train

                    #time.sleep(15)
                    # tf.keras.backend.clear_session()
                    # if ( not os.path.isfile( premultiply_file_path )):
                    # if ( os.path.isfile(recovery_file_path )):
                        # for idx, output_sig in enumerate(output_sig_set):
                        # start_time = time.perf_counter()
                        # for idx, output_sig in enumerate(output_sig_set):
                            # premultiply_sig_set[idx] = np.dot(pseudo_inv,output_sig)
                        # end_time = time.perf_counter()
                        # ave_recovery_time = ( end_time - start_time ) / idx
                        # np.save(premultiply_file_path, premultiply_sig_set)
                        # premultiply_sig_set.fill(0)
                    #else:
                        #os.remove( recovery_file_path )
    pass

def create_decoder(dictionary, test_set, dic_test_set, encoded_test_set):
    # This is the size of our encoded representations
    enc_dim = (dictionary.shape)[0]
    dec_dim = (dictionary.shape)[1]
    test_set_size = (test_set.shape)[0]
    train_split = int(0.5 * test_set_size)

    decoded_mag_model = keras.Sequential()
    decoded_mag_model.add(keras.Input(shape=(enc_dim,)))
    decoded_mag_model.add(layers.Dense(int(dec_dim/4), name="decoded_mag_model_layer1"))
    decoded_mag_model.add(layers.Dense(int(dec_dim/2), name="decoded_mag_model_layer2"))
    decoded_mag_model.add(layers.Dense(int(3*dec_dim/4), name="decoded_mag_model_layer3"))
    decoded_mag_model.add(layers.Dense(dec_dim, name="decoded_mag_model_out"))
    # decoded_mag_model.add(keras.Input(shape=(dec_dim,)))
    # decoded_mag_model.add(layers.Dense(int(dec_dim), name="decoded_mag_model_layer1"))
    # decoded_mag_model.add(layers.Dense(dec_dim, name="decoded_mag_model_out"))
    decoded_mag_model.summary()

    decoded_ang_model = keras.Sequential()
    decoded_ang_model.add(keras.Input(shape=(enc_dim,)))
    decoded_ang_model.add(layers.Dense(int(dec_dim/4), name="decoded_ang_model_layer1"))
    decoded_ang_model.add(layers.Dense(int(dec_dim/2), name="decoded_ang_model_layer2"))
    decoded_ang_model.add(layers.Dense(int(3*dec_dim/4), name="decoded_ang_model_layer3"))
    decoded_ang_model.add(layers.Dense(int(dec_dim), name="decoded_ang_model_out"))
    # decoded_ang_model.add(keras.Input(shape=(dec_dim,)))
    # decoded_ang_model.add(layers.Dense(int(dec_dim), name="decoded_ang_model_layer1"))
    # decoded_ang_model.add(layers.Dense(int(dec_dim), name="decoded_ang_model_out"))
    decoded_ang_model.summary() 

    mag_opt = keras.optimizers.Adam(learning_rate=0.001)
    ang_opt = keras.optimizers.Adam(learning_rate=0.001)
    decoded_mag_model.compile(optimizer=mag_opt, loss='mean_absolute_error')
    decoded_ang_model.compile(optimizer=ang_opt, loss='mean_absolute_error')
    # decoder_mag.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # decoder_ang.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # decoder.fit(np.real(encoded_test_set[:,:,:train_split]), np.real(test_set[:,:,:train_split]),
    # decoded_mag_model.fit(np.abs(encoded_test_set[:train_split,:]), np.abs(test_set[:train_split,:]),
    # decoded_mag_model.fit(np.abs(dic_test_set), np.abs(test_set),
    decoded_mag_model.fit(np.abs(encoded_test_set), np.abs(test_set),
                epochs=10,
                batch_size=64,
                shuffle=True,
                validation_data=(np.abs(encoded_test_set), np.abs(test_set)))
    # decoded_mag_model.fit(np.abs(dic_test_set), np.abs(test_set),
    #             epochs=20,
    #             batch_size=256,
    #             shuffle=True,
                # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
                # validation_data=(np.abs(encoded_test_set[train_split:,:]), np.abs(test_set[train_split:,:])))
                # validation_data=(np.abs(dic_test_set), np.abs(test_set)))
    decoded_ang_model.fit(np.angle(encoded_test_set), np.angle(test_set),
    # decoded_ang_model.fit(np.angle(dic_test_set), np.angle(test_set),
                epochs=10,
                batch_size=64,
                shuffle=True,
                validation_data=(np.angle(encoded_test_set), np.angle(test_set)))
    # decoder_ang.fit(np.angle(dic_test_set[:train_split,:]), np.angle(test_set[:train_split,:]),
    #             epochs=20,
    #             batch_size=256,
    #             shuffle=True,
                # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
                # validation_data=(np.angle(dic_test_set[train_split:,:]), np.angle(test_set[train_split:,:])))
    # decoded_mag_model.save("refine_mag_model.keras", overwrite=True)
    decoded_mag_model.save("decoder_mag_model_l1.keras", overwrite=True)
    # decoded_ang_model.save("refine_ang_model.keras", overwrite=True)
    decoded_ang_model.save("decoder_ang_model_l1.keras", overwrite=True)
    pass
    # # encoded_test_set = np.zeros((enc_dim,1,test_set_size),dtype=np.complex128)
    # encoded_test_set = np.zeros((test_set_size,enc_dim),dtype=np.complex128)
    # # test_set = test_set[:,:,np.random.permutation(test_set_size)]
    # test_set = test_set[np.random.permutation(test_set_size),:]
    # for idx, test_data in enumerate(test_set):
    #     # encoded_test_data = np.matmul(dictionary,test_data.T)
    #     LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t, LO_params, system_params)
    #     encoded_test_data = np.matmul(dictionary,test_data)
    #     # encoded_test_set[:,:,idx] = encoded_test_data
    #     encoded_test_set[idx,:] = encoded_test_data
    #     pass

    # encoded_complex_input_layer = keras.Input(shape=(enc_dim,))
    # encoded_real_input_layer = keras.Input(shape=(enc_dim,))
    # encoded_imag_input_layer = keras.Input(shape=(enc_dim,))
    # decoded_complex = layers.Dense(dec_dim, activation='sigmoid')(encoded_real_input_layer)
    # decoded_imag = layers.Dense(dec_dim, activation='sigmoid')(encoded_imag_input_layer)
    # decoded_real = layers.Dense(dec_dim, activation='relu')(encoded_real_input_layer)
    # decoded_imag = layers.Dense(dec_dim, activation='relu')(encoded_imag_input_layer)
    # decoded_real = layers.Dense(dec_dim, activation='softmax')(encoded_real_input_layer)
    # decoded_imag = layers.Dense(dec_dim, activation='softmax')(encoded_imag_input_layer)
    # decoder_complex = keras.Model(encoded_complex_input_layer, decoded_complex)
    # decoder_real = keras.Model(encoded_real_input_layer, decoded_real)
    # decoder_imag = keras.Model(encoded_imag_input_layer, decoded_imag)
    # decoder_real.compile(optimizer='adam', loss='binary_crossentropy')
    # decoder_imag.compile(optimizer='adam', loss='binary_crossentropy')
    # decoder_real.compile(optimizer='adam', loss='mean_squared_error')
    # decoder_imag.compile(optimizer='adam', loss='mean_squared_error')
    # decoder_complex.compile(optimizer='adam', loss='mean_squared_error')
    # decoder_real.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # decoder_imag.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # decoder.fit(np.real(encoded_test_set[:,:,:train_split]), np.real(test_set[:,:,:train_split]),
    # decoder_complex.fit(encoded_test_set[:train_split,:], test_set[:train_split,:],
    #             epochs=50,
    #             batch_size=256,
    #             shuffle=True,
    #             # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
    #             validation_data=(encoded_test_set[train_split:,:], test_set[train_split:,:]))
    # decoder_real.fit(np.real(encoded_test_set[:train_split,:]), np.real(test_set[:train_split,:]),
    #             epochs=50,
    #             batch_size=256,
    #             shuffle=True,
    #             # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
    #             validation_data=(np.real(encoded_test_set[train_split:,:]), np.real(test_set[train_split:,:])))
    # decoder_imag.fit(np.imag(encoded_test_set[:train_split,:]), np.imag(test_set[:train_split,:]),
    #             epochs=50,
    #             batch_size=256,
    #             shuffle=True,
    #             # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
    #             validation_data=(np.imag(encoded_test_set[train_split:,:]), np.imag(test_set[train_split:,:])))
    # decoder_complex.save("decoder_complex.keras", overwrite=True)
    # decoder_real.save("decoder_real.keras", overwrite=True)
    # decoder_imag.save("decoder_imag.keras", overwrite=True)

    # encoded_mag_input_layer = keras.Input(shape=(enc_dim,))
    # encoded_ang_input_layer = keras.Input(shape=(dec_dim,))
    # decoded_mag = layers.Dense(dec_dim, activation='sigmoid')(encoded_real_input_layer)
    # decoded_ang = layers.Dense(dec_dim, activation='sigmoid')(encoded_imag_input_layer)
    # decoded_mag = layers.Dense(dec_dim, activation='relu')(encoded_mag_input_layer)
    # decoded_ang = layers.Dense(dec_dim, activation='relu')(encoded_ang_input_layer)

    # decoded_mag_model.add(layers.Dense(int(dec_dim/5), name="layer1", kernel_initializer=keras.initializers.Zeros()))
    # decoded_mag_model.add(layers.Dense(int(dec_dim), name="layer2"))
    # decoded_mag_model.add(layers.Dense(int(dec_dim), name="layer5", kernel_initializer=keras.initializers.Zeros()))
    # decoded_mag = layers.Dense(dec_dim, activation='softmax')(encoded_mag_input_layer)
    # decoded_ang = layers.Dense(dec_dim)(encoded_ang_input_layer)
    # decoder_mag = keras.Model(encoded_mag_input_layer, decoded_mag)
    # decoder_mag = keras.Model(inputs=decoded_mag_model.inputs, outputs=decoded_mag_model.outputs)
    # decoder_ang = keras.Model(encoded_ang_input_layer, decoded_ang)
    # decoder_real.compile(optimizer='adam', loss='binary_crossentropy')
    # decoder_imag.compile(optimizer='adam', loss='binary_crossentropy')