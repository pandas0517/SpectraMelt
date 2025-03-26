'''
Created on Jul 10, 2024

@author: pete
'''
if __name__ == '__main__':
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from queue import Queue
    # from signals import simulate_system
    from scipy.fft import fft, ifft
    from math import pi
    from decoder import get_all_file_paths, get_all_sub_dirs, create_nyfr_output
    from decoder import batch_recover, create_dictionaries, create_mlp1_models
    from decoder import get_all_file_names, meta_input_output, analyze_dfs
    from OMP import OMP
    # from decoder import create_test_set, create_decoder, create_dictionaries
    # import tensorflow as tf



    system_params = {
        'dictionary': 'real',
        'filter': 'butter',
        'sampled_LO': 'y',
        'wavelets': 'n',
        'rand_demod': 'n',
        'rd_clock_freq': 4,
        'adc_clock_freq': 100,
        'wave_freq': [10,20],
        'start': -2,
        'stop': 2,
        'spacing': 0.001,
        'wbf_cut_mod': 4,
        'recovery': 'mlp1',
        'noise': 0 }
    #Input signal parameters
    wave_params = [
        {'amp': 1,
         'freq': 25,
         'phase': 0},
        {'amp': 0,
         'freq': 85,
         'phase': 0},
        {'amp': 0,
         'freq': 140,
         'phase': 0},
        {'amp': 0.1,
         'freq': 255,
         'phase': 0},
        {'amp': 0.1,
         'freq': 395,
         'phase': 0}]
    #Phase modulated local oscillator (NYFR) parameters
    LO_params = {
        'amp':1,
        'freq':100,
        'phase':0,
        'phase_delta': 0.4,
        'phase_freq': 0.5,
        'phase_offset': 0}
    # Gabor atoms with Gaussian window parameters
    # f_c = center frequencies of atoms
    # width = Gaussian Standard Deviation
    # large width increases frequency resolution while reducing time resolution
    # Filtering angle of fractional Fourier domain 
    psi_params = [
        {
            'amp': 0.5,
            'f_c': 20,
            'width': 0.001,
            'shift': 0,
            'angle': pi/4
        }
    ]
    # Filter parameters
    filter_params = {
        'order': 6,
        'angle': pi/2,
        'cutoff_freq': 50,
        'window_size': 150}

    # create_nyfr_output(system_params, filter_params, LO_params)
    # create_dictionaries(system_params, LO_params, filter_params)
    # create_mlp1_models(system_params)
    # batch_recover(system_params)
    # meta_input_output(system_params)
    # analyze_dfs(system_params)
    pass
    # modes = [ 'real_imag', 'mag_ang', 'complex' ]
    modes = [ 'mag_ang' ]
    # modes = [ 'mag_ang' ]
    num_sigs = 20
    recovery_dic_type = 'original'
    directory_list = {
        'input': "test_sets\\System_Config_1_Inputs\\",
        'output': "test_sets\\System_Config_1_Outputs\\",
        'train': "test_sets\\System_Config_1_Model_Inputs\\train\\",
        'test': "test_sets\\System_Config_1_Model_Inputs\\test\\",
        'time_file': "test_sets\\System_Config_1_Internal\\System\\Not_Sampled\\time.npy",
        'complex_tf_file': "test_sets\\System_Config_1_Internal\\System\\Not_Sampled\\complex_tf.npy",
        'complex_tf_sampled_file': "test_sets\\System_Config_1_Internal\\System\\Sampled\\complex_tf_sampled.npy",
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
    # input_file_paths = get_all_file_paths(directory_list['train'])
    dictionary_file_list = get_all_file_paths(directory_list['dictionary'][recovery_dic_type])
    dictionary_file_list_original = get_all_file_paths(directory_list['dictionary']['original'])
    # dictionary_file_list = get_all_file_paths(dictionary_base_directory)
    # t_test = np.load("test_sets\\System_Config_1_Internal\\System\\Not_Sampled\\time.npy")
    # complex_tf = np.load("test_sets\\System_Config_1_Internal\\System\\Not_Sampled\\complex_tf.npy")
    # complex_tf_sampled = np.load("test_sets\\System_Config_1_Internal\\System\\Sampled\\complex_tf_sampled.npy")
    t_test = np.load(directory_list['time_file'])
    complex_tf = np.load(directory_list['complex_tf_file'])
    complex_tf_sampled = np.load(directory_list['complex_tf_sampled_file'])
    for mode in modes:
    # mlp_models_path = directory_list['mlp_models'][recovery_dic_type]
        for input_set_file in input_file_paths:
            recovery_base_path = directory_list['recovery'][recovery_dic_type][system_params['recovery']]
            if ( system_params['recovery'] == 'mlp1' ):
                recovery_base_path = recovery_base_path + mode + "\\"
            # input_sig_set = np.load(input_set_file)
            input_path = Path(input_set_file)
            input_path_len = len(input_path.parts)
            input_file_name = input_path.parts[input_path_len - 1]
            input_phase_shift = input_path.parts[input_path_len - 2]
            input_noise_level = input_path.parts[input_path_len - 3]

            # input_file_name = input_path.parts[5]
            # input_noise_level = input_path.parts[3]
            # input_phase_shift = input_path.parts[4]
            output_sub_path = directory_list['output'] + input_noise_level + "\\" + input_phase_shift + "\\"
            recovery_sub_path = recovery_base_path + input_noise_level + "\\" + input_phase_shift + "\\"
            # mlp_models_sub_path = mlp_models_path + input_noise_level + "\\" + input_phase_shift + "\\"
            output_file_sub_dirs = get_all_sub_dirs(output_sub_path)
            recovery_file_sub_dirs = get_all_sub_dirs(recovery_sub_path)
            # mlp_models_sub_dirs = get_all_sub_dirs(mlp_models_sub_path)
            for index, sub_dir in enumerate(recovery_file_sub_dirs):
                output_file_path = os.path.join(output_file_sub_dirs[index], input_file_name)
                # mlp_model_file_path = os.path.join(mlp_models_sub_dirs[index], directory_list['mlp_models']['file_name'])
                output_sig_set = np.load(output_file_path)
                dictionary_file_name =  dictionary_file_list[index]
                dictionary_file_name_original =  dictionary_file_list_original[index]
                dictionary = np.load(dictionary_file_name)
                dictionary_original = np.load(dictionary_file_name_original)
                recovery_file_path = os.path.join(sub_dir, input_file_name)
                # if ( not os.path.exists(recovery_file_path) ):
                if ( os.path.exists(recovery_file_path) ):
                    input_sig_set = np.load(input_set_file)
                    output_sig_set = np.load(output_file_path)
                    recovery_sig_set = np.load(recovery_file_path)
                    for idx, output_sig in enumerate(output_sig_set):
                        if ( idx < num_sigs ):
                            recovered_signal = recovery_sig_set[idx]
                            pseudo = np.linalg.pinv(0.2*dictionary)
                            input_guess = np.dot(pseudo, output_sig)
                            pseudo_original = np.linalg.pinv(0.01*dictionary_original)
                            input_guess_original = np.dot(pseudo_original, output_sig)
                            # input_guess_omp = OMP(0.01*np.abs(dictionary_original), output_sig)[0]
                            input_sig_xf = fft(input_sig_set[idx])
                            model_sig_xf_guess = fft(np.dot(dictionary, input_sig_xf))
                            model_sig_xf_guess_original = fft(np.dot(dictionary_original, input_sig_xf))
                            output_sig_xf = fft(output_sig)
                            premultiply = fft(np.dot(dictionary, input_sig_xf))
                            plt.figure()
                            plt.subplot(7,1,1)
                            plt.plot(complex_tf, np.fft.fftshift(np.abs(input_sig_xf)))
                            plt.xlim(-500,500)
                            plt.subplot(7,1,2)
                            plt.plot(complex_tf, np.fft.fftshift(np.abs(recovered_signal)))
                            plt.xlim(-500,500)
                            plt.subplot(7,1,3)
                            plt.plot(complex_tf, np.fft.fftshift(np.abs(input_guess_original)))
                            plt.xlim(-500,500)
                            plt.subplot(7,1,4)
                            plt.plot(complex_tf, np.fft.fftshift(np.angle(input_guess_original)))
                            plt.xlim(-500,500)
                            plt.subplot(7,1,5)
                            plt.plot(complex_tf_sampled, np.fft.fftshift(np.abs(output_sig_xf)))
                            plt.subplot(7,1,6)
                            plt.plot(complex_tf_sampled, np.fft.fftshift(np.abs(model_sig_xf_guess)))
                            plt.subplot(7,1,7)
                            plt.plot(complex_tf_sampled, np.fft.fftshift(np.abs(model_sig_xf_guess_original)))
                            plt.show()
                            # input_sig_1 = input_sig_set[idx]
                            # coef_split = np.split(input_sig_1, 2)
                            # coef_real = coef_split[0]
                            # coef_imag = coef_split[1]
                            # input_sig = coef_real + 1j*coef_imag
                            # plt.xlim(-500,500)
                            # plt.subplot(5,1,1)
                            # plt.plot(t_test, test_data)
                            # plt.subplot(5,1,2)
                            # plt.plot(complex_tf, np.fft.fftshift(abs(test_data_xf)))
                            # plt.subplot(6,1,4)
                            # plt.plot(complex_tf, np.fft.fftshift(np.angle(test_data_xf, deg=True)))
                            # plt.subplot(6,1,4)
                            # plt.plot(tf_sampled, np.fft.fftshift(abs(output_test_data_xf)))
                            # plt.subplot(6,1,5)
                            # plt.plot(tf_sampled, np.fft.fftshift(np.real(output_test_data_xf)))
                            # plt.subplot(6,1,6)
                            # plt.plot(tf_sampled, np.fft.fftshift(np.imag(output_test_data_xf)))
            # input_path_list = input_set_file.split("\\")
            # output_file_name = input_path_list.pop()
            # output_base_path_list = input_path_list.copy()
            # recovery_base_path_list = input_path_list.copy()
            # output_base_path_list[1] = output_base_directory_name
            # recovery_base_path_list[1] = recovery_base_directory_names[system_params['recovery']]
            # output_base_dir = os.path.join(*output_base_path_list)
            # output_file_sub_dirs = get_all_sub_dirs(output_base_dir)
            # recovery_base_dir = "E:\\" + os.path.join(*recovery_base_path_list)
            # recovery_file_sub_dirs = get_all_sub_dirs(recovery_base_dir)
            # for index, sub_dir in enumerate(recovery_file_sub_dirs):
            #     output_file_path = os.path.join(output_file_sub_dirs[index % len(output_file_sub_dirs)], output_file_name)
            #     output_sig_set = np.load(output_file_path)
            #     dictionary_file_name =  dictionary_file_list[index]
            #     dictionary = np.load(dictionary_file_name)
            #     recovery_file_path = os.path.join(sub_dir, output_file_name)
    # input_directory = "test_sets\\System_Config_1_Inputs\\"
    # input_file_paths = get_all_file_paths(input_directory)
    # output_base_directory_name = "System_Config_1_Outputs"
    # recovery_base_directory_names = {
    #     'c_omp': 'System_Config_1_OMP_Custom_Recovery',
    #     'o_omp': 'System_Config_1_OMP_Recovery',
    #     'mlp1': 'System_Config_1_MLP1_Recovery',
    #     'spgl1': 'System_Config_1_SPGL_Recovery'
    # }
    # dictionary_base_directory = "test_sets\\System_Config_1_Internal\\Dictionary\\"
    # create_dictionaries(system_params, LO_params, filter_params)
    # batch_recover(system_params)
    # test_set, encoded_test_set, dic_test_set, decoder_test_set = create_test_set(dictionary, t, system_params, wave_params, filter_params, LO_params)
    # np.save("dictionary.npy", dictionary)
    # np.save("test_set.npy", test_set)
    # np.save("encoded_test_set.npy", encoded_test_set)
    # np.save("dic_test_set.npy", dic_test_set)
    # np.save("decoder_test_set.npy", decoder_test_set)
    # tdictionary = np.load("dictionary.npy")
    # ttest_set = np.load("test_set.npy")
    # tencoded_test_set = np.load("encoded_test_set.npy")
    # tdic_test_set = np.load("dic_test_set.npy")
    # decoder = create_decoder(tdictionary, ttest_set, tdic_test_set, tencoded_test_set)
    # decoder_real = tf.keras.models.load_model('decoder_real.keras')
    # decoder_imag = tf.keras.models.load_model('decoder_imag.keras')
    # test_set_size = (test_set.shape)[0]
    # for i in range(0, test_set_size):
    #     plt.figure()
    #     plt.subplot(3,1,1)
    #     plt.plot(complex_tf,np.fft.fftshift(abs(test_set[i])))
    #     plt.xlim(-500,500)
    #     plt.subplot(3,1,2)
    #     plt.plot(tf_sampled,np.fft.fftshift(abs(fft(encoded_test_set[i]))))
    #     plt.subplot(3,1,3)
    #     plt.plot(complex_tf,np.fft.fftshift(abs(dic_test_set[i])))
        # plt.subplot(5,1,4)
        # pseudo = np.linalg.pinv(dictionary)
        # # x_pre = np.linalg.inv(np.dot(np.conjugate(dictionary.T),dictionary)).dot(np.conjugate(dictionary.T)).dot(dic_test_set[i])
        # x_pre = np.dot(pseudo,dic_test_set[i])
        # plt.plot(complex_tf,np.fft.fftshift(abs(x_pre)))
        # plt.subplot(5,1,5)
        # x_pre1 =np.dot(dictionary.T,dic_test_set[i])
        # plt.plot(complex_tf,np.fft.fftshift(abs(x_pre1)))
        # plt.show()       
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.title("ADC frequency: " + str(adc_freq) + "Hz | LO frequency (f_s1): " + str(LO_freq) + \
    #         "Hz\nModulation Frequency (f_mod): " + str(LO_params['phase_freq']) + "Hz | Modulation Deviation (f_delta): " + \
    #         str(LO_params['phase_delta']) + "\nOriginal Frequency Domain Magnitude Signal\nTones at:" + str(wave_params[0]["freq"]) + "Hz, " + str(wave_params[2]["freq"]) + "Hz, " + str(wave_params[3]["freq"]) + "Hz, and " + str(wave_params[4]["freq"]) +"Hz")
    # plt.plot(complex_tf,np.fft.fftshift(abs(xf)))
    # plt.subplot(2,1,2)
    # plt.title("Recovered Magnitude Signal")
    # plt.plot(complex_tf,np.fft.fftshift(abs(coef)))
    # plt.plot(t,x)
    # plt.subplot(2,1,2)
    # plt.plot(complex_tf,np.fft.fftshift(np.real(coef)))
    # plt.show()
    # eps = 1+0*1j
    # tttt_1d = np.arange(45)
    # tttt_3d = tttt_1d.reshape((3,3,5))
    # tttt_3d = tttt_3d[:,:,np.random.permutation(tttt_3d.shape[2])]
    # for n in range(5):
    #     print(tttt_3d[:,:,n])
    #     pass
    # eps = 15
    # max_lo_freq = 4 + LO_params['freq']
    # num_lo_freq = int(( max_lo_freq - LO_params['freq'] ) / 2 )
    # num_lo_freq = int( max_lo_freq - LO_params['freq'] )
    # lo_freq_range = np.linspace(LO_params['freq'], max_lo_freq, num_lo_freq, endpoint=False)
    # lo_freq_range = [LO_params['freq']]
    # manager_queue = Queue()
    # for lo_freq in lo_freq_range:
    #     LO_params['freq'] = lo_freq
    #     system_params['adc_clock_freq'] = lo_freq
    #     print(lo_freq) 
    #     simulate_system(wave_params, eps, LO_params, system_params, psi_params, filter_params, manager_queue)
    # plt.title("Recovered signal from noise-free measurements")
    # plt.stem(idx_r, coef[idx_r])
    # while not manager_queue.empty():
    #     signals = manager_queue.get()
    #     LO_freq = signals[0]
    #     adc_freq = signals[1]
    #     complex_tf = signals[2]
    #     xf = signals[3]
    #     coef = signals[4]
    #     matching_tones = signals[5]
    #     zero_crossings = signals[6]
    #     LO_mix = signals[7]
    #     y_filtered = signals[8]
    #     t = signals[9]
    #     t_sampled = signals[10]
    #     tf_sampled = signals[11]
    #     y_sampled = signals[12]
    #     test_model = signals[13]
    #     filt_freq = signals[14]
    #     filt_freq_down = signals[15]
    #     y_mixed = signals[16]
    #     LO = signals[17]
    #     x = signals[18]
    #     wavelet_train = signals[19]
    #     single_wavelet = signals[20]
    #     filt_down = signals[21]
    #     downsample_train = signals[22]
    #     dictionary = signals[23]
        # y_start = np.where( complex_tf == -1000 )[0][0]
        # y_end = np.where( complex_tf == 1000 )[0][0]
        # test_y1 = np.copy(np.fft.fftshift(abs(y_mixed)))
        # y_mixed_max = test_y1[y_start:y_end].max()