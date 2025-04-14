'''
@author: pete
'''
if __name__ == '__main__':
    import os
    import sys
    # Add the src directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from pathlib import Path
    from dotenv import load_dotenv
    from NYFR_Test_Harness import NYFR_Test_Harness
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    import numpy as np
    from utility import get_all_file_paths, get_file_sub_dirs, get_all_sub_dirs

    load_dotenv()

    test_harness = NYFR_Test_Harness(filenames_json=os.getenv('FILENAMES'),
                                     directories_json=os.getenv('DIRECTORIES'),
                                     input_set_json=os.getenv('INPUTSET_CONF'),
                                     system_conf_json=os.getenv('SYSTEM_CONF'))
    test_harness.create_sets()
    test_harness.create_output_sets()
    test_harness.create_dictionaries()
    test_harness.create_dfs()
    test_harness.batch_recover()

    directories = test_harness.get_directories()
    nyfr = test_harness.get_nyfr()
    del test_harness
    recovery_params = nyfr.get_recovery_params()
    dictionary_params = nyfr.get_dictionary_params()
    
    input_file_paths = get_all_file_paths(directories['input'])
    dictionary_file_list = get_all_file_paths(directories['dictionary'][dictionary_params['version']])
    dictionary_file_list_original = get_all_file_paths(directories['dictionary']['original'])
    t_test = nyfr.get_real_time()
    complex_tf = nyfr.get_frequncy_bins()
    complex_tf_sampled = nyfr.get_sampled_freq_bins()
    num_sigs_per_set = 3
    for mode in recovery_params['modes']:
        for input_file_path in input_file_paths:
            recovery_base_path = directories['recovery'][dictionary_params['version']][recovery_params['type']]
            if ( recovery_params['type'] == 'mlp1' ):
                recovery_base_path = recovery_base_path + mode + "\\"
            noise_level, phase_shift, file_name = get_file_sub_dirs(input_file_path)
            output_sub_path = directories['output'] + noise_level + "\\" + phase_shift + "\\"
            recovery_sub_path = recovery_base_path + noise_level + "\\" + phase_shift + "\\"
            output_file_sub_dirs = get_all_sub_dirs(output_sub_path)
            recovery_file_sub_dirs = get_all_sub_dirs(recovery_sub_path)
            for index, sub_dir in enumerate(recovery_file_sub_dirs):
                output_file_path = os.path.join(output_file_sub_dirs[index], file_name)
                output_sig_set = np.load(output_file_path)
                dictionary_file_name =  dictionary_file_list[index]
                dictionary_file_name_original =  dictionary_file_list_original[index]
                dictionary = np.load(dictionary_file_name)
                dictionary_original = np.load(dictionary_file_name_original)
                recovery_file_path = os.path.join(sub_dir, file_name)
                if ( os.path.exists(recovery_file_path) ):
                    input_sig_set = np.load(input_file_path)
                    output_sig_set = np.load(output_file_path)
                    recovery_sig_set = np.load(recovery_file_path)
                    for idx, output_sig in enumerate(output_sig_set):
                        if ( idx < num_sigs_per_set ):
                            recovered_signal = recovery_sig_set[idx]
                            pseudo = np.linalg.pinv(0.2*dictionary)
                            input_guess = np.dot(pseudo, output_sig)
                            pseudo_original = np.linalg.pinv(0.01*dictionary_original)
                            input_guess_original = np.dot(pseudo_original, output_sig)
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