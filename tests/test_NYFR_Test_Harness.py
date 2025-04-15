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

    recovery_params = test_harness.get_recovery_params()
    dictionary_params = test_harness.get_dictionary_params()
    directories = test_harness.get_directories()
    filenames = test_harness.get_filenames()
    input_set_params = test_harness.get_input_set_params()
    system_params = test_harness.get_system_params()
    t_test = test_harness.get_real_time()
    complex_tf = test_harness.get_frequncy_bins()
    complex_tf_sampled = test_harness.get_sampled_freq_bins()
    num_sigs_per_set = 3
    num_subplots = 4
    for processing_system in system_params['processing_systems']:
        for mode in recovery_params['modes']:
            for noise_level, _ in input_set_params["noise_levels"]:
                for phase_shift, _ in input_set_params["phase_shifts"]:
                    for input_tones, _ in input_set_params["input_tones"]:
                        input_file_path = os.path.join(directories['input'],
                                                    noise_level,
                                                    phase_shift,
                                                    filenames["input_tones"][input_tones]['sigs'])
                        recovery_base_path = directories["recovery"][dictionary_params['version']][recovery_params['type']]
                        if ( recovery_params['type'] == 'MLP1' ):
                            recovery_base_path = os.path.join(recovery_base_path,
                                                            mode)
                        for f_mod, _ in input_set_params["f_mods"]:
                            for f_delta, _ in input_set_params["f_deltas"]:
                                output_file_path = os.path.join(directories['output'],
                                                                noise_level,
                                                                phase_shift,
                                                                f_mod,
                                                                f_delta,
                                                                filenames["input_tones"][input_tones]['sigs'])
                                dictionary_file_path = os.path.join(directories['dictionary'][dictionary_params['version']],
                                                                    f_mod,
                                                                    f_delta,
                                                                    filenames['dictionary']['name'])
                                recovery_file_path = os.path.join(recovery_base_path,
                                                                noise_level,
                                                                phase_shift,
                                                                f_mod,
                                                                f_delta,
                                                                filenames['recovery']['name'])
                                if ( os.path.exists(recovery_file_path) ):
                                    input_sig_set = np.load(input_file_path)
                                    output_sig_set = np.load(output_file_path)
                                    recovery_sig_set = np.load(recovery_file_path)
                                    dictionary = np.load(dictionary_file_path)
                                    for idx, output_sig in enumerate(output_sig_set):
                                        if ( idx < num_sigs_per_set ):
                                            recovered_signal = recovery_sig_set[idx]
                                            pseudo = np.linalg.pinv(0.2*dictionary)
                                            input_guess = np.dot(pseudo, output_sig)
                                            input_sig_xf = fft(input_sig_set[idx])
                                            model_sig_xf_guess = fft(np.dot(dictionary, input_sig_xf))
                                            output_sig_xf = fft(output_sig)
                                            premultiply = fft(np.dot(dictionary, input_sig_xf))
                                            plt.figure()
                                            plt.subplot(num_subplots,1,1)
                                            plt.plot(complex_tf, np.fft.fftshift(np.abs(input_sig_xf)))
                                            plt.xlim(-system_params["wbf_cut_freq"],system_params["wbf_cut_freq"])
                                            plt.subplot(num_subplots,1,2)
                                            plt.plot(complex_tf, np.fft.fftshift(np.abs(recovered_signal)))
                                            plt.xlim(-system_params["wbf_cut_freq"],system_params["wbf_cut_freq"])
                                            plt.subplot(num_subplots,1,3)
                                            plt.plot(complex_tf_sampled, np.fft.fftshift(np.abs(output_sig_xf)))
                                            plt.subplot(num_subplots,1,4)
                                            plt.plot(complex_tf_sampled, np.fft.fftshift(np.abs(model_sig_xf_guess))) 