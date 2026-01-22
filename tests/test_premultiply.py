'''
@author: pete
'''
if __name__ == '__main__':
    import os
    import sys
    from scipy.fft import fft
    import numpy as np
    import matplotlib.pyplot as plt
    # Add the src directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from dotenv import load_dotenv
    from NYFR_Test_Harness_v2 import NYFR_Test_Harness
    
    load_dotenv()

    test_harness = NYFR_Test_Harness(filenames_json=os.getenv('FILENAMES'),
                                     directories_json=os.getenv('DIRECTORIES'),
                                     input_set_json=os.getenv('INPUTSET_CONF'),
                                     system_conf_json=os.getenv('SYSTEM_CONF'))
    input_set_params = test_harness.get_input_set_params()
    input_set_params["noise_levels"] = [["no_noise", []]]
    input_set_params["phase_shifts"] = [["no_phase_shift", []]]
    input_set_params["input_tones"] = [["1_2", [1,2]]]
    input_set_params["f_mods"] = [["f_mod_0_1", 0.1]]
    input_set_params["f_deltas"] = [["f_delta_0_1", 0.1]]
    recovery_params = test_harness.get_recovery_params()
    dictionary_params = test_harness.get_dictionary_params()
    system_params = test_harness.get_system_params()
    files = test_harness.get_filenames()
    directories = test_harness.get_directories()
    t = test_harness.get_time()
    complex_tf = test_harness.get_frequncy_bins()
    t_sampled = test_harness.sample_signals(t, points_per_second=test_harness.get_points_per_second(),t=t)
    complex_tf_sampled = np.linspace(-test_harness.get_adc_clock_freq()/2, test_harness.get_adc_clock_freq()/2, int(t_sampled.size), endpoint=False)
    num_sigs_per_set = 10
    num_subplots = 6
    for processing_system in system_params['processing_systems']:
        for mode in recovery_params['modes']:
            for noise_level, _ in input_set_params["noise_levels"]:
                for phase_shift, _ in input_set_params["phase_shifts"]:
                    for input_tones, _ in input_set_params["input_tones"]:
                        input_file_path = os.path.join(directories['input'],
                                                    noise_level,
                                                    phase_shift,
                                                    files['input_tones'][input_tones]['sigs'])
                        fft_file_path = os.path.join(directories['fft'],
                                                        noise_level,
                                                        phase_shift,
                                                        files['input_tones'][input_tones]["sigs"])
                        recovery_base_path = directories['recovery'][dictionary_params['version']][recovery_params['type']]
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
                                                                files['input_tones'][input_tones]['sigs'])
                                premultiply_file_path = os.path.join(directories['premultiply'][dictionary_params['version']],
                                                                noise_level,
                                                                phase_shift,
                                                                f_mod,
                                                                f_delta,
                                                                files['input_tones'][input_tones]["sigs"])
                                dictionary_file_path = os.path.join(directories['dictionary'][dictionary_params['version']],
                                                                    f_mod,
                                                                    f_delta,
                                                                    files['dictionary']['name'])
                                recovery_file_path = os.path.join(recovery_base_path,
                                                                noise_level,
                                                                phase_shift,
                                                                f_mod,
                                                                f_delta,
                                                                files['input_tones'][input_tones]['sigs'])
                                if ( os.path.exists(recovery_file_path) ):
                                    input_sig_set = np.load(input_file_path)
                                    output_sig_set = np.load(output_file_path)
                                    recovery_sig_set = np.load(recovery_file_path)
                                    dictionary = np.load(dictionary_file_path)
                                    premultiply_sig_set = np.load(premultiply_file_path)
                                    fft_sig_set = np.load(fft_file_path)
                                    for idx, premultiply_sig in enumerate(premultiply_sig_set):
                                        if ( idx < num_sigs_per_set ):
                                            recovered_signal = recovery_sig_set[idx]
                                            # active_zones = np.zeros_like(recovered_signal)
                                            premultiply_sig = premultiply_sig_set[idx]
                                            output_sig = output_sig_set[idx]
                                            pseudo = np.linalg.pinv((4*dictionary)/test_harness.get_adc_clock_freq())
                                            input_guess = np.dot(pseudo, output_sig)
                                            input_sig_xf = fft(input_sig_set[idx])/(2*test_harness.get_wb_nyquist_rate())
                                            fft_sig = fft_sig_set[idx]
                                            model_sig_xf_guess = fft(np.dot(dictionary, input_sig_xf))
                                            output_sig_xf = fft(output_sig)
                                            # input_zones = np.array_split(np.fft.fftshift(np.abs(input_sig_xf)), test_harness.get_Zones())
                                            # for i, zone in enumerate(input_zones):
                                            #     if np.any( zone > 500 ):
                                            #         active_zones[i] = 1
                                            plt.figure()
                                            plt.subplot(num_subplots,1,1)
                                            plt.plot(complex_tf, np.fft.fftshift(np.abs(input_sig_xf)))
                                            plt.subplot(num_subplots,1,2)
                                            plt.plot(complex_tf, np.fft.fftshift(np.abs(recovered_signal)))
                                            plt.subplot(num_subplots,1,3)
                                            plt.plot(complex_tf, np.fft.fftshift(np.abs(input_guess)))
                                            # plt.subplot(num_subplots,1,4)
                                            # plt.plot(complex_tf, np.fft.fftshift(np.abs(premultiply_sig_set[idx])))
                                            plt.subplot(num_subplots,1,5)
                                            plt.plot(complex_tf_sampled, np.fft.fftshift(np.abs(output_sig_xf)))
                                            plt.subplot(num_subplots,1,6)
                                            plt.plot(complex_tf_sampled, np.fft.fftshift(np.abs(model_sig_xf_guess))) 
                                            plt.show()
                                    # np.save(premultiply_file_path, premultiply_sig_set)