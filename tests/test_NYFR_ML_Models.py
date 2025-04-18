'''
@author: pete
'''
if __name__ == '__main__':
    import os
    import sys
    # Add the src directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from dotenv import load_dotenv
    from NYFR_Test_Harness import NYFR_Test_Harness
    from NYFR_ML_Models import create_mlp1_models
    
    load_dotenv()

    test_harness = NYFR_Test_Harness(filenames_json=os.getenv('FILENAMES'),
                                     directories_json=os.getenv('DIRECTORIES'),
                                     input_set_json=os.getenv('INPUTSET_CONF'),
                                     system_conf_json=os.getenv('SYSTEM_CONF'))
    input_set_params = test_harness.get_input_set_params()
    input_set_params["noise_levels"] = [["no_noise", []]]
    input_set_params["phase_shifts"] = [["no_phase_shift", []]]
    input_set_params["f_mods"] = [["f_mod_0_5", 0.5]]
    input_set_params["f_deltas"] = [["f_delta_0_8", 0.8]]
    input_set_params["input_tones"] = [["5", [5]]]
    test_harness.set_input_set_params(input_set_params)
    create_mlp1_models(test_harness, training_conf=os.getenv('TRAINING_CONF'))
    test_harness.batch_recover()
    test_harness.display_test_signals()