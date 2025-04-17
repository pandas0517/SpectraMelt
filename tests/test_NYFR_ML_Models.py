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
    # create_mlp1_models(test_harness, training_conf=os.getenv('TRAINING_CONF'))
    test_harness.batch_recover()
    test_harness.display_test_signals()