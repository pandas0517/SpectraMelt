'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import load_config_from_json, get_logger
    from pathlib import Path
    from spectramelt.MLP import MLP
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift, ifftshift

    load_dotenv()
    
    create_output_set = True

    create_nyfr_wave_params = True
    display_nyfr_signals = True
    
    create_premultiply_set = True
    display_premultiply_signals = True
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr_config = load_config_from_json(Path(getenv('NYFR_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr_config.get('config_name'),
                      config_file_path=Path(getenv('DATASET_CONF')))
    mlp = MLP(config_file_path=Path(getenv('MLP_CONF')))
    
    directories = dataset.get_directories()
    premultiply_dir = directories.get('premultiply', "Premultiply")
    ml_models_dir = directories.get('ml_models', "ML_Models")
    input_dir = directories.get('outputs', "Outputs")
    
    inputset_params = dataset.get_inputset_params()
    saved_freq_modes = inputset_params.get('saved_freq_modes', [])
    # Currently: Get Magnitude Mode
    selected_freq_modes = saved_freq_modes[0]
    freq_file_keys = dataset.get_freq_file_keys()
    flat_filenames = dataset.get_flat_filenames()
    ml_model_filename = flat_filenames.get('ml_model', "ml_model.keras")

    if saved_freq_modes:
        for mode in selected_freq_modes:
            get_test_signal = True
            key = freq_file_keys[mode]
            
            filename = flat_filenames.get(key)
            premultiply_file = premultiply_dir / filename
            premultiply_h5_file = Path(premultiply_file).with_suffix(".h5")
            
            premultiply_file_list = []
            premultiply_test_sig = None
            for file_path in premultiply_dir.iterdir():
                if file_path.is_file() and file_path.name.endswith(filename):
                    premultiply_file_list.append(file_path)
                    if get_test_signal:
                        premultiply_signals = np.load(file_path)
                        premultiply_test_sig = premultiply_signals[0]
                        get_test_signal = False
                        
            input_file = input_dir / filename
            input_h5_file = Path(input_file).with_suffix(".h5")
            
            input_file_list = []
            for file_path in input_dir.iterdir():
                if file_path.is_file() and file_path.name.endswith(filename):
                    input_file_list.append(file_path)
                    
            ml_model_file = ml_models_dir / f"{mode}_{ml_model_filename}"
            pass
    
    atexit.register(logger.info, "Completed Test\n")