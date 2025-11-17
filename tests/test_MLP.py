'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import (
        load_config_from_json,
        get_logger,
        save_to_json
    )
    from pathlib import Path
    from spectramelt.MLP import MLP
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.fft import fft, ifft, fftshift, ifftshift

    load_dotenv()
    
    create_mlp_model = True
    prepare_large_dataset = False
    train_mlp_model = True
    
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
    ml_models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = directories.get('outputs', "Outputs")
    
    inputset_params = dataset.get_inputset_params()
    saved_freq_modes = inputset_params.get('saved_freq_modes', [])
    # Currently: Get Magnitude Mode
    selected_freq_modes = saved_freq_modes[0:1]
    freq_file_keys = dataset.get_freq_file_keys()
    flat_filenames = dataset.get_flat_filenames()
    ml_model_filename = flat_filenames.get('ml_model', "ml_model.keras")
    ml_config_filename = flat_filenames.get('ml_config', "ml_config.json")
    ml_config_file = ml_models_dir.parent / ml_config_filename

    if not ml_config_file.exists():
        save_to_json(mlp.get_mlp_params(), ml_config_file)

    if saved_freq_modes:
        for mode in selected_freq_modes:
            key = freq_file_keys[mode]
            
            filename = flat_filenames.get(key)
            premultiply_file = premultiply_dir / filename
            premultiply_h5_file = Path(premultiply_file).with_suffix(".h5")
            
            get_test_signal = True
            premultiply_file_list = []
            premultiply_test_sig = None
            for file_path in premultiply_dir.iterdir():
                if (file_path.is_file() and
                    file_path.name.endswith(filename) and 
                    "recovery" not in file_path.name.lower()):
                    premultiply_file_list.append(file_path)
                    if get_test_signal:
                        premultiply_signals = np.load(file_path)
                        premultiply_test_sig = premultiply_signals[0]
                        get_test_signal = False
                        
            output_file = output_dir / f"wbf_{filename}"
            output_h5_file = Path(output_file).with_suffix(".h5")
            
            get_test_signal = True
            output_file_list = []
            output_test_sig = None
            for file_path in output_dir.iterdir():
                if (file_path.is_file() and
                    file_path.name.endswith(filename) and
                    "wbf" in file_path.name.lower() and 
                    "recovery" not in file_path.name.lower()):
                    output_file_list.append(file_path)
                    if get_test_signal:
                        output_signals = np.load(file_path)
                        output_test_sig = output_signals[0]
                        get_test_signal = False
                    
            ml_model_file = ml_models_dir / f"{mode}_{ml_model_filename}"
            model_params = mlp.get_model_params()
            model_params['file_path'] = ml_model_file
            mlp.set_model_params(model_params)
            
            if create_mlp_model:
                mlp.create_model(len(premultiply_test_sig), len(output_test_sig))

            if prepare_large_dataset:
                mlp.prepare_large_dataset(premultiply_file_list,
                                          output_file_list,
                                          premultiply_h5_file,
                                          output_h5_file,
                                          sample_signal=premultiply_test_sig)
                
            if train_mlp_model:
                mlp.train_on_hdf5(premultiply_h5_file, output_h5_file)
            pass
    
    atexit.register(logger.info, "Completed Test\n")