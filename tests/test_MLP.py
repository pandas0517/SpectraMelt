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
    
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np

    load_dotenv()
    
    create_mlp_model = False
    prepare_large_dataset = False
    use_normed_h5_file = True
    train_mlp_model = True
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr_config = load_config_from_json(Path(getenv('NYFR_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr_config.get('config_name'),
                      config_file_path=Path(getenv('DATASET_CONF')))
    
    directories = dataset.get_directories()
    premultiply_dir = directories.get('premultiply', "Premultiply")
    ml_models_dir = directories.get('ml_models', "ML_Models")
    ml_models_dir.mkdir(parents=True, exist_ok=True)
    wideband_dir = directories.get('wideband', "Wideband")
    
    freq_modes = dataset.get_freq_modes()
    mlp_freq_modes = freq_modes.get('mlp', [])
    # selected_freq_modes = saved_freq_modes[1:2]
    selected_freq_modes = mlp_freq_modes
    filenames = dataset.get_filenames()
    ml_model_filename = filenames.get('ml_model', "ml_model.keras")
    ml_config_filename = filenames.get('ml_config', "ml_config.json")
    freq_signal_filename = filenames.get('freq_signals', "freq_signals.npz")
    ml_config_file = ml_models_dir.parent / ml_config_filename

    if selected_freq_modes:
        for mode in selected_freq_modes:           
            premultiply_file_list = []
            premultiply_test_sig = None

            get_test_signal = True
            for file_path in premultiply_dir.iterdir():
                if (file_path.is_file() and
                    file_path.name.endswith(freq_signal_filename) and 
                    "recovery" not in file_path.name.lower() and
                    "centered" in file_path.name.lower()):
                    premultiply_file_list.append(file_path)
                    if get_test_signal:
                        premultiply_test_signals = np.load(file_path)
                        premultiply_signals = premultiply_test_signals[mode]
                        premultiply_test_sig = premultiply_signals[0]
                        del premultiply_signals
                        del premultiply_test_signals
                        get_test_signal = False
            
            get_test_signal = True
            output_file_list = []
            output_test_sig = None
            for file_path in wideband_dir.iterdir():
                if (file_path.is_file() and
                    file_path.name.endswith(freq_signal_filename) and
                    "recovery" not in file_path.name.lower()):
                    output_file_list.append(file_path)
                    if get_test_signal:
                        output_test_signals = np.load(file_path)
                        output_signals = output_test_signals[mode]
                        output_test_sig = output_signals[0]
                        del output_signals
                        del output_test_signals
                        get_test_signal = False

            if use_normed_h5_file:
                premultiply_h5_file = premultiply_dir / f"{Path(freq_signal_filename).stem}_{mode}_norm.h5"
                output_h5_file = wideband_dir / f"wbf_{Path(freq_signal_filename).stem}_{mode}_norm.h5"
            else:
                premultiply_h5_file= premultiply_dir / f"{Path(freq_signal_filename).stem}_{mode}.h5"
                output_h5_file = wideband_dir / f"wbf_{Path(freq_signal_filename).stem}_{mode}.h5"               
                    
            ml_model_file = ml_models_dir / f"{mode}_{ml_model_filename}"

            from spectramelt.MLP import MLP
            mlp = MLP(config_file_path=Path(getenv('MLP_CONF')))
            if not ml_config_file.exists():
                save_to_json(mlp.get_mlp_params(), ml_config_file)
            mlp.set_model_file_path(ml_model_file)
            
            if create_mlp_model:
                mlp.create_model(len(premultiply_test_sig), len(output_test_sig))

            if prepare_large_dataset:
                mlp.prepare_large_dataset(premultiply_file_list,
                                          output_file_list,
                                          premultiply_h5_file,
                                          output_h5_file,
                                          mode,
                                          sample_signal=premultiply_test_sig)
                
            if train_mlp_model:
                mlp.train_on_hdf5(premultiply_h5_file, output_h5_file)
                mlp.reset_tensorflow_session()
    
    atexit.register(logger.info, "Completed Test\n")