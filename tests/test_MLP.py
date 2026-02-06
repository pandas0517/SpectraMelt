'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import (
        load_config_from_json,
        get_logger,
        save_to_json,
        plot_dynamic_frequency_modes,
        REQUIRED_AXIS_KEYS
    )
    from pathlib import Path
    from spectramelt.mlp_module import MLP
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np

    load_dotenv()
    
    create_premultiply_set = False
    display_premultiply_signals = False
    
    create_mlp_model = True
    prepare_large_dataset = True
    train_mlp_model = True
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr_config = load_config_from_json(Path(getenv('NYFR_CONF')))
    recovery_config = load_config_from_json(Path(getenv('RECOVERY_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr_config.get('config_name'),
                      config_file_path=Path(getenv('DATASET_CONF')))
    mlp = MLP(config_file_path=Path(getenv('MLP_CONF')))

    if create_premultiply_set:
        dataset.create_premultiply_set(nyfr_config, mlp.get_premultiply_params())
    
    directories = dataset.get_directories()
    premultiply_dir = Path(directories.get('premultiply', "Premultiply"))
    ml_models_dir = Path(directories.get('ml_models', "ML_Models"))
    ml_models_dir.mkdir(parents=True, exist_ok=True)
    wideband_dir = Path(directories.get('wideband', "Wideband"))
    
    filenames = dataset.get_filenames()
    input_freq_signal_filename = filenames.get('freq_signals', "freq_signals.npz")
    
    if display_premultiply_signals:
        freq_modes = nyfr_config['freq_modes']      
        wideband_freq_modes = freq_modes.get('wideband', [])
        wave_params = input_config.get('wave_params', None)
        freq_range = tuple(wave_params.get('freq_range'))
        
        wbf_time_freq_filename = filenames.get('wbf_time_freq', "wbf_time_freq.npz")
        time_freq_file = wideband_dir / wbf_time_freq_filename
        with np.load(time_freq_file) as time_freq:
            missing = [k for k in REQUIRED_AXIS_KEYS if k not in time_freq]
            if missing:
                raise ValueError(f"{time_freq_file} missing required arrays: {missing}")
            time = time_freq["time"]
            freq = time_freq["freq"]
        
        N = len(freq)
        signals_per_file = 3
        
        for file_path in premultiply_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_freq_signal_filename):         

                time_signals = None
                base_title = f"Output for Premultiplication Signals\n"
                wave_file = None
                test = np.load(file_path)
                plot_dynamic_frequency_modes(
                    file_path,
                    time,
                    freq,
                    wideband_freq_modes,
                    freq_range,
                    signals_per_file,
                    time_signals,
                    wave_file,
                    base_title,
                    fft_shift_flag=True
                )  

    mlp_freq_modes = mlp.get_freq_modes()
    # selected_freq_modes = mlp_freq_modes[0:1]
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

            premultiply_h5_file= premultiply_dir / f"{Path(freq_signal_filename).stem}_{mode}.h5"
            output_h5_file = wideband_dir / f"wbf_{Path(freq_signal_filename).stem}_{mode}.h5"               
                    
            ml_model_file = ml_models_dir / f"{mode}_{ml_model_filename}"

            if not ml_config_file.exists():
                save_to_json(mlp.get_all_params(), ml_config_file)
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