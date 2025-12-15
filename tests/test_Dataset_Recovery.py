'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import(
        load_config_from_json,
        get_logger,
        plot_dynamic_frequency_modes
    )
    from pathlib import Path
    from spectramelt.Recovery import Recovery
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import logging
    import pickle

    load_dotenv()
    
    create_set = True
    use_mlp = True

    create_recovery_dataframe = False
    set_recovery_dataframe = False

    display_recovered_signals = True
    DUT_type = "NYFR"
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr_config = load_config_from_json(Path(getenv('NYFR_CONF')))
    recovery = Recovery(config_file_path=Path(getenv('RECOVERY_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr_config.get('config_name'),
                      recovery_config_name=recovery.get_config_name(),
                      config_file_path=Path(getenv('DATASET_CONF')))
    directories = dataset.get_directories()

    recovery_dir = directories.get('recovery', "Recovery")
    wideband_dir = directories.get('wideband', "Wideband")
    
    filenames = dataset.get_filenames()
    wave_params_filename = filenames.get('wave_params', "wave_params.pkl")
    freq_signal_filename = filenames.get('freq_signals', "freq_signals.npz")
    
    freq_modes = dataset.get_freq_modes()
    input_wave_params = input_config.get('wave_params')
    freq_range = input_wave_params.get('freq_range')
    amp_range = input_wave_params.get('amp_range')
    
    recovery_freq_modes = freq_modes.get('recovery', [])
    selected_freq_modes = recovery_freq_modes
    # selected_freq_modes = recovery_freq_modes[1:2]

    if create_set:
        if use_mlp:
            from spectramelt.MLP import MLP
            mlp = MLP(config_file_path=Path(getenv('MLP_CONF')))
            dataset.create_recovery_set(recovery, mlp=mlp)
        else:
            dataset.create_recovery_set(recovery)

    if create_recovery_dataframe:
        dataset.create_recovery_dataframe()

    if set_recovery_dataframe:
        dataset.set_recovery_dataframe(selected_freq_modes)

    if display_recovered_signals:
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        
        wbf_time_freq_filename = filenames.get('wbf_time_freq', "wbf_time_freq.npz")
        time_freq = np.load(wideband_dir / wbf_time_freq_filename)
        time = time_freq["time"]
        freq = time_freq["freq"]
        
        N = len(freq)
        signals_per_file = 3
        
        for file_path in recovery_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(freq_signal_filename):         
                freq_signals = np.load(file_path)

                time_signals = None
                base_title = f"Recovery for {DUT_type} Signals\n"
                
                # Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(freq_signal_filename)[0]
                wave_file = wideband_dir / f"{key_part}{wave_params_filename}"
                
                if not wave_file.exists():
                    logger.error(f"Wave parameter file {wave_file} does not exist")
                    raise ValueError(f"Wave parameter file {wave_file} does not exist")
                with open(wave_file, "rb") as f:
                    wave_params = pickle.load(f)
                    
                plot_dynamic_frequency_modes(
                    freq_signals,
                    time_signals,
                    time,
                    freq,
                    recovery_freq_modes,
                    freq_range,
                    signals_per_file,
                    wave_params,
                    base_title,
                    file_path,
                    fft_shift=True
                )
                     
    atexit.register(logger.info, "Completed Test\n")