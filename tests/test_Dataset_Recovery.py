'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import(
        load_config_from_json,
        get_logger,
        plot_dynamic_frequency_modes,
        REQUIRED_AXIS_KEYS
    )
    from pathlib import Path
    from spectramelt.Recovery import Recovery
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import logging

    load_dotenv()
    
    create_recovery_set = False
    use_mlp = False
    decode_recovery_to_time = False

    create_recovery_dataframe = True
    set_recovery_dataframe = True

    display_recovered_signals = False
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
    # selected_freq_modes = recovery_freq_modes[0:1]

    if create_recovery_set:
        if use_mlp:
            from spectramelt.mlp_module import MLP
            mlp = MLP(config_file_path=Path(getenv('MLP_CONF')))
            dataset.create_recovery_set(recovery, mlp=mlp)
        else:
            dataset.create_recovery_set(recovery)
            
    if decode_recovery_to_time:
        dataset.decode_time_signals()

    if create_recovery_dataframe:
        dataset.create_recovery_dataframe()

    if set_recovery_dataframe:
        dataset.set_recovery_dataframe(selected_freq_modes)

    if display_recovered_signals:
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        
        time_signal_filename = filenames.get('time_signals', "time_signals.npy")
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
        
        for file_path in recovery_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(freq_signal_filename):         
                # Extract identifying portion (for example, everything up to "signals.npy")
                stem = file_path.name
                key_part = stem.split(freq_signal_filename)[0]
                key = key_part.replace("centered_", "")

                wave_file = wideband_dir / f"{key}{wave_params_filename}"
                if not wave_file.exists():
                    logger.warning(f"Wave parameter file {wave_file} does not exist")
                    wave_file = None

                recovery_time_file = recovery_dir / f"{key_part}{time_signal_filename}"
                if not recovery_time_file.exists():
                    logger.warning(f"Recovered time file {recovery_time_file} does not exist")
                    recovery_time_file = None
                    
                base_title = f"Recovery for {DUT_type} Signals\n"

                    
                plot_dynamic_frequency_modes(
                    file_path,
                    time,
                    freq,
                    recovery_freq_modes,
                    freq_range,
                    signals_per_file,
                    recovery_time_file,
                    wave_file,
                    base_title,
                    fft_shift_flag=True
                )
                     
    atexit.register(logger.info, "Completed Test\n")