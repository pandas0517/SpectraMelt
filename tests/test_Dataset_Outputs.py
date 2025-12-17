'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import (
        load_config_from_json,
        get_logger,
        plot_dynamic_frequency_modes,
        REQUIRED_AXIS_KEYS
    )
    from pathlib import Path
    from spectramelt.NYFR import NYFR
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import logging
    import pickle

    load_dotenv()
    
    create_output_set = False
    create_premultiply_set = False

    create_wbf_wave_params = False
    create_nyfr_wave_params = False

    display_output_signals = False
    display_wbf_signals = False
    display_premultiply_signals = True
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr = NYFR(config_file_path=Path(getenv('NYFR_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr.get_config_name(),
                      config_file_path=Path(getenv('DATASET_CONF')))

    if create_output_set:
        dataset.create_output_set(nyfr)

    if create_premultiply_set:
        dataset.create_premultiply_set()
        
    if create_wbf_wave_params:
        dataset.create_wbf_wave_params()

    if create_nyfr_wave_params:
        dataset.create_nyfr_wave_params(nyfr)

    directories = dataset.get_directories()
    input_dir = directories.get('inputs', "Inputs")
    output_dir = directories.get('outputs', "Outputs")
    wideband_dir = directories.get('wideband', "Wideband")    
    
    filenames = dataset.get_filenames()
    input_time_signal_filename = filenames.get('time_signals', "time_signals.npy")
    input_wave_params_filename = filenames.get('wave_params', "wave_params.pkl")
    input_freq_signal_filename = filenames.get('freq_signals', "freq_signals.npz")

    freq_modes = dataset.get_freq_modes()
    
    output_freq_modes = freq_modes.get('output', [])
    wideband_freq_modes = freq_modes.get('wideband', [])
    
    input_wave_params = input_config.get('wave_params')
    freq_range = input_wave_params.get('freq_range')
    amp_range = input_wave_params.get('amp_range')

    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)
         
    if display_output_signals:
        DUT_type = type(nyfr).__name__
        samp_time_freq_filename = filenames.get('samp_time_freq', "sampled_time_freq.npz")
        time_freq_file = output_dir / samp_time_freq_filename
        with np.load(time_freq_file) as time_freq:
            missing = [k for k in REQUIRED_AXIS_KEYS if k not in time_freq]
            if missing:
                raise ValueError(f"{time_freq_file} missing required arrays: {missing}")
            time = time_freq["time"]
            freq = time_freq["freq"]
        
        output_freq_range = [freq[0], -freq[0]]
        signals_per_file = 3
        
        for file_path in output_dir.iterdir():
            if (file_path.is_file() and
                file_path.name.endswith(input_time_signal_filename) and
                "centered" in file_path.name):
                stem = str(file_path)
                stem = stem.replace("_centered", "")
                key_part = stem.split(input_time_signal_filename)[0]
                wave_file = Path(f"{key_part}{input_wave_params_filename}")
                freq_signal_file = Path(f"{key_part}{input_freq_signal_filename}")
                
                if not wave_file.exists():
                    logger.warning(f"Wave parameter file {wave_file} does not exist")
                    wave_file = None
                    
                if not freq_signal_file.exists():
                    logger.error(f"{DUT_type} frequency file {freq_signal_file} does not exist")
                    raise ValueError(f"{DUT_type} frequency file {freq_signal_file} does not exist")

                base_title = f"Output for DUT Type {DUT_type}\n"
                
                plot_dynamic_frequency_modes(
                    freq_signal_file,
                    time,
                    freq,
                    output_freq_modes,
                    output_freq_range,
                    signals_per_file,
                    file_path,
                    wave_file,
                    base_title,
                )
    
    if display_wbf_signals or display_premultiply_signals:
        wbf_time_freq_filename = filenames.get('wbf_time_freq', "wbf_time_freq.npz")
        time_freq_file = wideband_dir / wbf_time_freq_filename
        with np.load(time_freq_file) as time_freq:
            missing = [k for k in REQUIRED_AXIS_KEYS if k not in time_freq]
            if missing:
                raise ValueError(f"{time_freq_file} missing required arrays: {missing}")
            time = time_freq["time"]
            freq = time_freq["freq"]
            
    if display_wbf_signals:
        signals_per_file = 3
        
        for file_path in wideband_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                stem = str(file_path)
                key_part = stem.split(input_time_signal_filename)[0]
                wave_file = Path(f"{key_part}{input_wave_params_filename}")
                freq_signal_file = Path(f"{key_part}{input_freq_signal_filename}")
                
                if not wave_file.exists():
                    logger.warning(f"Wave parameter file {wave_file} does not exist")
                    wave_file = None
                    
                if not freq_signal_file.exists():
                    logger.error(f"Wideband frequency file {freq_signal_file} does not exist")
                    raise ValueError(f"Wideband frequency file {freq_signal_file} does not exist")

                base_title = f"Output for Wideband Frequency\n"
                
                plot_dynamic_frequency_modes(
                    freq_signal_file,
                    time,
                    freq,
                    wideband_freq_modes,
                    freq_range,
                    signals_per_file,
                    file_path,
                    wave_file,
                    base_title,
                    fft_shift=True
                )
                        
    if display_premultiply_signals:
        premultiply_dir = directories.get('premultiply', "Premultiply")
        N = len(freq)
        signals_per_file = 3
        
        for file_path in premultiply_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_freq_signal_filename):         
                freq_signals = np.load(file_path)

                time_signals = None
                base_title = f"Output for Premultiplication Signals\n"
                wave_file = None

                plot_dynamic_frequency_modes(
                    freq_signals,
                    time,
                    freq,
                    output_freq_modes,
                    freq_range,
                    signals_per_file,
                    time_signals,
                    wave_file,
                    base_title,
                    fft_shift=True
                )
            
    atexit.register(logger.info, "Completed Test\n")