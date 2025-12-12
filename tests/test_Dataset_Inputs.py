'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import (
        get_logger,
        plot_dynamic_frequency_modes
    )
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import pickle
    import logging

    load_dotenv()

    create_set = False
    test_max_min = False
    update_input_waves = False
    display_signals = True

    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_signal = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    dataset = DataSet(input_config_name=input_signal.get_config_name(), config_file_path=Path(getenv('DATASET_CONF')))

    if create_set:
        dataset.create_input_set(input_signal)
        
    directories = dataset.get_directories()
    input_dir = directories.get('inputs', "Inputs")
    filenames = dataset.get_filenames()
    input_time_signal_filename = filenames.get('time_signals', "time_signals.npy")
    input_freq_signal_filename = filenames.get('freq_signals', "freq_signals.npz")
    
    if test_max_min:
        input_signal_params = input_signal.get_adc_params()
        v_ref_range = tuple(input_signal_params.get('v_ref_range', (0, 5)))
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                signals = np.load(file_path)
                for signal in signals:
                    if (signal > v_ref_range[1]).any() or (signal < v_ref_range[0]).any():
                        logger.info(f"Value found out of range {v_ref_range[0]} - {v_ref_range[1]} for input set {file_path.stem}")
                        break

    if update_input_waves:
        dataset.update_input_wave_params()

    if display_signals:
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        input_signal_wave_params = input_signal.get_wave_params()
        freq_range = input_signal_wave_params.get('freq_range')
        amp_range = input_signal_wave_params.get('amp_range')
        input_wave_params_filename = filenames.get('input.wave_params', "wave_params.pkl")
        
        freq_modes = dataset.get_freq_modes()
        input_freq_modes = freq_modes.get('input', [])
        
        real_time_freq_filename = filenames.get('real_time_freq', "real_time_freq.npz")
        real_time_freq = np.load(input_dir / real_time_freq_filename)
        real_time = real_time_freq["time"]
        real_freq = real_time_freq["freq"]

        signals_per_file = 3
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                stem = str(file_path)
                key_part = stem.split(input_time_signal_filename)[0]
                wave_params_file = Path(f"{key_part}{input_wave_params_filename}")
                freq_signal_file = Path(f"{key_part}{input_freq_signal_filename}")

                if not wave_params_file.exists():
                    logger.error(f"Wave parameter file {wave_params_file} does not exist")
                    raise ValueError(f"Wave parameter file {wave_params_file} does not exist")
                with open(wave_params_file, "rb") as f:
                    wave_params = pickle.load(f)

                if not freq_signal_file.exists():
                    logger.error(f"Input frequency file {freq_signal_file} does not exist")
                    raise ValueError(f"Input frequency file {freq_signal_file} does not exist")
                
                freq_signals = np.load(freq_signal_file)
                time_signals = np.load(file_path)
                base_title = f"Simulated Analog\n"

                plot_dynamic_frequency_modes(
                    wave_params,
                    freq_signals,
                    time_signals,
                    real_time,
                    real_freq,
                    input_freq_modes,
                    freq_range,
                    signals_per_file,
                    base_title,
                    file_path
                )
            
    atexit.register(logger.info, "Completed Test\n")