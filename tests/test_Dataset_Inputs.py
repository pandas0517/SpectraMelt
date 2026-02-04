'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import (
        get_logger,
        plot_dynamic_frequency_modes,
        REQUIRED_AXIS_KEYS
    )
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.Analog import Analog
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import logging

    load_dotenv()

    create_set = False
    test_max_min = False
    update_input_waves = False
    display_signals = True
    use_dB = True

    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_signal = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    input_wave_params = input_signal.get_wave_params()
    dataset = DataSet(input_config_name=input_signal.get_config_name(), config_file_path=Path(getenv('DATASET_CONF')))
    analog = Analog(config_file_path=Path(getenv('INPUT_CONF')))

    if create_set:
        dataset.create_input_set(analog, input_signal)
        
    directories = dataset.get_directories()
    input_dir: Path = directories.get('inputs', "Inputs")
    filenames = dataset.get_filenames()
    input_time_signal_filename = filenames.get('time_signals', "time_signals.npy")
    input_freq_signal_filename = filenames.get('freq_signals', "freq_signals.npz")
    
    if test_max_min:
        v_ref_range = tuple(input_wave_params.get('v_ref_range', (0, 5)))
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
        
        freq_modes = input_signal.get_freq_modes()
        input_freq_modes = freq_modes.get('input', [])
        
        time_freq_filename = filenames.get('real_time_freq', "real_time_freq.npz")
        time_freq_file = input_dir / time_freq_filename

        with np.load(time_freq_file) as time_freq:
            missing = [k for k in REQUIRED_AXIS_KEYS if k not in time_freq]
            if missing:
                raise ValueError(f"{time_freq_file} missing required arrays: {missing}")
            time = time_freq["time"]
            freq = time_freq["freq"]

        signals_per_file = 3
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(input_time_signal_filename):
                stem = str(file_path)
                key_part = stem.split(input_time_signal_filename)[0]
                wave_params_file = Path(f"{key_part}{input_wave_params_filename}")
                freq_signal_file = Path(f"{key_part}{input_freq_signal_filename}")

                if not wave_params_file.exists():
                    logger.warning(f"Wave parameter file {wave_params_file} does not exist")
                    wave_params_file = None

                if not freq_signal_file.exists():
                    logger.error(f"Input frequency file {freq_signal_file} does not exist")
                    raise ValueError(f"Input frequency file {freq_signal_file} does not exist")

                base_title = f"Simulated Analog\n"

                plot_dynamic_frequency_modes(
                    freq_signal_file,
                    time,
                    freq,
                    input_freq_modes,
                    freq_range,
                    signals_per_file,
                    file_path,
                    wave_params_file,
                    base_title,
                    decibels=use_dB
                )
            
    atexit.register(logger.info, "Completed Test\n")