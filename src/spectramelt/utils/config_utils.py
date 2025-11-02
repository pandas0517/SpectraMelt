import json
from pathlib import Path
from .logging_utils import get_logger
from typing import Any, Dict, Union

def load_config_from_json(file_path, log_file=None, level="INFO", console=True):
    logger = get_logger("config_utils", log_file, level, console)
        
    file_path = Path(file_path)
    try:
        with file_path.open('r', encoding='utf-8') as file:
            system_config = json.load(file)
            return system_config
    except FileNotFoundError:
        if logger is None:
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        if logger is None:
            raise json.JSONDecodeError(f"Error decoding JSON in file: {file_path}")
        else:
            logger.error(f"Error decoding JSON in file: {file_path}")
    except IsADirectoryError:
        if logger is None:
            raise IsADirectoryError(f"Expected a file but found a directory: {file_path}")
        else:
            logger.error(f"Expected a file but found a directory: {file_path}")
            

def save_to_json(data: Dict[str, Any], filename: Union[str, Path], indent: int = 4):
    """
    Save a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        Dictionary to save.
    filename : str or Path
        Path to the output JSON file.
    indent : int
        Number of spaces for JSON indentation (default=4).
    """
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def create_input_set_json(file_path, log_file=None, level="INFO", console=True):
    logger = get_logger("config_utils", log_file, level, console)
    file_path = Path(file_path)
    input_set = {}
    noise_levels, phase_shifts, f_mods, f_deltas = [], [], [], []

    input_set["input_config_name"] = input("Enter input configuration name: ")
    input_set["tot_num_freq_combos"] = input("Enter total number of frequency combinations: ")
    max_input_tones = input("Enter maximum number of input tones: ")
    input_set["amp_min"] = input("Enter minimum amplitude: ")
    input_set["amp_max"] = input("Enter maximum amplitude: ")

    tone_list = []
    for input_tone in range(2, int(max_input_tones) + 1):
        tone_list.append(["1_2", [1, 2]] if input_tone == 2 else [str(input_tone), [input_tone]])

    for container, prompt in zip(
        [noise_levels, phase_shifts, f_mods, f_deltas],
        ["Enter noise level: ", "Enter phase shift: ", "Enter f_mod: ", "Enter f_delta: "]
    ):
        add_more = True
        while add_more:
            container.append(input(prompt))
            add_more = input("Add another? (y/n): ").lower() == 'y'

    input_set.update({
        'noise_levels': noise_levels,
        'phase_shifts': phase_shifts,
        'f_mods': f_mods,
        'f_deltas': f_deltas,
        'input_tones': tone_list
    })

    try:
        with file_path.open('w', encoding='utf-8') as file:
            json.dump(input_set, file, indent=4)
        logger.info(f"Input set saved to {file_path}")
    except Exception as e:
        logger.exception(f"Failed to save input set JSON to {file_path}: {e}")


def create_filename_json(file_path, log_file=None, level="INFO", console=True):
    logger = get_logger("config_utils", log_file, level, console)
    file_path = Path(file_path)
    filenames = {}

    try:
        # Prompt inputs
        input_df_filename = input("Enter input data frame file name: ")
        output_df_filename = input("Enter output data frame file name: ")
        recovery_df_filename = input("Enter recovery data frame file name: ")
        dictionary_base_name = input("Enter dictionary file base name: ")
        time_base_name = input("Enter time file base name: ")
        frequency_base_name = input("Enter frequency file base name: ")
        sampled_frequency_base_name = input("Enter sampled frequency file base name: ")
        recovery_base_name = input("Enter recovery file base name: ")
        mlp_model_file_name = input("Enter MLP model file name: ")
        mlp_log_file_name = input("Enter MLP log file name: ")

        # Recovery modes and processing systems
        recovery_modes, processing_systems, input_tones = [], [], {}
        for container, limit, prompt in zip(
            [recovery_modes, processing_systems],
            [3, None],
            ["Enter recovery mode (mag_ang, real_imag, complex, active_zone): ",
             "Enter processing system: "]
        ):
            add_more = True
            count = 0
            while add_more and (limit is None or count < limit):
                container.append(input(prompt))
                add_more = input("Add another? (y/n): ").lower() == 'y'
                count += 1

        # Input tones
        add_input = True
        while add_input:
            num_input_tones = input("Enter number of input tones (1_2/3/4/5): ")
            input_tones[num_input_tones] = f"{num_input_tones}_tone_sigs.npy"
            add_input = input("Add another input signal filename? (y/n): ").lower() == 'y'

        mlp_log_name, mlp_log_extension = Path(mlp_log_file_name).stem, Path(mlp_log_file_name).suffix
        mlp_models = {
            "name": mlp_model_file_name,
            "log": {processing_system: f"{mlp_log_name}_{processing_system}{mlp_log_extension}" for processing_system in processing_systems}
        }

        # Recovery files
        recovery_file = {"df": recovery_df_filename, "name": recovery_base_name}
        for recovery_mode in recovery_modes:
            recovery_file[recovery_mode] = {}
            sub_modes = []
            if recovery_mode == 'mag_ang':
                sub_modes = ['mag', 'ang']
            elif recovery_mode == 'real_imag':
                sub_modes = ['real', 'imag']

            if recovery_mode == "complex":
                for ps in processing_systems:
                    recovery_file[recovery_mode][ps] = f"recovery_list_{ps}_complex.txt"
            elif recovery_mode == "active_zones":
                for ps in processing_systems:
                    recovery_file[recovery_mode][ps] = f"recovery_list_{ps}_active_zones.txt"
            else:
                for sm in sub_modes:
                    for ps in processing_systems:
                        recovery_file[recovery_mode][sm] = {ps: f"recovery_list_{ps}_{sm}.txt"}

        filenames.update({
            'dictionary': {"name": dictionary_base_name},
            'time': {"name": time_base_name, "frequency": frequency_base_name, "sample_freq": sampled_frequency_base_name},
            'recovery': recovery_file,
            'input_df': input_df_filename,
            'output_df': output_df_filename,
            'mlp_models': mlp_models,
            'input_tones': input_tones
        })

        with file_path.open('w', encoding='utf-8') as file:
            json.dump(filenames, file, indent=4)
        logger.info(f"File names saved to {file_path}")

    except Exception as e:
        logger.exception(f"Failed to create filename JSON at {file_path}: {e}")

def create_directories_json(file_path, log_file=None, level="INFO", console=True):
    logger = get_logger("config_utils", log_file, level, console)
    directories = {}

    system_config_name = input("Enter system configuration name: ")
    input_config_name = input("Enter input configuration name: ")
    drive_letter = input("Enter drive letter (e.g. F:): ")
    if drive_letter:
        drive_letter += os.sep
    base_dir = input("Enter base test directory name: ")
    input_dir = input("Enter input directory name: ")
    output_dir = input("Enter output directory name: ")
    fft_dir = input("Enter frequency file base name: ")
    time_dir = input("Enter time directory: ")
    time_sampled_dir = input("Enter sampled time directory: ")
    df_dir = input("Enter data frame directory: ")
    active_zones_dir = input("Enter active zones directory: ")
    premultiply_dir = input("Enter pre-multiply directory name: ")
    mlp_model_dir = input("Enter MLP model directory name: ")
    recovery_dir = input("Enter recovery directory name: ")
    dictionary_dir = input("Enter dictionary directory name: ")

    # Dictionary versions
    dictionary_versions = []
    while len(dictionary_versions) < 2:
        dictionary_versions.append(input("Add dictionary version (enhanced/original): "))
        if len(dictionary_versions) < 2:
            add_more = input("Add another dictionary version? (y/n): ").lower() == 'y'
            if not add_more:
                break

    # Recovery types
    recovery_types = []
    while len(recovery_types) < 4:
        recovery_types.append(input("Add recovery type (OMP_Custom/OMP/MLP1/SPGL1): "))
        if len(recovery_types) < 4:
            add_more = input("Add another recovery type? (y/n): ").lower() == 'y'
            if not add_more:
                break

    mlp_model_types = ["real", "imag", "mag", "ang", "complex"]

    # Base directories
    base_path = Path(drive_letter) / base_dir / system_config_name / input_config_name

    directories['input'] = base_path / input_dir
    directories['output'] = base_path / output_dir
    directories['fft'] = base_path / fft_dir
    directories['time'] = base_path / time_dir
    directories['time_sampled'] = base_path / time_sampled_dir
    directories['df'] = base_path / df_dir
    directories['active_zones'] = base_path / active_zones_dir
    directories['premultiply'] = base_path / premultiply_dir
    directories['mlp_models'] = base_path / mlp_model_dir
    directories['dictionary'] = base_path / dictionary_dir
    directories['recovery'] = base_path / recovery_dir
    directories['system_config_name'] = system_config_name

    try:
        with open(file_path, 'w') as file:
            json.dump(directories, file, indent=4)
        logger.info(f"Directory names saved to {file_path}")
    except Exception as e:
        logger.exception(f"Failed to save directory JSON to {file_path}: {e}")


def create_system_json(file_path, log_file=None, level="INFO", console=True):
    logger = get_logger("config_utils", log_file, level, console)
    system_config = {
        "system_params": {},
        "filter_params": {},
        "time_params": {},
        "LO_params": {},
        "dictionary_params": {},
        "recovery_params": {}
    }

    system_config["system_config_name"] = input("Enter system configuration name: ")
    system_config["system_params"]["wbf_cut_mod"] = input("Enter wideband filter cutoff frequency modifier (unused currently): ")
    system_config["system_params"]["wbf_cut_freq"] = input("Enter wideband filter cutoff frequency: ")
    system_config["system_params"]["adc_clock_freq"] = input("Enter ADC clock frequency: ")

    # Processing systems
    processing_systems = []
    add_processing_system = True
    while add_processing_system:
        processing_systems.append(input("Enter processing system: "))
        add_processing_system = input("Add another processing system? (y/n): ").lower() == 'y'
    system_config["system_params"]["processing_systems"] = processing_systems
    system_config["system_params"]["system_noise_level"] = input("Enter system noise level: ")

    # Filter params
    system_config["filter_params"]["type"] = input("Enter filter type: ")
    system_config["filter_params"]["order"] = input("Enter filter order: ")
    system_config["filter_params"]["cutoff_freq"] = input("Enter filter cutoff frequency: ")
    system_config["filter_params"]["angle"] = input("Enter filter angle: ")
    system_config["filter_params"]["window_size"] = input("Enter filter window size: ")

    # Time params
    system_config["time_params"]["start"] = input("Enter start time: ")
    system_config["time_params"]["stop"] = input("Enter stop time: ")
    system_config["time_params"]["sim_freq"] = input("Enter simulation frequency: ")
    system_config["time_params"]["save_real_time"] = input("Save real time? (y/n): ").lower() == 'y'

    # LO params
    system_config["LO_params"]["amp"] = input("Enter LO amplitude: ")
    system_config["LO_params"]["freq"] = input("Enter LO frequency: ")
    system_config["LO_params"]["phase"] = input("Enter LO phase: ")
    system_config["LO_params"]["phase_freq"] = input("Enter LO phase modulation frequency: ")
    system_config["LO_params"]["phase_delta"] = input("Enter LO phase modulation delta: ")
    system_config["LO_params"]["phase_offset"] = input("Enter LO phase modulation offset: ")

    # Dictionary and recovery params
    system_config["dictionary_params"]["type"] = input("Enter dictionary type: ")
    system_config["dictionary_params"]["version"] = input("Enter dictionary version: ")
    system_config["recovery_params"]["type"] = input("Enter recovery type: ")

    recovery_modes = []
    add_recovery_mode = True
    while add_recovery_mode:
        recovery_modes.append(input("Enter recovery mode (mag_ang, real_imag, complex, active_zone): "))
        add_recovery_mode = input("Add another recovery mode? (y/n): ").lower() == 'y'
    system_config["recovery_params"]["modes"] = recovery_modes

    try:
        with open(file_path, 'w') as file:
            json.dump(system_config, file, indent=4)
        logger.info(f"System configuration saved to {file_path}")
    except Exception as e:
        logger.exception(f"Failed to save system JSON to {file_path}: {e}")


def create_training_json(file_path, log_file=None, level="INFO", console=True):
    logger = get_logger("config_utils", log_file, level, console)
    training_config = {}
    training_config["processing_system"] = input("Enter processing system: ")
    training_config["total_num_sigs"] = input("Enter total number of signals: ")
    training_config["num_epochs"] = input("Enter number of epochs: ")
    training_config["batch_sz"] = input("Enter batch size: ")
    training_config["learning_rate"] = input("Enter learning rate: ")
    training_config["train_test_split_percentage"] = input("Enter train/test split percentage in decimal format: ")
    training_config["loss_type"] = input("Enter loss function: ")
    training_config["pre_multiply"] = input("Enter pre-multiply factor: ")
    training_config["active_zones_min_mag"] = input("Enter the minimum magnitude within active zones: ")
    training_config["save_fft_file"] = input("Save FFT file? (y/n): ").lower() == 'y'
    training_config["use_fft"] = input("Use FFT file? (y/n): ").lower() == 'y'
    training_config["save_active_zones_file"] = input("Save active zones file? (y/n): ").lower() == 'y'
    training_config["use_active_zones"] = input("Use active zones? (y/n): ").lower() == 'y'
    training_config["save_premultiply"] = input("Save pre-multiply file? (y/n): ").lower() == 'y'
    training_config["pre_omp"] = input("Use pre-OMP? (y/n): ").lower() == 'y'

    recovery_modes = []
    add_recovery_mode = True
    while add_recovery_mode:
        recovery_modes.append(input("Enter recovery mode (mag_ang, real_imag, complex, active_zone): "))
        add_recovery_mode = input("Add another recovery mode? (y/n): ").lower() == 'y'
    training_config["modes"] = recovery_modes

    training_config["early_stopping"] = {}
    training_config["early_stopping"]["monitor"] = input("Enter early stopping monitor metric: ")
    training_config["early_stopping"]["min_delta"] = input("Enter early stopping minimum delta: ")
    training_config["early_stopping"]["patience"] = input("Enter early stopping patience: ")
    training_config["early_stopping"]["verbose"] = input("Enter early stopping verbosity (0 or 1): ")
    training_config["early_stopping"]["start_from_epoch"] = input("Enter early stopping start from epoch: ")
    training_config["early_stopping"]["restore_best_weights"] = input("Restore best weights? (y/n): ").lower() == 'y'

    try:
        with open(file_path, 'w') as file:
            json.dump(training_config, file, indent=4)
        logger.info(f"Training configuration saved to {file_path}")
    except Exception as e:
        logger.exception(f"Failed to save training JSON to {file_path}: {e}")


def create_wave_json(file_path, log_file=None, level="INFO", console=True):
    logger = get_logger("config_utils", log_file, level, console)
    wave_config = {}
    wave_params = []

    add_wave_param = True
    while add_wave_param:
        wave_param = {}
        wave_param["amp"] = input("Enter wave amplitude: ")
        wave_param["freq"] = input("Enter wave frequency: ")
        wave_param["phase"] = input("Enter wave phase: ")
        wave_params.append(wave_param)
        add_wave_param = input("Add another wave parameter? (y/n): ").lower() == 'y'

    wave_config["wave_params"] = wave_params

    try:
        with open(file_path, 'w') as file:
            json.dump(wave_config, file, indent=4)
        logger.info(f"Wave configuration saved to {file_path}")
    except Exception as e:
        logger.exception(f"Failed to save wave JSON to {file_path}: {e}")