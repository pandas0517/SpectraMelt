import json
import sys
import os
from pathlib import Path

def load_settings(file_path):
    try:
        with open(file_path, 'r') as file:
            system_config = json.load(file)
            return system_config
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")
        sys.exit(1)
    except IsADirectoryError:
        print("Invalid directory. Please check the file path.")
        sys.exit(1)

def get_all_file_names(directory):
    file_names = []
    for _, _, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    return file_names

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def get_all_sub_dirs(directory):
    deepest_sub_dirs = []
    for root, dirnames, _ in os.walk(directory):
        if not dirnames:
            deepest_sub_dirs.append(root)
    return deepest_sub_dirs

def get_file_sub_dirs(input_file_path):
    file_path = Path(input_file_path)
    file_path_len = len(file_path.parts)
    file_name = file_path.parts[file_path_len - 1]
    phase_delta = file_path.parts[file_path_len - 2]
    noise_mod = file_path.parts[file_path_len - 3]
    return [noise_mod, phase_delta, file_name]

def delete_lines_with_string(file_path, target_string):
    try:
        # Read the file and store lines that do not contain the target string
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Write back only the lines that do not contain the target string
        with open(file_path, 'w') as file:
            for line in lines:
                if target_string not in line:
                    file.write(line)
        print(f"Lines containing '{target_string}' have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_input_set_json(file_path):
    input_set = {}
    noise_levels = []
    phase_shifts = []
    f_mods = []
    f_deltas = []
    add_noise_level = True
    add_phase_shift = True
    add_f_mod = True
    add_f_delta = True
    input_set["input_config_name"] = input("Enter input configuration name: ")
    input_set["tot_num_freq_combos"] = input("Enter total number of frequency combinations: ")
    max_input_tones = input("Enter maximum number of input tones: ")
    input_set["amp_min"] = input("Enter minimum amplitude: ")
    input_set["amp_max"] = input("Enter maximum amplitude: ")
    tone_list = []
    for input_tone in range(2, int(max_input_tones) + 1):
        if input_tone == 2:
            tone_list.append(["1_2", [1,2]])
        else:
            tone_list.append([str(input_tone), [input_tone]])
    while add_noise_level:
        noise_levels.append(input("Enter noise level: "))
        add_noise_level = input("Add another noise level? (y/n): ").lower() == 'y'

    while add_phase_shift:
        phase_shifts.append(input("Enter phase shift: "))
        add_phase_shift = input("Add another phase shift? (y/n): ").lower() == 'y'

    while add_f_mod:
        f_mods.append(input("Enter f_mod: "))
        add_f_mod = input("Add another f_mod? (y/n): ").lower() == 'y'

    while add_f_delta:
        f_deltas.append(input("Enter f_delta: "))
        add_f_delta = input("Add another f_delta? (y/n): ").lower() == 'y'

    input_set['noise_levels'] = noise_levels
    input_set['phase_shifts'] = phase_shifts
    input_set['f_mods'] = f_mods
    input_set['f_deltas'] = f_deltas
    input_set['input_tones'] = tone_list

    with open(file_path, 'w') as file:
        json.dump(input_set, file, indent=4)
    print(f"Input set saved to {file_path}")

def create_filename_json(file_path):
    filenames = {}
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

    add_recovery_mode = True
    recovery_modes = []
    total_recovery_modes = 0
    while add_recovery_mode and total_recovery_modes < 3:
        recovery_modes.append(input("Enter recovery mode (mag_ang, real_imag, complex, active_zone): "))
        add_recovery_mode = input("Add another recovery mode? (y/n): ").lower() == 'y'
        total_recovery_modes += 1

    add_processing_systems = True
    processing_systems = []
    while add_processing_systems:
        processing_systems.append(input("Enter processing systems: "))
        add_processing_systems = input("Add another processing system? (y/n): ").lower() == 'y'

    add_input_tone_file_name = True
    input_tones = {}
    while add_input_tone_file_name:
        num_input_tones = input("Enter number of input tones (1_2/3/4/5): ")
        input_tones[num_input_tones] = num_input_tones + "_tone_sigs.npy"
        add_input_tone_file_name = input("Add another input signal filename? (y/n): ").lower() == 'y'

    mlp_log_name = os.path.splitext(mlp_log_file_name)[0]
    mlp_log_extension = os.path.splitext(mlp_log_file_name)[1]
    mlp_models = {
        "name": mlp_model_file_name,
        "log": {
            "name": mlp_log_file_name
        }
    }
    for processing_system in processing_systems:
        mlp_models["log"][processing_system] = mlp_log_name + "_" + processing_system + mlp_log_extension

    recovery_file = {
        "df": recovery_df_filename,
        "name": recovery_base_name
    }
    for recovery_mode in recovery_modes:
        recovery_file[recovery_mode] = {}
        sub_modes = []
        if recovery_mode == 'mag_ang':
            sub_modes = [ 'mag', 'ang' ]
        elif recovery_mode == 'real_imag':
            sub_modes = [ 'real', 'imag' ]

        if recovery_mode == "complex":
            for processing_system in processing_systems:
                recovery_file[recovery_mode][processing_system] = "recovery_list_" + processing_system + "_complex.txt"
        elif recovery_mode == "active_zones":
            for processing_system in processing_systems:
                recovery_file[recovery_mode][processing_system] = "recovery_list_" + processing_system + "_active_zones.txt"
        else:
            for sub_mode in sub_modes:
                for processing_system in processing_systems:
                    recovery_file[recovery_mode][sub_mode][processing_system] = "recovery_list_" + processing_system + "_" + sub_mode + ".txt"
        
        filenames['dictionary'] = {
            "name": dictionary_base_name
        }
        filenames['time'] = {
            "name": time_base_name,
            "frequency": frequency_base_name,
            "sample_freq": sampled_frequency_base_name
        }
        filenames['recovery'] = recovery_file           
        filenames['input_df'] = input_df_filename
        filenames['output_df'] = output_df_filename
        filenames['mlp_models'] = mlp_models
        filenames['input_tones'] = input_tones

    with open(file_path, 'w') as file:
        json.dump(filenames, file, indent=4)
    print(f"File names saved to {file_path}")

def create_directories_json(file_path):
    directories = {}
    system_config_name = input("Enter system configuration name: ")
    input_config_name = input("Enter input configuration name: ")
    drive_letter = input("Enter drive letter (e.g. F:): ")
    if drive_letter is not None:
        drive_letter = drive_letter + os.sep
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
    dictionary_versions = []
    total_dictionary_versions = 0
    add_dictionary_version = True
    while add_dictionary_version and total_dictionary_versions < 2:
        dictionary_versions.append(input("Add dictionary version (enhanced/original): "))
        add_dictionary_version = input("Add another dictionary version? (y/n): ").lower() == 'y'
        total_recovery_modes += 1
    
    recovery_types = []
    total_recovery_types = 0
    add_recovery_types = True
    while add_recovery_types and total_recovery_types < 4:
        recovery_types.append(input("Add recovery type (OMP_Custom/OMP/MLP1/SPGL1): "))
        add_recovery_types = input("Add another dictionary version? (y/n): ").lower() == 'y'
        total_recovery_types += 1

    mlp_model_types = [ "real", "imag", "mag", "ang", "complex"]
    directories['system_config_name'] = system_config_name
    directories['input'] = os.path.join(drive_letter,
                                    base_dir,
                                    system_config_name,
                                    input_config_name,
                                    input_dir)
    directories['output'] = os.path.join(drive_letter,
                                    base_dir,
                                    system_config_name,
                                    input_config_name,
                                    output_dir)
    directories['fft'] = os.path.join(drive_letter,
                                base_dir,
                                system_config_name,
                                input_config_name,
                                fft_dir)
    directories['time'] = os.path.join(drive_letter,
                                base_dir,
                                system_config_name,
                                input_config_name,
                                time_dir)
    directories['time_sampled'] = os.path.join(drive_letter,
                                            base_dir,
                                            system_config_name,
                                            input_config_name,
                                            time_sampled_dir)
    directories['df'] = os.path.join(drive_letter,
                                base_dir,
                                system_config_name,
                                input_config_name,
                                df_dir)
    directories['active_zones'] = os.path.join(drive_letter,
                                            base_dir,
                                            system_config_name,
                                            input_config_name,
                                            active_zones_dir)
    recovery = {}
    dictionary = {}
    mlp_models = {}
    premultiply = {}
    for dictionary_version in dictionary_versions:
        premultiply[dictionary_version] = os.path.join(drive_letter,
                                                        base_dir,
                                                        system_config_name,
                                                        input_config_name,
                                                        premultiply_dir,
                                                        dictionary_version)
        dictionary[dictionary_version] = os.path.join(drive_letter,
                                                        base_dir,
                                                        input_config_name,
                                                        dictionary_dir,
                                                        dictionary_version)
        for mlp_model_type in mlp_model_types:
            mlp_models[dictionary_version][mlp_model_type] = os.path.join(drive_letter,
                                                                            base_dir,
                                                                            system_config_name,
                                                                            input_config_name,
                                                                            mlp_model_dir,
                                                                            dictionary_version,
                                                                            mlp_model_type)
        for recovery_type in recovery_types:
            recovery[dictionary_version][recovery_type] = os.path.join(drive_letter,
                                                                        base_dir,
                                                                        system_config_name,
                                                                        input_config_name,
                                                                        recovery_dir,
                                                                        dictionary_version,
                                                                        recovery_type)
    directories['mlp_models'] = os.path.join(drive_letter,
                                        base_dir,
                                        system_config_name,
                                        input_config_name,
                                        mlp_model_dir)
    directories['dictionary'] = os.path.join(drive_letter,
                                        base_dir,
                                        system_config_name,
                                        input_config_name,
                                        dictionary_dir)
    directories['recovery'] = os.path.join(drive_letter,
                                        base_dir,
                                        system_config_name,
                                        input_config_name,
                                        recovery_dir)
    directories['premultiply'] = os.path.join(drive_letter,
                                        base_dir,
                                        system_config_name,
                                        input_config_name,
                                        premultiply_dir)

    with open(file_path, 'w') as file:
        json.dump(directories, file, indent=4)
    print(f"Directory names saved to {file_path}")

def create_system_json(file_path):
    system_config = {}
    system_config["system_config_name"] = input("Enter system configuration name: ")
    system_config["system_params"]["wbf_cut_mod"] = input("Enter wideband filter cuttoff frequency modifier(unused currently): ")
    system_config["system_params"]["wbf_cut_freq"] = input("Enter wideband filter cuttoff frequency: ")
    system_config["system_params"]["adc_clock_freq"] = input("Enter ADC clock frequency: ")
    add_processing_system = True
    processing_systems = []
    while add_processing_system:
        processing_systems.append(input("Enter processing system: "))
        add_processing_system = input("Add another processing system? (y/n): ").lower() == 'y'
    system_config["system_params"]["processing_systems"] = processing_systems
    system_config["system_params"]["system_noise_level"] = input("Enter system noise level: ")
    system_config["filter_params"]["type"] = input("Enter filter type: ")
    system_config["filter_params"]["order"] = input("Enter filter order: ")
    system_config["filter_params"]["cutoff_freq"] = input("Enter filter cutoff frequency: ")
    system_config["filter_params"]["angle"] = input("Enter filter angle: ")
    system_config["filter_params"]["window_size"] = input("Enter filter window size: ")
    system_config["time_params"]["start"] = input("Enter start time: ")
    system_config["time_params"]["stop"] = input("Enter stop time: ")
    system_config["time_params"]["sim_freq"] = input("Enter simulation frequency: ")
    system_config["time_params"]["save_real_time"] = input("Save real time? (y/n): ").lower() == 'y'
    system_config["LO_params"]["amp"] = input("Enter LO amplitude: ")
    system_config["LO_params"]["freq"] = input("Enter LO frequency: ")
    system_config["LO_params"]["phase"] = input("Enter LO phase: ")
    system_config["LO_params"]["phase_freq"] = input("Enter LO phase modulation frequency: ")
    system_config["LO_params"]["phase_delta"] = input("Enter LO phase modulation delta: ")
    system_config["LO_params"]["phase_offset"] = input("Enter LO phase modulation offset: ")
    system_config["dictionary_params"]["type"] = input("Enter dictionary type: ")
    system_config["dictionary_params"]["version"] = input("Enter dictionary version: ")
    system_config["recovery_params"]["type"] = input("Enter recovery type: ")
    add_recovery_mode = True
    recovery_modes = []
    while add_recovery_mode:
        recovery_modes.append(input("Enter recovery mode (mag_ang, real_imag, complex, active_zone): "))
        add_recovery_mode = input("Add another recovery mode? (y/n): ").lower() == 'y'
    system_config["recovery_params"]["modes"] = recovery_modes
    with open(file_path, 'w') as file:
        json.dump(system_config, file, indent=4)
    print(f"System configuration saved to {file_path}")

def create_training_json(file_path):
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
    while add_recovery_mode:
        recovery_modes.append(input("Enter recovery mode (mag_ang, real_imag, complex, active_zone): "))
        add_recovery_mode = input("Add another recovery mode? (y/n): ").lower() == 'y'
    training_config["modes"] = recovery_modes

    training_config["early_stopping"]["monitor"] = input("Enter early stopping monitor metric: ")
    training_config["early_stopping"]["min_delta"] = input("Enter early stopping minimum delta: ")
    training_config["early_stopping"]["patience"] = input("Enter early stopping patience: ")
    training_config["early_stopping"]["verbose"] = input("Enter early stopping verbosity (0 or 1): ")
    training_config["early_stopping"]["start_from_epoch"] = input("Enter early stopping start from epoch: ")
    training_config["early_stopping"]["restore_best_weights"] = input("Restore best weights? (y/n): ").lower() == 'y'
    with open(file_path, 'w') as file:
        json.dump(training_config, file, indent=4)
    print(f"Training configuration saved to {file_path}")

def create_wave_json(file_path):
    wave_config = {}
    add_wave_param = True
    wave_params = []
    while add_wave_param:
        wave_param = {}
        wave_param["amp"] = input("Enter wave amplitude: ")
        wave_param["freq"] = input("Enter wave frequency: ")
        wave_param["phase"] = input("Enter wave phase: ")
        wave_params.append(wave_param)
        add_wave_param = input("Add another wave parameter? (y/n): ").lower() == 'y'
    wave_config["wave_params"] = wave_params
    with open(file_path, 'w') as file:
        json.dump(wave_config, file, indent=4)
    print(f"Wave configuration saved to {file_path}")