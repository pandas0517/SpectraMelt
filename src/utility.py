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