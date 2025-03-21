import json
import sys

def load_settings(file_path):
    print("Loading Settings from file: {}", file_path)
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