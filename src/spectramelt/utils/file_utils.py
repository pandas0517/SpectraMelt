from pathlib import Path
from .logging_utils import get_logger
import gc
import numpy as np


def update_npz(path, **arrays):
    """
    Load an existing .npz, update/add arrays, and resave safely.

    Parameters
    ----------
    path : str or Path
        Path to .npz file
    arrays : dict
        Named numpy arrays to add or overwrite
    """
    path = Path(path)

    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            data_dict = dict(data)
    else:
        data_dict = {}

    data_dict.update(arrays)
    np.savez(path, **data_dict)

    # aggressive cleanup (important for long runs)
    del data_dict
    gc.collect()


def replace_file(old_filepath, new_filepath, log_file=None, level="INFO", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    old_path = Path(old_filepath)
    new_path = Path(new_filepath)

    try:
        if old_path.is_file():
            old_path.unlink()
            logger.info(f"Deleted old file: {old_path}")
        old_path.write_bytes(new_path.read_bytes())
        logger.info(f"Replaced {old_path} with {new_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {old_path} or {new_path}")
    except Exception as e:
        logger.exception(f"An error occurred while replacing file: {e}")


def replace_extension(file_path, new_extension, log_file=None, level="INFO", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    new_path = Path(file_path).with_suffix(f".{new_extension}")
    logger.info(f"Replaced extension: {file_path} -> {new_path}")
    return str(new_path)


def get_all_file_names(directory, log_file=None, level="INFO", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    directory = Path(directory)
    files = [f.name for f in directory.rglob("*") if f.is_file()]
    logger.info(f"Found {len(files)} files in {directory}")
    return files


def get_all_file_paths(directory, log_file=None, level="INFO", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    directory = Path(directory)
    files = [str(f) for f in directory.rglob("*") if f.is_file()]
    logger.info(f"Found {len(files)} file paths in {directory}")
    return files


def get_all_sub_dirs(directory, log_file=None, level="INFO", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    directory = Path(directory)
    sub_dirs = [str(d) for d in directory.rglob("*") if d.is_dir() and not any(d.iterdir())]
    logger.info(f"Found {len(sub_dirs)} deepest subdirectories in {directory}")
    return sub_dirs


def get_file_sub_dirs(input_file_path, log_file=None, level="INFO", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    path = Path(input_file_path)
    parts = path.parts
    if len(parts) < 3:
        logger.error(f"Path '{input_file_path}' does not have enough parts")
        raise ValueError(f"Path '{input_file_path}' does not have enough parts")
    result = [parts[-3], parts[-2], parts[-1]]
    logger.info(f"Extracted subdirectories from '{input_file_path}': {result}")
    return result


def delete_lines_with_string(file_path, target_string, log_file=None, level="INFO", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    file_path = Path(file_path)
    try:
        lines = file_path.read_text().splitlines()
        filtered_lines = [line for line in lines if target_string not in line]
        file_path.write_text("\n".join(filtered_lines) + "\n")
        logger.info(f"Lines containing '{target_string}' have been deleted in {file_path}")
    except Exception as e:
        logger.exception(f"An error occurred while deleting lines in {file_path}: {e}")
        

def build_flat_paths(directories):
    """
    Build full paths from nested 'base' and 'tail' dicts and return
    a flat dictionary where keys are dot-separated path strings and
    values are Path objects.
    """
    dataset_dir = directories["dataset_dir"]

    flat_paths = {}

    def recursive_build(keys, base_dict, tail_dict, parent_key=""):
        for k in base_dict:
            new_parent_key = f"{parent_key}.{k}" if parent_key else k
            base_value = base_dict[k]
            tail_value = (
                Path(*tail_dict[k]) if isinstance(tail_dict.get(k), list)
                else Path(tail_dict[k]) if isinstance(tail_dict.get(k), str)
                else None
            ) if tail_dict else None

            if isinstance(base_value, dict):
                recursive_build(keys + [k], base_value, tail_value, new_parent_key)
            else:
                # Build full path
                full_path = Path(base_value) / dataset_dir / (tail_value if tail_value else "")
                flat_paths[new_parent_key] = full_path

    recursive_build([], directories["base"], directories["tail"])
    return flat_paths


def flatten_files(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_files(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items