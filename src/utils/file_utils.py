from pathlib import Path
from logging_utils import get_logger


def replace_file(old_filepath, new_filepath, log_file=None, level="DEBUG", console=True):
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


def replace_extension(file_path, new_extension, log_file=None, level="DEBUG", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    new_path = Path(file_path).with_suffix(f".{new_extension}")
    logger.info(f"Replaced extension: {file_path} -> {new_path}")
    return str(new_path)


def get_all_file_names(directory, log_file=None, level="DEBUG", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    directory = Path(directory)
    files = [f.name for f in directory.rglob("*") if f.is_file()]
    logger.info(f"Found {len(files)} files in {directory}")
    return files


def get_all_file_paths(directory, log_file=None, level="DEBUG", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    directory = Path(directory)
    files = [str(f) for f in directory.rglob("*") if f.is_file()]
    logger.info(f"Found {len(files)} file paths in {directory}")
    return files


def get_all_sub_dirs(directory, log_file=None, level="DEBUG", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    directory = Path(directory)
    sub_dirs = [str(d) for d in directory.rglob("*") if d.is_dir() and not any(d.iterdir())]
    logger.info(f"Found {len(sub_dirs)} deepest subdirectories in {directory}")
    return sub_dirs


def get_file_sub_dirs(input_file_path, log_file=None, level="DEBUG", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    path = Path(input_file_path)
    parts = path.parts
    if len(parts) < 3:
        logger.error(f"Path '{input_file_path}' does not have enough parts")
        raise ValueError(f"Path '{input_file_path}' does not have enough parts")
    result = [parts[-3], parts[-2], parts[-1]]
    logger.info(f"Extracted subdirectories from '{input_file_path}': {result}")
    return result


def delete_lines_with_string(file_path, target_string, log_file=None, level="DEBUG", console=True):
    logger = get_logger("file_utils", log_file, level, console)

    file_path = Path(file_path)
    try:
        lines = file_path.read_text().splitlines()
        filtered_lines = [line for line in lines if target_string not in line]
        file_path.write_text("\n".join(filtered_lines) + "\n")
        logger.info(f"Lines containing '{target_string}' have been deleted in {file_path}")
    except Exception as e:
        logger.exception(f"An error occurred while deleting lines in {file_path}: {e}")