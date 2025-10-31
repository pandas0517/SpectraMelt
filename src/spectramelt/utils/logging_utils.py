import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# ANSI escape colors for console logs
LOG_COLORS = {
    logging.DEBUG: "\033[37m",    # White
    logging.INFO: "\033[36m",     # Cyan
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",    # Red
    logging.CRITICAL: "\033[41m", # Red background
}
RESET_COLOR = "\033[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = LOG_COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{log_color}{message}{RESET_COLOR}"

def find_project_root(marker_files=("pyproject.toml", "setup.py", ".git")):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    return Path.cwd()  # fallback

# Base directory and default log file
# BASE_DIR = Path(__file__).resolve().parents[3]
BASE_DIR = find_project_root()
DEFAULT_LOG_FILE = BASE_DIR / "app.log"


def get_logger(
    name: str = "file_utils",
    log_file: Optional[Path | str] = None,
    level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """Returns a logger object. If log_file is None, uses the default log path.
    
    Parameters
    ----------
    name : str
        Logger name.
    log_file : Path | str | None
        Path to the log file. Defaults to DEFAULT_LOG_FILE.
    level : str
        Logging level as string: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    console : bool
        Whether to log to console as well.
    """
    if log_file is None:
        log_file = DEFAULT_LOG_FILE

    # Ensure uppercase and validate
    level = level.upper()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level not in valid_levels:
        raise ValueError(f"Invalid log level '{level}'. Must be one of {valid_levels}.")

    # Convert string to logging constant
    numeric_level = getattr(logging, level)
    
    setup_logger(log_file=str(log_file), level=numeric_level, console=console)
    return logging.getLogger(name)


def setup_logger(
    log_file="app.log",
    level=logging.INFO,
    max_bytes=5_000_000,
    backup_count=5,
    console=True  # <-- Add this parameter
):
    """
    Configure application logging with rotation, colors, and optional console output.

    Parameters
    ----------
    log_file : str
        Path to the output log file.
    level : int
        Logging level (default=logging.INFO)
    max_bytes : int
        Max size before rotating logs (default 5 MB)
    backup_count : int
        Number of old log files to keep (default 5)
    console : bool
        Whether to print logs to the console (default True)
    """

    # Clear old handlers if already configured
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Formatter with function name + line number
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Initialize handlers list with file handler
    handlers = [file_handler]

    # Optional console handler with color formatting
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        handlers.append(console_handler)

    # Apply handlers and settings
    logging.basicConfig(level=level, handlers=handlers)
