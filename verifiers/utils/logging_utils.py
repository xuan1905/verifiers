import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Setup basic logging configuration for the verifiers package.
    
    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create a StreamHandler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    # Get the root logger for the verifiers package
    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False 