"""logger.py

A general-purpose logging utility.

This module provides a `Logger` class that wraps Python's built-in
`logging` module to output logs to the console and (optionally) to a uniquely
named log file. Log files are rotated automatically to prevent uncontrolled growth.

Example:
    from logger import Logger

    logger = Logger(__name__, save_to_file=True).get_logger()
    logger.info("This is an INFO log.")
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()


class Logger:
    """Encapsulates logging configuration.

    Attributes:
        logger (logging.Logger): Configured logger instance.
    """

    def __init__(
        self,
        name: str = __name__,
        log_dir: str = "my_library/logs",
        level: int = None,
        save_to_file: bool = None
    ) -> None:
        """Initializes the Logger class.

        Args:
            name (str): The name of the logger (typically `__name__`).
            log_dir (str): Directory to store log files.
            level (int): Logging level (e.g., logging.INFO).
            save_to_file (bool): Whether to save logs to a file. Default is False.
        """
        self.level = level or getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
        if save_to_file is not None:
            self.save_to_file = save_to_file
        else:
            self.save_to_file = os.getenv("SAVE_LOG_TO_FILE", "False") == "True"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False  # Prevent duplicate logs

        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Optional rotating file handler
            if self.save_to_file:
                os.makedirs(log_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = os.path.join(log_dir, f"{name}_{timestamp}.log")

                file_handler = RotatingFileHandler(
                    log_filename,
                    maxBytes=10**6,
                    backupCount=5,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Returns the configured logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger


if __name__ == "__main__":
    # Create a logger that saves logs to a uniquely named file
    # logger = Logger(__name__, save_to_file=True).get_logger()
    logger = Logger(__name__, save_to_file=False).get_logger()

    logger.info("This is an INFO log.")
    logger.warning("This is a WARNING log.")
    logger.error("This is an ERROR log.")
