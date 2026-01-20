
import logging

def setup_logger(log_file, log_level=logging.DEBUG):
    """
    Set up a logger that writes to both a file and the console.
    :param log_file: Path to the log file.
    :param log_level: Logging level (default: DEBUG).
    :return: configured logger."""

    # create a logger with the module's name
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # clear old handlers if the logger already has any
    if logger.hasHandlers():
        logger.handlers.clear()

    # file handler: write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # console handler: print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # add both handlers to the logger
    logger.addHandler(file_handler)
    # if log_level <= logging.DEBUG:
    #     logger.addHandler(console_handler)

    return logger