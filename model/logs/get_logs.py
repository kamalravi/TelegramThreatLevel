# logs
import logging
import os

def setup_logger(log_dir_fname):
    # Ensure directory exists
    log_dir = os.path.dirname(log_dir_fname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a formatter with the desired format
    formatter = logging.Formatter('%(asctime)s | %(filename)s: %(lineno)s: %(funcName)s() | %(levelname)s: %(message)s')

    # Set up file handler for logging to a file
    file_handler = logging.FileHandler(log_dir_fname)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Set up stream handler for logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger