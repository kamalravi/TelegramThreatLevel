# logs
import logging

def logger(log_dir_fname):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # formatter = logging.Formatter('%(asctime)s | %(name)s |  %(levelname)s: %(message)s')
    # FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    formatter = logging.Formatter('%(asctime)s | %(filename)s: %(lineno)s: %(funcName)s() |  %(levelname)s: %(message)s')

    # log_dir_fname ideally describes the dataPrep/model/KFold; all in one string
    # log_fname = './logs/' + taskName + '.log'
    log_fname = log_dir_fname
    file_handler = logging.FileHandler(log_fname)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger