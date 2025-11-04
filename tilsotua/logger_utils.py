import logging
from datetime import datetime

TILSOTUA_LOGNAME = 'tilsotua'

def get_log():
    log_name = TILSOTUA_LOGNAME

    # get the log if already exists
    log = logging.getLogger(log_name)
    if not log.handlers:
        print('an error occurred while getting the log')
        return None

    return log


def configure_logger(log_dir):
    log_name = TILSOTUA_LOGNAME

    # get the log if already exists
    log = logging.getLogger(log_name)
    if log.handlers:
        return log

    # append the date for the log
    utd = datetime.utcnow().strftime('%Y%m%d')

    # set-up the logger
    log_path = f'{log_dir}/{log_name}_{utd}.log'
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(funcName)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    return log