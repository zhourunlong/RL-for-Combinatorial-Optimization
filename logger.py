from collections import defaultdict
import logging
import numpy as np

class Logger:
    def __init__(self, directory_name):
        self.console_logger = get_logger()
        self.stats = defaultdict(lambda: [])
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.unprinted = defaultdict(float)

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))
        self.tb_logger(key, value, t)
        self.unprinted[key] = value

    def print_recent_stats(self):
        log_str = ""
        for (k, v) in self.unprinted.items():
            log_str += "\t%s: %.4f" % (k, v)
        self.unprinted.clear()
        self.console_logger.info(log_str)
    
    def info(self, log_str):
        self.console_logger.info(log_str)



def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    return logger
