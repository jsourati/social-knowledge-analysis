import os
import sys
import copy
import logging
import numpy as np

def set_up_logger(log_name, logfile_path, logger_disable, file_mode='w'):
    """Setting up handler of the "root" logger as the single main logger
    """
    
    logger = logging.getLogger()
    if logger_disable:
        handler = logging.NullHandler()
    elif logfile_path is None:
        handler = logging.StreamHandler()
    else:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename=logfile_path,
                                      encoding='utf-8',
                                      mode=file_mode)
    handler.setFormatter(logging.Formatter("%(asctime)s : %(levelname)s : %(message)s"))
    logger.handlers = []
    logger.addHandler(handler)

    return logger

def find_first_time_cocrs(cocrs, col):
    """Detecting first-time cooccurrences in a given column
    of the cooccurrence matrix
    """

    bin_indic = (np.sum(cocrs[:,:col],axis=1)==0) * (cocrs[:,col]>0)
    return np.where(bin_indic)[0]
