import os
import sys
import copy
import logging
import numpy as np


def set_up_logger(log_name, logfile_path, logger_disable, file_mode='w'):
    """Setting up handler of the "root" logger as the single main logger
    """
    
    logger = logging.getLogger(log_name)
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

def locate_array_in_array(moving_arr, fixed_arr):
    """For each overlapping element in moving_arr, find its location 
    index in fixed_arr

    The assumption is that moving the array is a subset of the fixed array
    """

    assert np.sum(np.isin(moving_arr, fixed_arr))==len(moving_arr), \
        'Moving array should be a subset of fixed array.'
    
    sinds = np.argsort(fixed_arr)
    locs_in_sorted = np.searchsorted(fixed_arr[sinds], moving_arr)

    return sinds[locs_in_sorted]
    

def find_first_time_cocrs(cocrs, col):
    """Detecting first-time cooccurrences in a given column
    of the cooccurrence matrix
    """

    bin_indic = (np.sum(cocrs[:,:col],axis=1)==0) * (cocrs[:,col]>0)
    return np.where(bin_indic)[0]


def prune_deepwalk_sentences(sents, remove='author'):

    # removing authors or chemicals
    if remove=='author':
        hl = [[s for s in h.split(' ') if 'a_' not in s] for h in sents]
    elif remove=='chemical':
        hl = [[s for s in h.split(' ') if ('a_' in s) or ('thermoelectric' in s)]
              for h in sents]
    elif remove=='author_affiliation':
        hl = [[s for s in h.split(' ') if '_' not in s] for h in sents]

    # rejoining the split terms and ignoring those with singular terms
    hl = [' '.join(h) for h in hl if len(h)>1]

    # removing dots
    hl = [h.split('.')[0] for h in hl]

    # removing those sentences only containing the keyword
    hl = [h for h in hl if len(np.unique(h.split(' ')))>1]

    return hl


def lighten_color(color, amount=0.5):
    """
    Downloaded
    -----------
    This piece of code is downloaded from
    https://stackoverflow.com/a/49601444/8802212

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
