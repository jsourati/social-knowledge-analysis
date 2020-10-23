import os
import sys
import pdb
import json
import logging
import numpy as np
from scipy import sparse

from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score

path = '/home/jamshid/codes/social-knowledge-analysis'
sys.path.insert(0, path)

import measures
import hypergraphs
from misc import helpers
from data import utils, readers

config_path = '/home/jamshid/codes/data/sql_config_0.json'
msdb = readers.DB(config_path,
                  db_name='msdb',
                  entity_tab='chemical',
                  entity_col='formula')

pr = utils.MaterialsTextProcessor()


def average_accs_dict(accs_dict, stat='mean'):

    yrs = list(accs_dict.keys())
    stats = np.zeros(len(accs_dict[np.min(yrs)]))
    for i in range(len(stats)):
        vals = [accs[i] for _,accs in accs_dict.items() if len(accs)>i]
        stats[i] = np.mean(vals) if stat=='mean' else np.std(vals)
    return np.array(stats)
        
        
def eval_predictor(predictor_func,
                   gt_func,
                   year_of_pred,
                   **kwargs):
    """Evaluating a given predictor function in how accurate its predictions
    match the actual discoveries returned by a given ground-truth function

    The evaluations are done for individual years strating from a given year 
    of prediction to 2018.
    """

    metric = kwargs.get('metric', 'cumul_precision')
    last_year = kwargs.get('last_year', 2019)
    save_path = kwargs.get('save_path', None)
    return_preds = kwargs.get('return_preds', False)
    logfile_path = kwargs.get('logfile_path', None)
    logger_disable = kwargs.get('logger_disable',False)
    logger = helpers.set_up_logger(__name__, logfile_path, logger_disable)
    

    """ Generating the Prediction """
    preds = predictor_func(year_of_pred)
    logger.info('Number of actual predictions: {}'.format(len(preds)))
    if metric=='auc':
        if len(preds)!=2:
            raise ValueError('When asking for AUC metric, predictor should return score array too.')
        scores = preds[1]
        preds = preds[0]

    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write('\n'.join(preds)+'\n')
    
    """ Evaluating the Predictions for the Upcoming Years """
    years_of_eval = np.arange(year_of_pred, last_year)
    iter_list = []  # to be the prec. values or actul disc. (for AUC)   
    for i, yr in enumerate(years_of_eval):
        gt = gt_func(yr)

        if metric=='cumul_precision':      # Cumulative Precision
            iter_list += [np.sum(np.in1d(gt, preds)) / len(preds)]
        elif metric=='auc':    # Area Under Curve
            iter_list += gt.tolist()

    if metric == 'cumul_precision':
        res = np.cumsum(iter_list)
    elif metric == 'auc':
        y = np.zeros(len(preds))
        y[np.isin(preds,iter_list)] = 1
        res = roc_auc_score(y, scores)

    if return_preds:
        return res, preds
    else:
        return res


def eval_author_predictor(discoverers_predictor_func,
                          gt_discoverers_func,
                          year_of_pred,
                          **kwargs):
    """If return_yrs is True, then the function returns a dictionary
    that contains the years which true positives happened
    """

    last_year = kwargs.get('last_year', 2019)
    return_yrs = kwargs.get('return_yrs', False)
    
    preds = discoverers_predictor_func(year_of_pred)

    years_of_eval = np.arange(year_of_pred, last_year)
    precs = np.zeros(len(years_of_eval))
    inclusive_preds = preds
    yrs_dict = {}
    for i, yr in enumerate(years_of_eval):
        gt = gt_discoverers_func(yr)

        tpbin = np.isin(preds, gt)
        precs[i] = np.sum(tpbin)

        if return_yrs:
            tp = inclusive_preds[np.isin(inclusive_preds,gt)]
            for au in tp:
                # int() function is added below to keep the year types
                # int and not numpy.int64 which is not JSON serializable
                if au in yrs_dict:
                    yrs_dict[au] += [int(yr)]
                else:
                    yrs_dict[au] = [int(yr)]

        # exclude the ones that have already been labeled as a discoverer
        #preds = preds[~tpbin]

    if return_yrs:
        return precs/len(inclusive_preds), yrs_dict
    else:
        return precs/len(inclusive_preds)

