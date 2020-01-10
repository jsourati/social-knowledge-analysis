import os
import sys
import pdb
import logging
import numpy as np

path = '/home/jamshid/codes/social-knowledge-analysis'
sys.path.insert(0, path)

from misc import helpers

def ranking_scores_acc(scores, cocrs, yridx, k=50):
    """Returning the accuracy of predicting co-occurrence of a pair of
    (entity/property) in the future publications based on a given set
    of pre-calculated scores

    The provided co-occurrence matrix indicates the co-occurrences that
    happened in reality in the future years. The provided scores have to
    be computed based on only the information collected from the years
    proceeded those associated with the co-occurrence matrix.

    :Parameters:

    * scores: array or list (length: n)
        list of scores pre-calculated for all entities

    * cocrs: 2D array or matrix (dim: nxm)
       cooccurrence matrix, where the rows corresponds to entities (n) 
       and the columns to years (y_0, y_1, ..., y_m) such that the
       (i,j)-th element indicates cooccurrence of the i-th entity (with
       property y) in the j-th year (i.e., y_j)

    * yridx: integer
        index of the year from which the prediction will begin (referring to y_yridx)
        the scores should be calculated using the data before this year

    The procedure is as follows:
    - ranking the scores and taking the first K ranks as entities predicted
      to be of the specified property
    - for each after yr >= y_yridx:
        = finding entities co-occurred with y for the first time in year yr
        = counting how many of the predicted entities exist in the real 
          co-occurrences
    """

    assert len(scores)==cocrs.shape[0], 'Number of scores shuold match number of \
                                         columns in co-occurrence matrix.'

    ranked_idx = np.argsort(-scores)[:k]
    accs = np.zeros(cocrs.shape[1]-yridx)
    for i in range(yridx, cocrs.shape[1]):
        cocrred_ents = helpers.find_first_time_cocrs(cocrs, i)
        overlap = set(ranked_idx).intersection(set(cocrred_ents))
        accs[i-yridx] = len(overlap)/k

    return accs

def average_accs_dict(accs_dict):

    yrs = list(accs_dict.keys())
    av_accs = np.zeros(len(accs_dict[np.min(yrs)]))
    for i in range(len(av_accs)):
        vals = [accs[i] for _,accs in accs_dict.items() if len(accs)>i]
        av_accs[i] = np.mean(vals)
    return np.array(av_accs)
