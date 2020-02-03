import os
import sys
import pdb
import logging
import numpy as np

from gensim.models import Word2Vec

path = '/home/jamshid/codes/social-knowledge-analysis'
sys.path.insert(0, path)

import measures
from misc import helpers
from data import utils, readers

config_path = '/home/jamshid/codes/data/sql_config_0.json'
msdb = readers.MatScienceDB(config_path, 'scopus')

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


def average_accs_dict(accs_dict, stat='mean'):

    yrs = list(accs_dict.keys())
    stats = np.zeros(len(accs_dict[np.min(yrs)]))
    for i in range(len(stats)):
        vals = [accs[i] for _,accs in accs_dict.items() if len(accs)>i]
        stats[i] = np.mean(vals) if stat=='mean' else np.std(vals)
    return np.array(stats)


def compare_emb_yrSD(model_paths_dict,
                     cocrs_path,
                     yr_SDs_path,
                     full_chems_path,
                     yrs_path,
                     Y_terms,
                     memory,
                     sd_score_type='SUM',
                     logfile_path=None):
    """Comparing embedding- and SD-based scoring functions in predicting
    co-occurrence

    :parameters:

    * model_paths_dict: dict
        dictionary of pre-trained models with the keys equal to the years
        associated with the models
    
    * cocrs_path: string
        path to the pre-calculated co-occurrence matrix

    * yr_SDs_path: string
        path to the pre-calculated year-wisec SD matrix

    * full_chems_path: string
        path to the set of chemicals associated with columns of 
        the co-occurrence and SD matrices

    * yrs_path: string
        path to the years associated with the rows of co-occurrence
        and SD matrices

    * y_term: string or list of strings
        the property term corresponding to the co-occurrence and SD matrices

    * memory: int
        the memory to be used in SD-based prediction
    """
    
    logger = helpers.set_up_logger(__name__, logfile_path, False)

    pr = utils.MaterialsTextProcessor()
    
    cocrs = np.loadtxt(cocrs_path)
    yr_SDs = np.loadtxt(yr_SDs_path)
    yrs = np.loadtxt(yrs_path)
    with open(full_chems_path, 'r') as f:
        full_chems = f.read().splitlines()
    full_chems = np.array(full_chems)
    logger.info('Number of materials in full set of chemicals: {}'.format(len(full_chems)))
    logger.info('The property-related keywords are {}.\n\n\n'.format(Y_terms))

    xvals_dict = {}
    accs_dict = {}
    for yr, model_path in model_paths_dict.items():
        logger.info('Start computing scores for year {}'.format(yr))
        yr_string = 'PROGRESS FOR {}: '.format(yr)
        model = Word2Vec.load(model_path)

        # extract chemicals from the vocabulary of the model
        logger.info(yr_string+'extracting chemicals from the vocabulary of the model with count thresholds.'.format(yr))
        model_chems = []
        for w in model.wv.index2word:
            if pr.is_simple_formula(w) and model.wv.vocab[w].count>3:
                if (pr.normalized_formula(w)==w) or (w in ['H2','O2','N2']):
                    model_chems += [w]
        logger.info(yr_string+'there are {} chemicals extracted.'.format(len(model_chems)))

        # get the intersection between the extracted chemicals and the full set
        logger.info(yr_string+'there are {} chemicals removed from model chemicals because they were missing in the full chemical set'.format(len(set(model_chems)-set(full_chems))))
        model_chems = list(set(model_chems).intersection(set(full_chems)))
        model_chems_indic_in_full = np.in1d(full_chems, np.array(model_chems))
        # this is the actual order of materials we will consider in model chemicals
        model_chems = full_chems[model_chems_indic_in_full]
        logger.info(yr_string+'number of total chemicals after correction: {}'.format(len(model_chems)))

        # get the submatrices corresponding to the corrected set of chemicals
        sub_cocrs  = cocrs[model_chems_indic_in_full,:]
        sub_yr_SDs = yr_SDs[model_chems_indic_in_full,:]

        # take materials unstudied until year yr
        yr_loc = np.where(yrs==yr)[0][0]
        unstudied_ents = np.sum(sub_cocrs[:,:yr_loc],axis=1)==0
        logger.info(yr_string+'number of unstudied materials: {}'.format(np.sum(unstudied_ents)))

        # computing the scores
        if type(Y_terms) is str:
            embedd_scores = measures.cosine_sims(model, model_chems[unstudied_ents], Y_terms)
        elif type(Y_terms) is list:
            embedd_scores = np.zeros((len(Y_terms), np.sum(unstudied_ents)))
            for i, y in enumerate(Y_terms):
                if y not in model.wv.vocab:
                    logger.info('{} is not in the vocabulary.'.format(y))
                    continue
                embedd_scores[i,:] = measures.cosine_sims(model,
                                                          model_chems[unstudied_ents],
                                                          y)
            embedd_scores = np.mean(embedd_scores[np.sum(embedd_scores,axis=1)!=0,:], axis=0)
        sd_scores = measures.SD_metrics(sub_yr_SDs[unstudied_ents, :yr_loc],
                                        mtype=sd_score_type,
                                        memory=memory)

        gamma = np.abs(np.mean(embedd_scores) / (np.mean(sd_scores[sd_scores>0])))
        logger.info(yr_string+'gamma coefficient computed: {}'.format(gamma))
        betas = np.arange(0,1+1e-6,0.05)
        logger.info(yr_string+'beta range: {}'.format(betas))

        embedd_scores = (embedd_scores - embedd_scores.min()) / (embedd_scores.max()-embedd_scores.min())
        sd_scores = (sd_scores-sd_scores.min()) / (sd_scores.max()-sd_scores.min())
        accs = np.zeros((len(betas), len(yrs)-yr_loc))
        for i,b in enumerate(betas):
            scores = b*embedd_scores + (1-b)*sd_scores
            accs[i,:] = np.cumsum(ranking_scores_acc(scores,
                                                     sub_cocrs[unstudied_ents,:],
                                                     yr_loc))
        accs_dict[str(yr)] = accs.tolist()
        xvals_dict[str(yr)] = list([int(x) for x in np.arange(1,len(yrs)-yr_loc+1)])

    return xvals_dict, accs_dict
        
        
def eval_collective_SD(model_paths_dict,
                       cocrs,
                       yrs,
                       full_chems,
                       Y_terms,
                       case_sensitives=[],
                       logfile_path=None):
    
    logger = helpers.set_up_logger(__name__, logfile_path, False)

    pr = utils.MaterialsTextProcessor()

    # getting the yearwise Y-authors
    Y_authors = msdb.get_yearwise_authors_by_keywords(Y_terms,
                                                      return_papers=False,
                                                      case_sensitives=case_sensitives)
    
    xvals_dict = {}
    accs_dict = {}
    for yr, model_path in model_paths_dict.items():
        logger.info('Start computing scores for year {}'.format(yr))
        yr_string = 'PROGRESS FOR {}: '.format(yr)
        model = Word2Vec.load(model_path)

        # extract chemicals from the vocabulary of the model
        logger.info(yr_string+'extracting chemicals from the vocabulary of the model with count thresholds.'.format(yr))
        model_chems = []
        for w in model.wv.index2word:
            if pr.is_simple_formula(w) and model.wv.vocab[w].count>3:
                if (pr.normalized_formula(w)==w) or (w in ['H2','O2','N2']):
                    model_chems += [w]
        logger.info(yr_string+'there are {} chemicals extracted.'.format(len(model_chems)))

        # get the intersection between the extracted chemicals and the full set
        logger.info(yr_string+'there are {} chemicals removed from model chemicals because they were missing in the full chemical set'.format(len(set(model_chems)-set(full_chems))))
        model_chems = list(set(model_chems).intersection(set(full_chems)))
        model_chems_indic_in_full = np.in1d(full_chems, np.array(model_chems))
        # this is the actual order of materials we will consider in model chemicals
        model_chems = full_chems[model_chems_indic_in_full]
        logger.info(yr_string+'number of total chemicals after correction: {}'.format(len(model_chems)))

        # get the submatrices corresponding to the corrected set of chemicals
        sub_cocrs  = cocrs[model_chems_indic_in_full,:]

        # take materials unstudied until year yr
        yr_loc = np.where(yrs==yr)[0][0]
        unstudied_idx = np.where(np.sum(sub_cocrs[:,:yr_loc],axis=1)==0)[0]

        # get unique authors of Y-terms for all years<yr
        Yset = np.unique(sum([vals for y,vals in Y_authors.items() if y<yr],[]))
        
        # computing scores for unstudied samples
        sd_scores = np.zeros(len(unstudied_idx))
        for i,chm in enumerate(full_chems[unstudied_idx]):
            # all authors published on x
            X_authors = msdb.get_yearwise_authors_by_keywords([chm],chemical=True,return_papers=False)
            # only those authors published on X before years<yr
            Xset = np.unique(sum([vals for y,vals in X_authors.items() if y<yr],[]))
            overlap = set(Xset).intersection(set(Yset))
            union = set(Xset).union(set(Yset))
            sd_scores[i] = len(overlap) / len(union)

        accs = np.cumsum(ranking_scores_acc(sd_scores,
                                            sub_cocrs[unstudied_idx,:],
                                            yr_loc))
        accs_dict[str(yr)] = accs
        xvals_dict[str(yr)] = list([int(x) for x in np.arange(1,len(yrs)-yr_loc+1)])

    return xvals_dict, accs_dict
