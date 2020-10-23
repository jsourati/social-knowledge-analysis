import os
import sys
import pdb
import logging
import numpy as np
from scipy import sparse

from gensim.models import Word2Vec

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


def ySD(cocrs, ySD, years_of_cocrs_columns, **kwargs):
    """Returning a discovery predictor based on yearwise SD metric
    """

    memory = kwargs.get('memory', 5)
    scalarization = kwargs.get('scalarization', 'SUM')
    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)


    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    def ySD_predictor(year_of_pred, sub_chems):

        if sub_chems is not None:
            overlap_indic = np.in1d(chems, sub_chems)
            sub_cocrs = cocrs[overlap_indic, :]
            sub_ySD  = ySD[overlap_indic, :]
        else:
            sub_chems = chems
            sub_cocrs = cocrs
            sub_ySD = ySD

        """ Restricting Attention to Unstudied Materials """
        yr_loc = np.where(years_of_cocrs_columns==year_of_pred)[0][0]
        unstudied_indic = np.sum(sub_cocrs[:,:yr_loc], axis=1)==0
        sub_chems = sub_chems[unstudied_indic]
        sub_ySD = sub_ySD[unstudied_indic,:yr_loc]

        """ Computing and Sorting Scores """ 
        scores = measures.ySD_scalar_metric(sub_ySD,
                                            mtype=scalarization,
                                            memory=memory)

        sorted_inds = np.argsort(-scores)[:pred_size]

        if return_scores:
            return sub_chems[sorted_inds], scores[sorted_inds]
        else:
            return sub_chems[sorted_inds]


    return ySD_predictor


def embedding(ents,
              model_or_path,
              keyword,
              studied_ents_func,
              **kwargs):
    """

    `studied_ents_func` should be a function that gets a year and
    returns all the entities that have been studied up until that year
    (excluding that year)

    `kwargs.constraint_func` (if given) should be a function that takes
    a set of entities and return those that satisfy a built-in constraint
    """

    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)
    constraint_func = kwargs.get('constraint_func', None)
        
    if isinstance(model_or_path, str):
        model = Word2Vec.load(model_or_path)
    else:
        model = model_or_path

    # in general, we cannot rely on the vocabulary of our model to
    # form the set of entities, because if the model's vocabulary contains
    # phrases, we cannot distinguish them from actual entities
    voc = np.array([x for x in model.wv.vocab])
    ents = ents[np.isin(ents,voc)]
    
    def embedding_predictor(year_of_pred):
        
        """ Restricting Attention to Unstudied Materials """
        studied_ents = studied_ents_func(year_of_pred)
        
        """ Computing and Sorting Scores """
        unstudied_ents = ents[~np.isin(ents,studied_ents)]
        # filter the unstudied entities if a constraint function is given
        if constraint_func is not None:
            unstudied_ents = constraint_func(unstudied_ents)
        scores = measures.cosine_sims(model, unstudied_ents, keyword)

        sorted_inds = np.argsort(-scores)[:pred_size]

        if return_scores:
            return unstudied_ents[sorted_inds], scores[sorted_inds]
        else:
            return unstudied_ents[sorted_inds]

    return embedding_predictor


def hypergraph_trans_prob(ents,
                          row_years,
                          path_to_VM,
                          path_to_VMkw,
                          studied_ents_func,
                          **kwargs):
    
    VM = sparse.load_npz(path_to_VM)
    VMkw = sparse.load_npz(path_to_VMkw)
    R = sparse.hstack((VM, VMkw), 'csc')

    assert R.shape[0]==len(row_years), "number of given years should exactly " + \
        "match the number of rows in vertex-weight matrix"

    pred_size = kwargs.get('pred_size', 50)
    nstep = kwargs.get('nstep', 1)
    memory = kwargs.get('memory', 5)
    #scalarization = kwargs.get('scalarization', 'SUM')
    return_scores = kwargs.get('return_scores', False)
    constraint_func = kwargs.get('constraint_func', None)

    def access_score(year_of_pred):

        """ Keeping Only Papers in the Selected Date Range """
        lower_year = year_of_pred-memory if memory>0 else -np.inf
        subR = R[(lower_year<=row_years)*(row_years<year_of_pred), :]
        
        """ Restricting Attention to Unstudied Materials """
        studied_ents = studied_ents_func(year_of_pred)
        unstudied_ents = ents[~np.isin(ents,studied_ents)]

        # filter the unstudied entities if a constraint function is given
        if constraint_func is not None:
            unstudied_ents = constraint_func(unstudied_ents)

        scores = measures.accessibility_scores(ents,
                                               R=subR,
                                               sub_ents=unstudied_ents,
                                               nstep=nstep)

        sorted_inds = np.argsort(-scores)[:pred_size]

        if return_scores:
            return unstudied_ents[sorted_inds], scores[sorted_inds]
        else:
            return unstudied_ents[sorted_inds]

    return access_score


def random_deepwalk(cocrs,
                    years_of_cocrs_columns,
                    path_to_deepwalk,
                    **kwargs):

    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)


    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    # chemicals in the deepwalk
    deepwalk_chems,_ = hypergraphs.extract_chems_from_deepwalks(path_to_deepwalk)
    
    def random_deepwalk_predictor(year_of_pred, sub_chems):
        
        if sub_chems is not None:
            overlap_indic = np.in1d(chems, sub_chems)
            sub_cocrs = cocrs[overlap_indic, :]
        else:
            sub_chems = chems
            sub_cocrs = cocrs

        """ Restricting Attention to Unstudied Materials """
        yr_loc = np.where(years_of_cocrs_columns==year_of_pred)[0][0]
        unstudied_indic = np.sum(sub_cocrs[:,:yr_loc], axis=1)==0
        sub_chems = sub_chems[unstudied_indic]

        # randomly choosing chemicals from those that exist in the deepwalks
        deepwalk_sub_chems = deepwalk_chems[np.isin(deepwalk_chems, sub_chems)]
        # random selection by actually generating random numbers
        # (to be used in AUC metric evaluation)
        scores = np.random.random(len(deepwalk_sub_chems))
        sorted_inds = np.argsort(-scores)[:pred_size]

        if return_scores:
            return deepwalk_sub_chems[sorted_inds], scores[sorted_inds]
        else:
            return deepwalk_sub_chems[:pred_size]

    return random_deepwalk_predictor


def countsort_deepwalk(path_to_dw_sents,
                       studied_ents_func,
                       **kwargs):

    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)
    constraint_func = kwargs.get('constraint_func', None)


    # extract entities in the deepwalk sentences and their counts
    dw_ents, counts = hypergraphs.extract_chems_from_deepwalks(path_to_dw_sents)
    
    def countsort_deepwalk_predictor(year_of_pred):
        
        """ Restricting Attention to Unstudied Materials """
        studied_ents = studied_ents_func(year_of_pred)
        
        """ Computing and Sorting Scores """
        unstudied_dw_ents = dw_ents[~np.isin(dw_ents,studied_ents)]
        # filter the unstudied entities if a constraint function is given
        if constraint_func is not None:
            unstudied_dw_ents = constraint_func(unstudied_dw_ents)

        # sort entities based on their counts in the deepwalk sentences
        unstudied_dw_counts = counts[np.isin(dw_ents, unstudied_dw_ents)]
        
        sorted_inds = np.argsort(-unstudied_dw_counts)[:pred_size]

        if return_scores:
            return unstudied_dw_ents[sorted_inds], unstudied_dw_counts[sorted_inds]
        else:
            return unstudied_dw_ents[sorted_inds]

    return countsort_deepwalk_predictor


def first_passage_distance_deepwalk(cocrs,
                                    years_of_cocrs_columns,
                                    path_to_deepwalk,
                                    **kwargs):

    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)
    # saved_dists should be in form of a dictionary: {ent_x: AFP_x}
    #                                                 -----  -----
    #                                                 (str)  (float)
    saved_dists = kwargs.get('saved_dists', None)

    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    # sentences
    sents = open(path_to_deepwalk, 'r').read().splitlines()
    KW = sents[0].split(' ')[0]

    if saved_dists is None:
        # get the present entities
        dw_chems = hypergraphs.extract_chems_from_deepwalks(path_to_deepwalk)[0]
        dists = hypergraphs.compute_av_first_passage_distance(sents, KW, dw_chems)
    else:
        dw_chems = np.array([x for x in saved_dists.keys()])
        dists    = np.array([x for x in saved_dists.values()]) 

    def predictor(year_of_pred, sub_chems):
        if sub_chems is not None:
            overlap_indic = np.in1d(chems, sub_chems)
            sub_cocrs = cocrs[overlap_indic, :]
            sub_chems = chems[np.isin(chems,sub_chems)]
        else:
            sub_chems = chems
            sub_cocrs = cocrs

        """ Restricting Attention to Unstudied Materials """
        yr_loc = np.where(years_of_cocrs_columns==year_of_pred)[0][0]
        unstudied_indic = np.sum(sub_cocrs[:,:yr_loc], axis=1)==0
        sub_chems = sub_chems[unstudied_indic]

        # sort entities based on their counts in the deepwalk sentences
        dw_sub_chems = dw_chems[np.isin(dw_chems, sub_chems)]
        dw_dists = dists[np.isin(dw_chems, sub_chems)]
        
        sorted_inds = np.argsort(dw_dists)[:pred_size]

        if return_scores:
            return dw_sub_chems[sorted_inds], dists[sorted_inds]
        else:
            return dw_sub_chems[sorted_inds]

    return predictor


def hypergraph_author_accesss(path_to_VM_core,
                              path_to_VM_kw,
                              row_yrs,
                              **kwargs):

    VM = sparse.load_npz(path_to_VM_core)
    kwVM = sparse.load_npz(path_to_VM_kw)
    R = sparse.hstack((VM, kwVM), 'csc')

    pred_size = kwargs.get('pred_size', 50)
    nstep = kwargs.get('nstep', 1)
    memory = kwargs.get('memory', 5)

    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    def author_access_scores(year_of_pred, size=0):

        if size==0:
            size=pred_size
        
        scores = measures.author_accessibility_scalar_score(R,
                                                            row_yrs,
                                                            year_of_pred,
                                                            memory,
                                                            nstep)

        sorted_inds = np.argsort(-scores)[:size]

        return sorted_inds

    return author_access_scores


def author_embedding(model_or_path, y_term, **kwargs):
    """Author (discoverer) prediction from a model that is trained based
    on deepwalks over keyword term and author ndoes

    The model is assumed to have a vocabulary consisting of only the keyword
    term and author identifiers in the form of "a_AUID"
    """

    pred_size = kwargs.get('pred_size', 50)
    constraint_func = kwargs.get('constraint_func', None)
    
    if isinstance(model_or_path, str):
        model = Word2Vec.load(model_or_path)
    else:
        model = model_or_path

    # exclude phrases from the list of authors
    authors = np.array([x for x in model.wv.vocab.keys()
                        if (x.count('a_')==1) and (x.count('_')==1)])

    # filter the author-set if a constraining function is given
    if constraint_func is not None:
        authors = constraint_func(authors)

    scores = measures.cosine_sims(model, authors, y_term)
    sorted_inds = np.argsort(-scores)[:pred_size]

    # in author prediction, we really don't need year of prediction
    # but define it that way to keep consistency with other functions
    def predict_func(year_of_pred):
        return np.array([int(x[2:]) for x in authors[sorted_inds]])

    return predict_func


def mfw2v_deepwalk(cocrs,
                   years_of_cocrs_columns,
                   mfw2v,
                   **kwargs):

    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)

    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    # keyword is always the first token
    KW = mfw2v.ind2tok[0]
    def predictor(year_of_pred, sub_chems):
        
        if sub_chems is not None:
            overlap_indic = np.in1d(chems, sub_chems)
            sub_cocrs = cocrs[overlap_indic, :]
            sub_chems = chems[np.isin(chems,sub_chems)]
        else:
            sub_chems = chems
            sub_cocrs = cocrs

        """ Restricting Attention to Unstudied Materials """
        yr_loc = np.where(years_of_cocrs_columns==year_of_pred)[0][0]
        unstudied_indic = np.sum(sub_cocrs[:,:yr_loc], axis=1)==0
        sub_chems = sub_chems[unstudied_indic]

        # sort chemicals based on their counts in the deepwalk sentences
        sorted_dw_chems = mfw2v.get_most_similar_terms(KW, None, len(mfw2v.uni_counts))
        sorted_scores = np.array([x[1] for x in sorted_dw_chems])
        sorted_dw_chems = np.array([x[0] for x in sorted_dw_chems])
        sorted_dw_sub_chems = sorted_dw_chems[np.isin(sorted_dw_chems, sub_chems)]
        sorted_dw_sub_chems_scores = sorted_scores[np.isin(sorted_dw_chems, sub_chems)]

        if return_scores:
            return sorted_dw_sub_chems[:pred_size], sorted_dw_chems_scores[:pred_size]
        else:
            return sorted_dw_sub_chems[:pred_size]

    return predictor


def pred_with_sims(cocrs,
                   years_of_cocrs_columns,
                   path_to_dw,
                   path_to_sims,
                   **kwargs):

    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)

    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    dw_chems = hypergraphs.extract_chems_from_deepwalks(path_to_dw)[0]
    
    def predictor(year_of_pred, sub_chems):
        
        if sub_chems is not None:
            overlap_indic = np.in1d(chems, sub_chems)
            sub_cocrs = cocrs[overlap_indic, :]
            sub_chems = chems[np.isin(chems,sub_chems)]
        else:
            sub_chems = chems
            sub_cocrs = cocrs

        """ Restricting Attention to Unstudied Materials """
        yr_loc = np.where(years_of_cocrs_columns==year_of_pred)[0][0]
        unstudied_indic = np.sum(sub_cocrs[:,:yr_loc], axis=1)==0
        sub_chems = sub_chems[unstudied_indic]
        
        # sort chemicals based on their similarities
        sims = np.loadtxt(path_to_sims)

        dw_sub_chems = dw_chems[np.isin(dw_chems, sub_chems)]
        sub_sims = sims[np.isin(dw_chems,sub_chems)]
        
        sorted_inds = np.argsort(-sub_sims)[:pred_size]
    
        if return_scores:
            return dw_sub_chems[sorted_inds], sub_sims[sorted_inds]
        else:
            return dw_sub_chems[sorted_inds]

    return predictor
