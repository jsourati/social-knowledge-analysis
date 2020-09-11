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


def embedding(cocrs, years_of_cocrs_columns, model_or_path, y_term, **kwargs):

    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)
    
    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    if isinstance(model_or_path, str):
        model = Word2Vec.load(model_or_path)
    else:
        model = model_or_path

    
    def embedding_predictor(year_of_pred, sub_chems):

        # make sure that the order of chemicals in sub_chems is the same as
        # the order of their appearance in the universal array chems
        # because the order of unstudied_indic that is used
        # in the following lines corresponds to chems
        sub_chems = chems[np.isin(chems,sub_chems)]

        # embedding-based predictions can only work with subset of chemicals
        # that exist in the vocabulary of their pre-trained model
        overlap_indic = np.in1d(chems, sub_chems)
        sub_cocrs = cocrs[overlap_indic, :]
        
        """ Restricting Attention to Unstudied Materials """
        yr_loc = np.where(years_of_cocrs_columns==year_of_pred)[0][0]
        unstudied_indic = np.sum(sub_cocrs[:,:yr_loc], axis=1)==0
        sub_chems = sub_chems[unstudied_indic]

        """ Computing and Sorting Scores """
        scores = measures.cosine_sims(model, sub_chems, y_term)

        sorted_inds = np.argsort(-scores)[:pred_size]

        if return_scores:
            return sub_chems[sorted_inds], scores[sorted_inds]
        else:
            return sub_chems[sorted_inds]

    return embedding_predictor

def embedding_SD_lincomb(cocrs,
                         ySD,
                         years_of_cocrs_columns,
                         path_to_wvmodel,
                         y_term,
                         **kwargs):
    """Linear combination of embedding- and SD-based predictions
    """

    pred_size = kwargs.get('pred_size', 50)
    beta = kwargs.get('beta', 0.5)
    memory = kwargs.get('memory', 5)
    scalarization = kwargs.get('scalarization', 'SUM')


    assert 0<beta<1, 'beta should be between zero and one.'

    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    model = Word2Vec.load(path_to_wvmodel)

    def embedding_SD_predictor(year_of_pred, sub_chems):
        
        # Similar to embedding-based predictions, this predictor 
        # also can only work with subset of chemicals 
        # that exist in the vocabulary of their pre-trained model
        overlap_indic = np.in1d(chems, sub_chems)
        sub_cocrs = cocrs[overlap_indic, :]
        sub_ySD = ySD[overlap_indic, :]
        
        """ Restricting Attention to Unstudied Materials """
        yr_loc = np.where(years_of_cocrs_columns==year_of_pred)[0][0]
        unstudied_indic = np.sum(sub_cocrs[:,:yr_loc], axis=1)==0
        sub_chems = sub_chems[unstudied_indic]
        sub_ySD = sub_ySD[unstudied_indic,:yr_loc]

        """ Two Types of Scores """
        scores_0 = measures.ySD_scalar_metric(sub_ySD,
                                              mtype=scalarization,
                                              memory=memory)
        scores_1 = measures.cosine_sims(model, sub_chems, y_term)

        """ Normalizing Scores' Scales """
        scores_0 = (scores_0 - scores_0.min()) / (scores_0.max() - scores_0.min())
        scores_1 = (scores_1 - scores_1.min()) / (scores_1.max() - scores_1.min())

        """ Combining and Sorting Scores """
        scores = beta*scores_1 + (1-beta)*scores_0
        sorted_inds = np.argsort(-scores)[:pred_size]

        return sub_chems[sorted_inds]

    return embedding_SD_predictor

def hypergraph_access(cocrs,
                      years_of_cocrs_columns,
                      path_to_VM_core,
                      path_to_VM_kw,
                      **kwargs):
    
    VM = sparse.load_npz(path_to_VM_core)
    kwVM = sparse.load_npz(path_to_VM_kw)
    R = sparse.hstack((VM, kwVM), 'csc')

    pred_size = kwargs.get('pred_size', 50)
    nstep = kwargs.get('nstep', 1)
    memory = kwargs.get('memory', 5)
    #scalarization = kwargs.get('scalarization', 'SUM')
    return_scores = kwargs.get('return_scores', False)

    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    def access_score(year_of_pred, sub_chems):
        
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

        years = np.arange(year_of_pred-memory, year_of_pred)
        scores = measures.accessibility_scores(years,
                                               R=R,
                                               sub_chems=sub_chems,
                                               nstep=nstep)

        sorted_inds = np.argsort(-scores)[:pred_size]

        if return_scores:
            return sub_chems[sorted_inds], scores[sorted_inds]
        else:
            return sub_chems[sorted_inds]

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


def countsort_deepwalk(cocrs,
                       years_of_cocrs_columns,
                       path_to_deepwalk,
                       **kwargs):

    pred_size = kwargs.get('pred_size', 50)
    return_scores = kwargs.get('return_scores', False)

    # get all chemicals
    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    # chemicals in the deepwalk
    deepwalk_chems, counts = hypergraphs.extract_chems_from_deepwalks(path_to_deepwalk)
    
    def countsort_deepwalk_predictor(year_of_pred, sub_chems):
        
        if sub_chems is not None:
            overlap_indic = np.in1d(chems, sub_chems)
            sub_cocrs = cocrs[overlap_indic, :]

            # make sure that the order of chemicals in sub_chems is the same as
            # the order of their appearance in the universal array chems
            # because the order of unstudied_indic that is used
            # in the following lines corresponds to chems
            sub_chems = chems[np.isin(chems,sub_chems)]

        else:
            sub_chems = chems
            sub_cocrs = cocrs

        """ Restricting Attention to Unstudied Materials """
        yr_loc = np.where(years_of_cocrs_columns==year_of_pred)[0][0]
        unstudied_indic = np.sum(sub_cocrs[:,:yr_loc], axis=1)==0
        sub_chems = sub_chems[unstudied_indic]

        # sort chemicals based on their counts in the deepwalk sentences
        deepwalk_sub_chems = deepwalk_chems[np.isin(deepwalk_chems, sub_chems)]
        deepwalk_sub_counts = counts[np.isin(deepwalk_chems, sub_chems)]
        
        sorted_inds = np.argsort(-deepwalk_sub_counts)[:pred_size]

        if return_scores:
            return deepwalk_sub_chems[sorted_inds], deepwalk_sub_counts[sorted_inds]
        else:
            return deepwalk_sub_chems[sorted_inds]

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
                                                            year_of_pred,
                                                            memory)

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
    
    if isinstance(model_or_path, str):
        model = Word2Vec.load(model_or_path)
    else:
        model = model_or_path

    authors = np.array(list(model.wv.vocab.keys()))
    
    def author_embedding_predictor(year_of_pred):
        scores = measures.cosine_sims(model, authors, y_term)
        sorted_inds = np.argsort(-scores)[:pred_size]

        return np.array([int(x[2:]) for x in authors[sorted_inds]])

    return author_embedding_predictor


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
