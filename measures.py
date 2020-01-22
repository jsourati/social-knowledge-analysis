import os
import sys
import pdb
import json
import logging
import pymysql
import pdb
import numpy as np

from gensim.models import Word2Vec

path = '/home/jamshid/codes/social-knowledge-analysis'
sys.path.insert(0, path)

from data import readers
from misc import helpers
from training.train import MyCallBack

config_path = '/home/jamshid/codes/data/sql_config_0.json'
msdb = readers.MatScienceDB(config_path, 'scopus')


def cooccurrences(Y_terms, ents, **kwargs):
    """Getting co-occurrences of a given list of entities and 
    a set of keywords (Y-terms) in  abstracts of the database
    """

    msdb.crsr.execute('SELECT COUNT(*) FROM chemical_paper_mapping;')
    cnt = msdb.crsr.fetchall()[0][0]
    print('Number of rows in chemical-paper-mapping: {}'.format(cnt))
    
    # setting up the logger
    logger_disable = kwargs.get('logger_disable', False)
    logfile_path =   kwargs.get('logfile_path', None)
    logger = helpers.set_up_logger(__name__, logfile_path, logger_disable)

   # downloading papers with Y-terms (Y-papers) and categorizing them yearwise
    logger.info('Downloading papers with terms {} in their abstracts'.format(Y_terms))
    (_,Y_papers), (_,Y_dates) = msdb.get_papers_by_keywords(Y_terms, ['paper_id','date'],'OR').items()
    Y_years = np.array([y.year for y in Y_dates])
    Y_distinct_yrs = np.unique(Y_years)
    min_yr = np.min(Y_years)
    max_yr = np.max(Y_years)
    yrs = np.arange(min_yr, max_yr+1)

    logger.info('{} papers with Y-terms have been downloaded. \
                 The earliest one is published in {}'.format(len(Y_papers), min_yr))
    cocrs = np.zeros((len(ents), len(yrs)))
    ents = np.array(ents)
    for i,yr in enumerate(Y_years):
        yr_loc = yr - min_yr

        # add co-occurrences to all chemicals present in this paper
        # all chemicals in this paper
        present_ents = msdb.get_chemicals_by_paper_id(int(Y_papers[i]))
        present_ents_formula = present_ents['formula'] if len(present_ents)>0 else []
        present_ents_formula = list(set(present_ents_formula).intersection(set(ents)))
        present_ents_locs = [np.where(ents==frml)[0][0] for frml in present_ents_formula]
        
        for cloc in present_ents_locs:
            cocrs[cloc, yr_loc] += 1
            
        if not(i%1000):
            logger.info('{} papers is reviewed.'.format(i))

    return cocrs, yrs

def yearwise_authors_IOU(X,Y, x_chemical=True,y_chemical=False):
    """Returning author intersection of union for different years

    The years are going to be based on the dates of the Y-related papers
    """

    if isinstance(Y, list):
        Y_authors = msdb.get_yearwise_authors_by_keywords(Y, y_chemical)
    else:
        Y_authors = Y
    y_yrs = list(Y_authors.keys())
    
    if isinstance(X, list):
        X_authors = msdb.get_yearwise_authors_by_keywords(X,
                                                          x_chemical,
                                                          min_yr=np.min(y_yrs),
                                                          max_yr=np.max(y_yrs))
    elif isinstance(X,dict):
        X_authors = X
        X.update({key: [] for key in Y_authors if key not in X_authors})

    overlap_dict = {}
    for yr, Y_auths in Y_authors.items():
        X_auths = X_authors[yr]
        overlap_dict[yr] = list(set(Y_auths).intersection(set(X_auths)))

    return overlap_dict


def SD(Y_terms, chems, **kwargs):
    """Returning overall and year-wise Social Density (SD) values for a set
    of chemical compounds and a set of properties (Y-terms)

    Jaccardian SD(X,Y) = |A(X) intersect. A(Y)| \ |A(X)|+|A(Y)|
    """

    # setting up the logger
    logger_disable = kwargs.get('logger_disable', False)
    logfile_path =   kwargs.get('logfile_path', None)
    logger = helpers.set_up_logger(__name__, logfile_path, logger_disable)

    msdb.crsr.execute('SELECT COUNT(*) FROM paper;')
    logger.info('Total number of documents in the DB: {}'.format(
        msdb.crsr.fetchall()[0][0]))


    # downloading papers with Y-terms (Y-papers) and 
    # categorizing them yearwise
    logger.info('Downloading authors for terms {} in their abstracts'.format(Y_terms))
    Y_authors = msdb.get_yearwise_authors_by_keywords(Y_terms)
    unique_Y_authors = np.unique(sum([val for _,val in Y_authors.items()],[]))
    min_yr = np.min(list(Y_authors.keys()))
    max_yr = np.max(list(Y_authors.keys()))
    logger.info('Downloading is done. The oldest paper is published in {}.'.format(min_yr))
    logger.info('The total number of unique authots is {}.'.format(len(unique_Y_authors)))

    # but we only need those years where there exist non-zero set of Y-authors
    # otherwise, the SD will be zero no matter how many X-authors we get
    nnz_Y_yrs = [key for key, val in Y_authors.items() if len(val)>0]

    # iterating over chemicals and compute SD for each
    SDs = np.zeros(len(chems))
    yr_SDs = np.zeros((len(chems), max_yr-min_yr+1))
    years = np.arange(min_yr, max_yr+1)
    save_dirname = kwargs.get('save_dirname', None)
    logger.info('Iterating over chemicals for computing social densities began.')
    for i, chm in enumerate(chems):
        if not(i%1000):
            logger.info('Iteration {}..'.format(i))
            if save_dirname is not None:
                np.savetxt(os.path.join(save_dirname, 'SDs.txt'), SDs)
                np.savetxt(os.path.join(save_dirname, 'yr_SDs.txt'), yr_SDs)

        X_authors = msdb.get_yearwise_authors_by_keywords([chm], chemical=True)
        overlap_dict = yearwise_authors_IOU(X_authors, Y_authors)
        for yr in nnz_Y_yrs:
            yr_SDs[i,yr-min_yr] = 2*len(overlap_dict[yr])/(len(Y_authors[yr])+len(X_authors[yr]))
        
        # overall SD
        unique_X_authors = np.unique(sum([val for _,val in X_authors.items()],[]))
        overlap = set(unique_Y_authors).intersection(set(unique_X_authors))
        SDs[i] = 2*len(overlap) / (len(unique_Y_authors)+len(unique_X_authors))

    return SDs, yr_SDs, years

def SD_metrics(yr_SDs, mtype='SUM', **kwargs):
    """Computing scores based on year-wise social densities

    :Parameters:

    * yr_SDs: 2D array or matrix
        matrix of social densities between the entities and a given
        property, such that the rows correspond to entities and columns
        to years

    * mtype: string
        metric type; default is SUM
    """

    if mtype=='SUM':
        memory = kwargs.get('memory', 5)
        scores = np.sum(yr_SDs[:, -memory:], axis=1)

    if mtype=='RANDOM':
        """In random selection, we randomly select some samples from those
        with non-zero SD signal in the period determined with the memory.
        The scores will be one for the selected samples and zero elsewhere
        """
        memory = kwargs.get('memory', 5)
        random_selection_size = kwargs.get('random_selection_size', 50)
        scores = np.zeros(yr_SDs.shape[0])
        nnz_idx = np.where(np.sum(yr_SDs[:,-memory:], axis=1)>0)[0]
        rand_sel = nnz_idx[np.random.permutation(len(nnz_idx))[:random_selection_size]]
        scores[rand_sel] = 1.

    return scores

def cosine_sims(model, chems, Y_term):
    """
    Note the following vectors in a gensim's word2vec model:

    model.wv.vectors, model.wv.syn0 and model.wv[word]:
        all these three give word embedding vectors, the first two use
        word indices and the last use the word's in string format to
        return the embedding vector
        ---SANITY CHECK---
        these three vectors are the same:
        model.wv.vectors[i,:], model.wv.syn0[i,:], model.wv[model.wv.index2word[i]]

    model.wv.trainables.syn1neg:
        output embedding used in negative sampling (take index, not string value)

    model.wv.trainables.syn1:
        output embedding used in heirarchical softmax
    """

    zw_y = model.wv[Y_term]
    zw_y = zw_y / np.sqrt(np.sum(zw_y**2))
    
    sims = -1.*np.ones(len(chems))
    for i,chm in enumerate(chems):
        if chm not in model.wv.vocab:
            continue

        idx = model.wv.vocab[chm].index
        zo_x = model.trainables.syn1neg[idx,:]
        #zo_x = model.wv[chm]
        zo_x = zo_x / np.sqrt(np.sum(zo_x**2))

        sims[i] = np.dot(zw_y, zo_x)

    return sims
