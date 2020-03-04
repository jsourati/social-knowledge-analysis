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
from data.utils import MatTextProcessor
from hypergraphs import compute_transition_prob, compute_transprob_via_1author, \
   preprocess_VM

config_path = '/home/jamshid/codes/data/sql_config_0.json'
msdb = readers.MatScienceDB(config_path, 'msdb')
#pr = MatTextProcessor()

ignored_toks = ["from", "as", "at", "by", "of", "on", "into", "to", "than", "all", "its",
                "over", "in", "the", "a", "an", "/", "under","=", ".", ",", "(", ")",
                "<", ">", "\"", "", "", "", "", "-", "%", "&", "",
                "<nUm>", "been", "be", "are", "which", "where",
                 "were", "have", "important", "has", "can", "or", "we", "our",
                "article", "paper", "show", "there", "if", "these", "could", "publication",
                "while", "measured", "measure", "demonstrate", "investigate", "investigated",
                "demonstrated", "when", "prepare", "prepared", "use", "used", "determine",
                "determined", "find", "successfully", "newly", "present",
                "reported", "report", "new", "characterize", "characterized", "experimental",
                "result", "results", "showed", "shown", "such", "after",
                "but", "this", "that", "via", "is", "was", "and", "using", "for", "here"]



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
    case_sensitives = kwargs.get('case_sensitives', [])
    (_,Y_papers), (_,Y_dates) = msdb.get_papers_by_keywords(Y_terms,
                                                            cols=['paper_id','date'],
                                                            logical_comb='OR',
                                                            case_sensitives=case_sensitives).items()
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
        present_ents = msdb.get_chemicals_by_paper_ids(int(Y_papers[i]), cols=['formula'])
        present_ents_formula = present_ents[int(Y_papers[i])]['formula'] if len(present_ents)>0 else []
        present_ents_formula = list(set(present_ents_formula).intersection(set(ents)))
        present_ents_locs = [np.where(ents==frml)[0][0] for frml in present_ents_formula]
        
        for cloc in present_ents_locs:
            cocrs[cloc, yr_loc] += 1
            
        if not(i%1000):
            logger.info('{} papers is reviewed.'.format(i))

    return cocrs, yrs

def yearwise_authors_set_op(X_authors ,Y_authors):
    """Returning author intersection of union for different years

    The years are going to be based on the dates of the Y-related papers
    """

    # make sure 
    X_authors.update({key: [] for key in Y_authors if key not in X_authors})

    overlap_dict = {}
    union_dict   = {}
    for yr, Y_auths in Y_authors.items():
        X_auths = X_authors[yr]
        overlap_dict[yr] = list(set(Y_auths).intersection(set(X_auths)))
        union_dict[yr] = list(set(Y_auths).union(set(X_auths)))

    return overlap_dict, union_dict


def yearwise_SD(Y_terms, chems, **kwargs):
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


    # getting unique authors of Y-terms in different years
    case_sensitives = kwargs.get('case_sensitives',[])
    logger.info('Downloading authors for terms {} in their abstracts'.format(Y_terms))
    R = msdb.get_authors_by_keywords(Y_terms,
                                     cols=['author_id','P.date'],
                                     return_papers=False,
                                     case_sensitives=case_sensitives)
    if len(R)==0:
        raise ValueError('Given property terms are not associated with any papers in the data base')
    Y_years = np.array([y.year for y in R['date']])
    Y_authors = {y: R['author_id'][Y_years==y] for y in np.unique(Y_years)}
    unique_Y_authors = np.unique(R['author_id'])
    min_yr = np.min(Y_years)
    max_yr = np.max(Y_years)
    logger.info('Downloading is done. The oldest paper is published in {}.'.format(min_yr))
    logger.info('The total number of unique authors is {}.'.format(len(unique_Y_authors)))

    # iterating over chemicals and compute SD for each
    yr_SDs = np.zeros((len(chems), max_yr-min_yr+1))
    years = np.arange(min_yr, max_yr+1)
    save_dirname = kwargs.get('save_dirname', None)
    logger.info('Iterating over chemicals for computing social densities began.')
    for i, chm in enumerate(chems):
        if not(i%1000) or (i==len(chems)-1):
            logger.info('Iteration {}..'.format(i))
            if save_dirname is not None:
                np.savetxt(os.path.join(save_dirname, 'yr_SDs.txt'), yr_SDs)

        # getting unique authors of this materials in different years
        R = msdb.get_authors_by_chemicals([chm],
                                          cols=['author_id','P.date'],
                                          years=np.unique(Y_years),
                                          return_papers=False)
        if len(R)==0: continue
        X_years = np.array([y.year for y in R[chm]['date']])
        X_authors = {y: R[chm]['author_id'][X_years==y] for y in np.unique(X_years)}
        overlap_dict, union_dict = yearwise_authors_set_op(X_authors, Y_authors)
        for yr in Y_authors:
            yr_SDs[i,yr-min_yr] = len(overlap_dict[yr])/len(union_dict[yr])
        
    return yr_SDs, years

def ySD_scalar_metric(yr_SDs, mtype='SUM', **kwargs):
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
    
    sims = np.ones(len(chems))
    for i,chm in enumerate(chems):
        if chm not in model.wv.vocab:
            sims[i] = np.nan
            continue

        idx = model.wv.vocab[chm].index
        zo_x = model.trainables.syn1neg[idx,:]
        #zo_x = model.wv[chm]
        zo_x = zo_x / np.sqrt(np.sum(zo_x**2))

        sims[i] = np.dot(zw_y, zo_x)

    return sims

def accessibility_scores(year, **kwargs):
    """Computing accessibility between chemicals and the property keywords

    The hypergraph will be built based on a given pre-computed vertex weight matrix,
    or the paths to the raw sub-matrices (corresponding to (1) authors/chemicals nodes
    and (2) property keywords nodes, as two separate paths)

    Identifying node types in `P` is based on our certainty that the first chucnk
    of P is still corresponding to authors, the second chunk to chemicals and the
    third chunk to the property-related keywords.
    """
    
    """ Building General Vertex Weight Matrix (R) """
    R = kwargs.get('R', None)
    path_VM_core = kwargs.get('path_VM_core', None)
    path_VM_kw = kwargs.get('path_VM_kw', None)
    sub_chems = kwargs.get('sub_chems', [])

    assert (R is not None) or \
        ((path_VM_core is not None) and (path_VM_kw is not None)), \
        'Either the pre-computed vertex weight matrix (R), or the paths \
         to the submatrices need to be given.'

    if R is None:
        VM = sparse.load_npz(path_VM_core)
        kwVM = sparse.load_npz(path_VM_kw)
        R = sparse.hstack((VM, kwVM), 'csc')

    """ Preprocessing R """
    R, col_types, chems = preprocess_VM(R, years=[year], prune=True)

    # here, chems should be the same as sub_chems (if given), but possibly
    # in a different order, so we need it as one of the outputs

    """ Computing the Transition Probabilities """
    P = compute_transition_prob(R)

    """ Computing the Accessibility Scores """
    # number of authors, chemicals.. the rest will be property kewords
    # (these are fixed given that the data base is fixed..change the former
    # if the latter is modified)
    nA = np.sum(col_types==0)
    nC = np.sum(col_types==1)
    nKW = np.sum(col_types==2)

    A_inds = np.arange(nA)
    C_inds = np.arange(nA,nA+nC)
    KW_inds   = np.arange(nA+nC, len(col_types))

    transprob_CtoKW = compute_transprob_via_1author(P,
                                                    source_inds=C_inds,
                                                    dest_inds=KW_inds,
                                                    author_rows=A_inds)
    transprob_KWtoC = compute_transprob_via_1author(P,
                                                    source_inds=KW_inds,
                                                    dest_inds=C_inds,
                                                    author_rows=A_inds)

    # computing two-way transition probability (accessibility score)
    access_CtoKW = 0.5*(transprob_CtoKW + transprob_KWtoC.T)

    # summarizing multiple accessibility scores for chemicals corresponding to 
    # multiple proprety-related keywords
    access_CtoKW = np.squeeze(np.array(np.sum(access_CtoKW, axis=1)))

    return access_CtoKW, chems

def accessibility_scalar_metric(R,
                                year,
                                memory,
                                sub_chems,
                                mtype='MEAN'):

    years = np.arange(year-5,year)
    yrs_scores = np.zeros((memory,len(sub_chems)))
    for i, yr in enumerate(years):
        scores, chems = accessibility_scores(yr, R=R)
        # find location of elements of chems in sub_chems, so that
        # these local scores will be loaded in correct locations
        chems_in_subchems = helpers.locate_array_in_array(chems, sub_chems)
        yrs_scores[i,chems_in_subchems] = scores

    if mtype=='SUM':
        scores = np.sum(yrs_scores, axis=0)
    elif mtype=='MEAN':
        scores = np.mean(yrs_scores, axis=0)
    elif mtype=='MAX':
        scores = np.max(yrs_scores, axis=0)

    return scores


def sims_two_lists(S1, S2, model, emb_types='ww'):
    """Similarity between two lists of strings
    """

    if emb_types[0]=='w':
        mat1 = np.array([model.wv[x].tolist() for x in S1])
    elif emb_types[0]=='o':
        mat1 = np.array([model.wv(x).tolist() for x in S1])

    if emb_types[1]=='w':
        mat2 = np.array([model.wv[x].tolist() for x in S2])
    elif emb_types[1]=='o':
        mat2 = np.array([model.wv(x).tolist() for x in S2])

    inner_mat = np.inner(mat1, mat2)
    outer_mat = np.sqrt(np.outer(np.sum(mat1**2,axis=1),
                                 np.sum(mat2**2,axis=1)))

    return np.max(inner_mat/outer_mat)

def author_sim_curves(author_id, keywords, model):
    """Similarity of the content of an author's previous publications
    to a given set of keywords
    """

    (_,pids),(_,dates),(_,T),(_,A) = msdb.get_papers_by_author_id(
        [author_id],['paper_id','date','title','abstract']).items()
    years = [d.year for d in dates]

    sims = np.zeros(len(years))
    for i, abst in enumerate(A):
        tokens = sum(pr.mat_preprocess(abst)+pr.mat_preprocess(T[i]), [])
        tokens = [t for t in tokens if t in model.wv.vocab]
        tokens = list(set(tokens) - set(ignored_toks))
        
        sims[i] = sims_two_lists(tokens,keywords,model,'ww')
        
    return years, sims


    
