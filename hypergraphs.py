import os
import sys
import pdb
import json
import logging
import pymysql
import numpy as np
from scipy import sparse

from gensim.models import Word2Vec


path = '/home/jamshid/codes/social-knowledge-analysis'
sys.path.insert(0, path)

from data import readers
from misc import helpers
config_path = '/home/jamshid/codes/data/sql_config_0.json'
msdb = readers.MatScienceDB(config_path, 'msdb')

def compute_vertex_matrix(**kwargs):
    """Forming vertex matrix of the hypergraph, which is a |E|x|V|
    matrix and its (i,j) element is equal to 1 if hyperedge (article)
    i has node j and zero otherwise
    
    The hyperdeges are the articles and nodes are the union of author and
    chemical nodes
    """

    # setting up the logger
    logger_disable = kwargs.get('logger_disable', False)
    logfile_path =   kwargs.get('logfile_path', None)
    logger = helpers.set_up_logger(__name__, logfile_path, logger_disable)

    savefile_path = kwargs.get('savefile_path',None)

    msdb.crsr.execute('SELECT COUNT(*) FROM author;')
    nA = msdb.crsr.fetchone()[0]
    msdb.crsr.execute('SELECT COUNT(*) FROM chemical;')
    nC = msdb.crsr.fetchone()[0]
    
    logger.info('There are {} author nodes and {} chemical nodes in the database.'.format(nA,nC))

    nP = 1507143
    
    VM = sparse.lil_matrix((nP,nA+nC), dtype=np.uint8)
    # filling the matrix with batches
    cnt = 0
    batch_size = 500
    logger.info('Starting to fill the vertex matrix with batches of size {}'.format(batch_size))
    while cnt<nP:
        pids = np.arange(cnt, cnt + batch_size)
        auids = msdb.get_authors_by_paper_ids(pids, cols=['author_id'])
        chemids = msdb.get_chemicals_by_paper_ids(pids, cols=['chem_id'])

        cols = []
        rows = []
        for i,pid in enumerate(pids):
            au_cols   = auids[pid]['author_id'] if pid in auids else []
            chem_cols = chemids[pid]['chem_id'] + nA if pid in chemids else []
            cols += [np.concatenate((au_cols, chem_cols))]
            rows += [pid*np.ones(len(au_cols)+len(chem_cols))]

        cols = np.concatenate(cols)
        rows = np.concatenate(rows)
        VM[rows,cols] = 1

        cnt += batch_size

        if not(cnt%500):
            logger.info('{} articles have been processed'.format(cnt))
            if not(cnt%10000) and (savefile_path is not None):
                sparse.save_npz(savefile_path, VM.tocsc())

    return VM

def compute_vertex_KW_submatrix(los, **kwargs):
    """Forming a submatrix corresponding to conceptual nodes
    given as a set of keywords priveded in `los` (list of strings) arguments
    """

    case_sensitives = kwargs.get('case_sensitives', [])
    
    nP = 1507143
    ncols = len(los)
    VM = sparse.lil_matrix((nP,ncols), dtype=np.uint8)

    for i, kw in enumerate(los):
        if kw in case_sensitives:
            cs = [kw]
        else:
            cs = []

        R = msdb.get_papers_by_keywords([kw], case_sensitives=cs)
        rows = R['paper_id']
        cols = i*np.ones(len(rows))

        VM[rows, cols] = 1

    return VM


def compute_transprob(R):
    """Computing the transition probability matrix given the
    binary (0-1) vertex weight matrix (dim.; |E|x|V|)
    """

    row_collapse = np.array(np.sum(R,axis=0))[0,:]
    iDV = np.zeros(len(row_collapse), dtype=float)
    iDV[row_collapse>0] = 1./row_collapse[row_collapse>0]
    iDV = sparse.diags(iDV, format='csr')

    col_collapse = np.array(np.sum(R,axis=1))[:,0]
    iDE = np.zeros(len(col_collapse), dtype=float)
    iDE[col_collapse>0] = 1./col_collapse[col_collapse>0]
    iDE = sparse.diags(iDE, format='csr')
    
    return iDV * R.T * iDE * R


def prune_vertex_weight_matrix(R):
    """Pruning a vertex weight matrix by removing empty rows and columns

    This function does not change the order of columns or rows, it only removes
    the empty ones. Therefore, it is safe to use the output binary `cmask` to 
    get the new set of columns

    E.g. if orig_cols = array([A_1,A_2,...,A_n,C_1,...,C_m,K1,...,K_p])
        then new_cols = orig_cols[cmask] 
    """

    # remove papers with zero authors and chemicals (hyperedges with no nodes!)
    mask = np.ones(R.shape[0],dtype=bool)
    mask[np.array(np.sum(R,axis=1))[:,0]==0] = 0
    R = R.tocsr()[mask,:]

    # remove isolated nodes (nodes that are not in any hyperedges)
    cmask = np.ones(R.shape[1],dtype=bool)
    cmask[np.array(np.sum(R,axis=0))[0,:]==0] = 0
    R = R.tocsc()[:,cmask]
    
    return R, cmask

     
def restrict_rows_to_years(R, years):
    """Restricting a hypergraph with vertex weight matrix R to
    a given set of years

    Restriction is done by keeping only the hyperedges (articles) 
    whose date is in given years, pruning the resulting hypergraph 
    (by removing isolated nodes) and computing the transition
    probability matrix.
    """

    """ Restricting R to Articles in the Specified Years """
    # choosing rows (articles) associated with the given years
    yrs_arr = ','.join([str(x) for x in years])
    msdb.crsr.execute('SELECT paper_id FROM paper WHERE \
                       YEAR(date) IN ({});'.format(yrs_arr))
    yr_pids = np.array([x[0] for x in msdb.crsr.fetchall()])
    R = R[yr_pids,:]
    
    return R


def compute_multistep_transprob(P, source_inds, dest_inds, **kwargs):
    """Computing probability of multi-step transitions between two sets of nodes
    via a third intermediary set of nodes
    """

    interm_inds = kwargs.get('interm_inds', None)
    nstep = kwargs.get('nstep', 1)

    if interm_inds is None:
        # number of authors 
        msdb.crsr.execute('SELECT COUNT(*) FROM author;')
        nA = msdb.crsr.fetchone()[0]
        interm_inds = np.arange(nA)

    source_subP = P[source_inds,:]
    dest_subP = P[:,dest_inds]

    if nstep == 1:
        return source_subP[:,dest_inds]
    
    elif nstep==2:
        return source_subP[:,interm_inds] * dest_subP[interm_inds,:]
    
    elif nstep > 2:
        # for nstep=t, we need to have
        # P[source,A] * P[A,A]^t * P[A,dest] =
        # (((P[source,A] * P[A,A]) * P[A,A]) * ... ) * P[A,A] * P[A,inds]
        #               |------------------------------------|
        #                multiply for t times (preserve the order)
        #

        interm_subP = P[interm_inds,:][:,interm_inds]    #P[A,A]
        left_mat = source_subP[:,interm_inds] * interm_subP
        for t in range(1,nstep-2):
            left_mat = left_mat * interm_subP
        return left_mat * dest_subP[interm_inds,:]
        
    
