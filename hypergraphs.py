import os
import sys
import pdb
import json
import logging
import pymysql
import numpy as np
from scipy import sparse
from collections import deque

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


def find_neighbors(idx, R):
    """Returning neighbors of a node indexed by `idx`
    """

    # indices of the hyperedges
    he_inds = R[:,idx].indices
    nbr_indic = R[he_inds,:].sum(axis=0)

    return np.where(nbr_indic)[1]


def year_discoveries(R, year, **kwargs):
    """Finding cooccurrences between the set of entities with at least 
    one of the property-related keywords that happened for the first 
    time in a given year
    """

    chems = kwargs.get('chems', [])
    row_years = kwargs.get('row_years', [])
    return_papers = kwargs.get('return_papers', False)

    if len(chems)==0:
        msdb.crsr.execute('SELECT formula FROM chemical;')
        chems = np.array([x[0] for x in msdb.crsr.fetchall()])

    if len(row_years)==0:
        msdb.crsr.execute('SELECT YEAR(date) FROM paper;')
        row_years = np.array([x[0] for x in msdb.crsr.fetchall()])

        
    nA = 1739453
    nC = 107466
    nKW = R.shape[1] - nA - nC

    # entities unstudied in the previous years
    KW_pubs = np.sum(R[row_years < year,-nKW:], axis=1)
    C_pubs  = R[row_years < year,nA:nA+nC]
    unstudied_ents = np.asarray(np.sum(C_pubs.multiply(KW_pubs),axis=0)==0)[0,:]

    # entities studied this year
    KW_pubs = np.sum(R[row_years==year,-nKW:], axis=1)
    C_pubs  = R[row_years==year,nA:nA+nC]
    CKW_pubs = C_pubs.multiply(KW_pubs).tocsc()
    yr_studied_ents = np.asarray(np.sum(CKW_pubs,axis=0)>0)[0,:]

    new_studied_ents = chems[unstudied_ents * yr_studied_ents]
    
    if return_papers:
        # we explicitly need row indices associated with the discovery year, so
        # that discovery papers can be returned through their IDs 
        year_pids = np.where(row_years==year)[0]

        new_studied_ents_inds = np.where(unstudied_ents * yr_studied_ents)[0]
        new_studies_papers = {}
        for idx in new_studied_ents_inds:
            new_studies_papers[chems[idx]] = year_pids[
                np.where((CKW_pubs[:,idx]>0).toarray())[0]]

        return new_studied_ents, new_studies_papers
    else:
        return new_studied_ents 

def year_discoverers(R, year, **kwargs):

    kwargs['return_papers'] = True
    disc_ents, papers = year_discoveries(R, year, **kwargs)
    paper_ids = np.concatenate([pids for _,pids in papers.items()])
    discoverers = msdb.get_authors_by_paper_ids(paper_ids, cols=['author_id'])

    return np.concatenate([auids['author_id'] for _,auids in discoverers.items()])
    
     
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


def random_walk_seq(R, start_idx, L, lazy=True, node_weight_func=None):
    """Generating a random walk with a specific length and from 
    a starting point 
    """


    R = R.tocsc()
    seq = [start_idx]

    if not(lazy) and (np.sum(R[:,start_idx])==0):
        print("Non-lazy random walk cannot start from an isolated vertex.")
        return None
    
    v = start_idx
    for i in range(L-1):
        
        # selecting edge
        v_edges = R[:,v].indices
        edge_weights = R[:,v].data        
        eind = (edge_weights/edge_weights.sum()).cumsum().searchsorted(np.random.sample())
        e = v_edges[eind]

        # selecting a node inside e
        row = R[e,:].tocsr()
        e_nodes = row.indices
        
        if not(lazy):
            row[0,v]=0
            e_nodes = list(e_nodes)
            e_nodes.remove(v)
            e_nodes = np.array(e_nodes)
            # if no other node remains in the paper, exit
            if len(e_nodes)==0:
                return seq
            
            
        if node_weight_func is None:
            node_weights = row.data
            node_weights = node_weights/node_weights.sum()
        else:
            arr_row = np.float32(row.toarray())[0,:]
            node_weights = node_weight_func(arr_row)
            node_weights = node_weights[node_weights>0]

        nind = node_weights.cumsum().searchsorted(np.random.sample())
        v = e_nodes[nind]

        seq += [v]

    return seq


def gen_DeepWalk_sentences(R,
                           chem_to_author_ratio,
                           length,
                           size,
                           file_path=None,
                           logger=None):
    """Generating a sequence of random walks starting from the last column
    of the vertex weight matrix
    """

    nA = 1739453
    nC = 107466

    msdb.crsr.execute('SELECT formula FROM chemical;')
    chems = np.array([x[0] for x in msdb.crsr.fetchall()])
    
    f = lambda data: node_weighting_1(data, chem_to_author_ratio)

    sents = []
    for i in range(size):
        seq = random_walk_seq(R, R.shape[1]-1, length, lazy=False, node_weight_func=f)
        toks = ['a_{}'.format(s) if s<nA else
                (chems[s-nA] if nA<=s<nA+nC else 'thermoelectric') for s in seq]
        sent = ' '.join(toks) + '.'

        sents += [sent]

        if not(i%500) and i>0:
            if file_path is not None:
                with open(file_path, 'a') as tfile:
                    tfile.write('\n'.join(sents[i-500:i]))
                    nlines = i
                logger.info('{} randm walks are saved'.format(i))

    if file_path is not None:
        with open(file_path, 'a') as f:
            f.write('\n'.join(sents[nlines:]))

    return sents

def node_weighting_1(data, alpha):
    """Giving weights to existing nodes in a hyperedge  such that
    the probabiliy of choosing chemical nodes is alpha times the
    probability of choosing an author node in each random walk step
    """

    nA = 1739453
    nC = 107466
    
    A = np.sum(data[:nA]) + data[-1]  # assume data[-1]=KW
    C = np.sum(data[nA:nA+nC])
    if A>0 and C>0:
        data[:nA] = data[:nA] / ((alpha+1)*A)
        data[-1] = data[-1] / ((alpha+1)*A)
        data[nA:nA+nC] = alpha*data[nA:nA+nC] /  ((alpha+1)*C)
        
    return data

def bfs(R, start_idx, stopping_points=[]):
    """Breadth first search algorithm for strating from a 
    given point and returning its connected component

    If a set of stopping points are given, the BFS algorithm
    stops each time it encounters an index in stoppin_points (ignoring
    all its neighbors and children), and the output will
    be a subset of stopping points that are in the same 
    connected component
    """

    logger = helpers.set_up_logger(__name__, None,False)

    visited = [start_idx]
    dists = {start_idx: 0}
    final_dists = {start_idx: 0}
    Q = deque([start_idx])


    logcnt = 100
    while len(Q)>0:
        v = Q.popleft()
        N = find_neighbors(v,R)

        # consider only the unvisited points
        N = N[~np.isin(N,visited)]
        dists.update({u: dists[v]+1 for u in N})
        visited += list(N)
        
        stop_indic = np.isin(N, stopping_points)
        u_stops = N[stop_indic]
        u_nonstops = N[~stop_indic]
        
        # only add non-stopping points to the queue
        Q.extend(list(u_nonstops))
        
        # add stopping points to final_dists
        final_dists.update({u: dists[v]+1 for u in u_stops})
        
        if len(final_dists)>logcnt:
            logger.info('Length of added stops: {}, length of queue: {}'.format(len(final_dists), len(Q)))
            logcnt += 100

    if len(stopping_points)>0:
        return final_dists
    else:
        return dists
                    
