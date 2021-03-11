import os
import sys
import pdb
import json
import random
import logging
import pymysql
import numpy as np
import networkx as nx
from scipy import sparse
from collections import deque
from multiprocessing import Pool, cpu_count

from gensim.models import Word2Vec

path = '/home/jamshid/codes/social-knowledge-analysis'
sys.path.insert(0, path)

from data import readers
from misc import helpers

config_path = '/home/jamshid/codes/data/sql_config_0.json'
msdb = readers.DB(config_path,
                  db_name='msdb',
                  entity_tab='chemical',
                  entity_col='formula')


logger = logging.getLogger()
logger.handlers = [logging.StreamHandler()]
logger.setLevel(logging.INFO)


def compute_vertex_matrix(db, **kwargs):
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

    nP = db.count_table_rows('paper')
    Pids = db.get_1d_query('SELECT id FROM paper;')
    nA = db.count_table_rows('author')
    Aids = db.get_1d_query('SELECT id FROM author;')
    nE = db.count_table_rows(db.entity_tab)
    Eids = db.get_1d_query('SELECT id FROM {};'.format(db.entity_tab))
    logger.info('#papers={}, #author={}, #entities={}'.format(nP,nA,nE))
    
    VM = sparse.lil_matrix((nP,nA+nE), dtype=np.uint8)
    # filling the matrix with batches
    cnt = 0
    batch_size = 500
    logger.info('Starting to fill the vertex matrix with batche size {}'.format(batch_size))
    while cnt<nP:
        inds = np.arange(cnt, min(cnt + batch_size, nP))
        batch_Pids = Pids[inds]
        q_Aids = db.get_LoA_by_PID(batch_Pids)
        q_Eids = db.get_LoE_by_PID(batch_Pids)

        cols = []
        rows = []
        for i,pid in enumerate(batch_Pids):
            # each PID has a number of authors and entities;
            # locate them in the global array of author and entity IDs;
            # these locations would be their rows in vertex matrix
            au_cols  = np.where(np.isin(Aids, q_Aids[pid]['id']))[0] if pid in q_Aids else []
            ent_cols = np.where(np.isin(Eids, q_Eids[pid]['id']))[0]+nA if pid in q_Eids else []
            
            cols += [np.concatenate((au_cols, ent_cols))]
            rows += [inds[i]*np.ones(len(au_cols)+len(ent_cols))]

        cols = np.concatenate(cols)
            
        rows = np.concatenate(rows)
        VM[rows,cols] = 1

        cnt += batch_size

        if not(cnt%100000):
            logger.info('{} articles have been processed'.format(cnt))
            if not(cnt%10000) and (savefile_path is not None):
                sparse.save_npz(savefile_path, VM.tocsc())

    return VM

def compute_vertex_aff_submatrix(Aff2Pid=None, **kwargs):
    """Computing vertex weight matrix for hypernodes corresponding to author
    affiliations (with all papers--hyperedges included)

    If Aff2Pid dictionary is not given, it should be formed first. When doing that,
    note that the variable group_concat_max_len is increased from its default value (1024)
    by executing 'SET SESSION group_concat_max_len=1000000;'
    """

    if Aff2Pid is None:
        scomm = 'SELECT A2A.aff_id, GROUP_CONCAT(DISTINCT(P.paper_id)) FROM paper P' \
                'INNER JOIN paper_author_mapping P2A ON P2A.paper_id=P.paper_id' \
                'INNER JOIN author_affiliation_mapping A2A ON A2A.author_id=P2A.author_id' \
                'GROUP BY A2A.aff_id ORDER BY A2A.aff_id;'
        msdb.crsr.execute(scomm)
        Aff,Pids = zip(*msdb.crsr.fetchall())
        Aff2Pid = {Aff[i]: np.array([int(x) for x in Pids[i].split(',')])
                   for i in range(len(Aff))}
    
    nP = 1507143
    nAff = len(Aff2Pid)
    
    VM = sparse.lil_matrix((nP,nAff), dtype=np.uint8)
    
    cols = np.concatenate([np.ones(len(Aff2Pid[i]))*i for i in range(nAff)])
    rows = np.concatenate([Aff2Pid[i] for i in range(nAff)])
    VM[rows, cols] = 1
    
    return VM

def compute_vertex_KW_submatrix(db, los, **kwargs):
    """Forming a submatrix corresponding to conceptual nodes
    given as a set of keywords priveded in `los` (list of strings) arguments
    """

    case_sensitives = kwargs.get('case_sensitives', [])
    
    nP = db.count_table_rows('paper')
    ncols = len(los)
    VM = sparse.lil_matrix((nP,ncols), dtype=np.uint8)
    for i, kw in enumerate(los):
        if kw in case_sensitives:
            cs = [kw]
        else:
            cs = []

        R = db.get_papers_by_keywords([kw], case_sensitives=cs)
        rows = R['id']
        cols = i*np.ones(len(rows))

        VM[rows, cols] = 1

    return VM


def find_neighbors(idx, R):
    """Returning neighbors of a node indexed by `idx`

    NOTE: input `idx` can be an array of indices
    """

    # indices of the hyperedges (there might be repeated hyperedges
    # here, if idx is an array, but we don't care since the final
    # result is distinct values of the column list)
    he_inds = R[:,idx].nonzero()[0]

    return np.unique(R[he_inds,:].nonzero()[1])

def find_authors(idx, GR, nA, coauthor_deg=1, separate=False):
    """Finding coauthors of a given author/entity specified by its
    node index

    `GR` is either the networkx graph object, or the sparse vertex-weight matrix

    `coauthor_deg` determines the degree of the co-authorship to be considered
    when finding the common authors. For example, degree=1 implies only the common 
    authors who have papers on both entities; degree=2 
    implies considering co-authors of the common authors too; degree=3 considers
    co-authors of co-authors of the common authors too; so on.
    """

    assert coauthor_deg>=1, "The co-authorship should be greater than or equal to one."

    if isinstance(GR, nx.classes.graph.Graph):
        nfinder = lambda x: np.array(list(GR.neighbors(x))) \
            if not(isinstance(x,(list,np.ndarray))) \
            else np.concatenate([list(GR.neighbors(xi)) for xi in x])
    else:
        nfinder = lambda x: find_neighbors(x, GR)
    
    authors = {}
    
    nbrs = nfinder(idx)
    
    if separate:
        prev_authors = authors[1] = nbrs[nbrs<nA]
        for i in range(1,coauthor_deg):
            nbrs = nfinder(authors[i])
            anbrs = nbrs[nbrs<nA]
            authors[i+1] = anbrs[~np.isin(anbrs,prev_authors)]
            prev_authors = np.concatenate([prev_authors,anbrs])

    else:
        authors = nbrs[nbrs<nA]
        for i in range(1,coauthor_deg):
            nbrs = nfinder(authors)
            if not(np.any(nbrs<nA)): break
            authors = np.unique(np.concatenate((authors, nbrs[nbrs<nA])))

    return authors
    
     
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
    msdb.crsr.execute('SELECT id FROM paper WHERE \
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

    #      edge sel.   node sel.
    #        prob.      prob.
    #      ---------   -------
    return iDV * R.T * iDE * R


def compute_transprob_alpha(R, alpha, nA):
    """Computing alpha-balanced transition probability, where
    the edge selection process is the same as above, but the node 
    selection process is modified such that the probability of selecting
    non-author nodes is alpha times the probability of selecting authors
    """

    # same procedure for edge selection 
    row_collapse = np.array(np.sum(R,axis=0))[0,:]
    iDV = np.zeros(len(row_collapse), dtype=float)
    iDV[row_collapse>0] = 1./row_collapse[row_collapse>0]
    iDV = sparse.diags(iDV, format='csr')

    # the node selection is different than above
    A_col_collapse = np.array(np.sum(R[:,:nA],axis=1))[:,0]
    A_iDE = np.zeros(len(A_col_collapse), dtype=float)
    A_iDE[A_col_collapse>0] = 1/A_col_collapse[A_col_collapse>0]/(alpha+1)
    E_col_collapse = np.array(np.sum(R[:,nA:],axis=1))[:,0]
    E_iDE = np.zeros(len(E_col_collapse), dtype=float)
    E_iDE[E_col_collapse>0] = alpha/E_col_collapse[E_col_collapse>0]/(alpha+1)

    # taking care of paper when there are only authors or non-authors;
    # for these cases, use the original method
    col_collapse = np.array(np.sum(R,axis=1))[:,0]
    A_iDE[(E_iDE==0)*(col_collapse>0)] = 1./A_iDE[(E_iDE==0)*(col_collapse>0)]
    E_iDE[(A_iDE==0)*(col_collapse>0)] = 1./E_iDE[(A_iDE==0)*(col_collapse>0)]
    A_iDE = sparse.diags(A_iDE, format='csr')
    E_iDE = sparse.diags(E_iDE, format='csr')

    # finally, computing the matrix with node selection probabilities
    # and fill in the blocks corresponding the author and entity columns
    N1 = A_iDE*R[:,:nA]
    N2 = E_iDE*R[:,nA:]
    node_prob = sparse.hstack((N1,N2),'csr')

    return iDV * R.T * node_prob
    


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

    
def compute_multistep_transprob_memeff(P, R, source_inds, dest_inds, **kwargs):
    """Computing multi-step transition probabilities in a more
    memory-efficient method than the classical algorithm

    """

    interm_inds = kwargs.get('interm_inds', None)
    nstep = kwargs.get('nstep', 1)

    # use the classic method for steps shorter than three
    if interm_inds is None:
        interm_inds = np.arange(R.shape[1])

    if nstep<2:
        return compute_multistep_transprob(P, source_inds, dest_inds,
                                           interm_inds=interm_inds,
                                           nstep=nstep)
    else:

        # initializing the variables for the loop
        nbrs = find_neighbors(source_inds, R)
        if interm_inds is not None:
            nbrs = nbrs[np.isin(nbrs, interm_inds)]
        current_mat = P[source_inds,:][:,nbrs]
        next_nbrs = nbrs
        
        # repeat ...
        for t in range(nstep-1):
            if t==nstep-2:
                next_nbrs = dest_inds
            else:
                next_nbrs = find_neighbors(next_nbrs, R)
                if interm_inds is not None:
                    next_nbrs = next_nbrs[np.isin(next_nbrs,interm_inds)]

            # update the transition matrix and neighbors
            #pdb.set_trace()
            next_mat = P[nbrs,:][:,next_nbrs]
            current_mat = current_mat * next_mat
            nbrs = next_nbrs

    return current_mat


def random_walk_seq(R, start_idx, L,
                    lazy=True,
                    node_weight_func=None,
                    node2vec_q=None,
                    rand_seed=None):
    """Generating a random walk with a specific length and from 
    a starting point 
    """


    R = R.tocsc()
    seq = [start_idx]       # set of hyper-nodes
    eseq = []               # set of hyper-edges

    if not(lazy) and (np.sum(R[:,start_idx])==0):
        print("Non-lazy random walk cannot start from an isolated vertex.")
        return None

    if rand_seed is not None:
        randgen = np.random.RandomState(rand_seed).random
    else:
        randgen = np.random.sample

    if node2vec_q is not None:
        q = node2vec_q
        prev_idx = None    # previous (hyper)node 

    v = start_idx
    for i in range(L-1):

        # selecting edge
        if node2vec_q is not None:
            e = node2vec_sample_edge(R, v, prev_idx, q, randgen)
            prev_idx = v   # update previous node
        else:
            v_edges = R[:,v].indices
            edge_weights = R[:,v].data   # this is an np.array
            eind = (edge_weights/edge_weights.sum()).cumsum().searchsorted(randgen())
            e = v_edges[eind]
        
        eseq += [e]

        # selecting a node inside e
        row = np.float32(np.squeeze(R[e,:].toarray()))
        
        if not(lazy):
            row[v]=0
        if ~np.any(row>0):
            return seq, eseq
            
        if node_weight_func is None:
            e_nodes = np.where(row>0)[0]
            node_weights = row[row>0]
            node_weights = node_weights/node_weights.sum()
        else:
            # here, we get the edge-nodes (e_nodes) after applying
            # the weighting function, since it might change the values
            # of the node probabilities
            node_weights = node_weight_func(row)
            if ~np.any(node_weights>0):
                return seq, eseq
            e_nodes = np.where(node_weights>0)[0]
            node_weights = node_weights[node_weights>0]

        CSW = node_weights.cumsum()
        if CSW[-1]<1.: CSW[-1]=1.
        nind = CSW.searchsorted(randgen())
        v = e_nodes[nind]

        seq += [v]

    return seq, eseq
    

def gen_DeepWalk_sentences_fromKW(R,
                                  db,
                                  ratio,
                                  length,
                                  size,
                                  keyword,
                                  node2vec_q=None,
                                  block_types=[],
                                  file_path=None,
                                  eseq_file_path = None,
                                  rand_seed=None,
                                  logger=None):
    """Generating a sequence of random walks starting from the last column
    of the vertex weight matrix

    Input argument block_types specifies type of the "column blocks" in the vertex
    matrix, with format ((B1,n1), (B2,n2),...), where Bi and ni are the i-th block and
    its size. It is assumed that these blocks are grouped in the same order as in
    this variable(they are not shuffled).

    Input `ratio` is either a scalar that determines the ratio between the probability of 
    choosing a chemical node to the probability of author selection (if two types
    of nodes are present), or an array-line that determines mixture coefficients
    corresponding to various groups of nodes (if multiples types of nodes are present)

    The argument `block_types` determines groups of columns that exist in the given
    vertex matrix R. It should be given as a dictionary with a format like the following:
    {'author': nA, 'entity': nE}, where nA and nE are the number of author nodes and
    entity nodes, respectively.
    """

    ents = db.get_1d_query('SELECT {} FROM {};'.format(db.entity_col, db.entity_tab))

    if len(block_types)==0:
        nA = db.count_table_rows('author')
        nE = db.count_table_rows(db.entity_tab)
        block_types = {'author': nA, 'entity': nE}
        type_ranges = {'author': [0,nA], 'entity': [nA,nA+nE]}
    else:
        assert np.sum([v[1] for v in block_types])==R.shape[1]-1, \
            'Sum of sizes in block_types should be equal to the number of columns in R.'
        cnt = 0
        type_ranges = {}
        for k,v in block_types:
            type_ranges[k] = [cnt, cnt+v]
            cnt += v

    # function for translating a selected node in random walk to a
    # meaningful string 
    def translate_entry(idx):
        for k,v in type_ranges.items():
            if v[0] <= idx < v[1]:
                if k=='author':
                    return 'a_{}'.format(idx-v[0])
                elif k=='affiliation':
                    return 'aff_{}'.format(idx-v[0])
                elif k=='entity':
                    return ents[idx-v[0]]
                
        # if the entry does not belong to any of the ranges --> KW
        return keyword

    
    if ratio is None:
        f = None
    elif np.isscalar(ratio):
        if 0 < ratio < np.inf:
            f = lambda data: node_weighting_alpha(data, ratio, block_types)
        elif ratio==np.inf:
            f = lambda data: node_weighting_ent(data, block_types)
        elif ratio==0:
            f = lambda data: node_weighting_author(data, block_types)
    else:
        assert len(block_types)>2, 'Having array-like ratio is only for multiple types of nodes'
        f = lambda data: node_weighting_waff(data, ratio)

    increments = None
    if rand_seed is not None:
        increments = np.arange(100,size*100+1,size)
        np.random.shuffle(increments)

    sents = []
    eseqs_list = []
    nlines=0
    for i in range(size):
        seq, eseq = random_walk_seq(R, R.shape[1]-1, length,
                                    lazy=False,
                                    node_weight_func=f,
                                    node2vec_q=node2vec_q,
                                    rand_seed=None if rand_seed is None else rand_seed+increments[i])
        eseqs_list += [' '.join([str(x) for x in eseq])]

        # parsing the hyper nodes
        toks = [translate_entry(s) for s in seq]
        sent = ' '.join(toks)

        sents += [sent]

        if not(i%500) and i>0:
            if file_path is not None:
                with open(file_path, 'a') as tfile:
                    tfile.write('\n'.join(sents[i-500:i])+'\n')
                    nlines = i
            if eseq_file_path:
                with open(eseq_file_path, 'a') as tfile:
                    tfile.write('\n'.join(eseqs_list[i-500:i])+'\n')
                    nlines = i
            if logger is not None:
                logger.info('{} randm walks are saved'.format(i))

    if file_path is not None:
        with open(file_path, 'a') as f:
            f.write('\n'.join(sents[nlines:])+'\n')
    if eseq_file_path is not None:
        with open(eseq_file_path, 'a') as f:
            f.write('\n'.join(eseqs_list[nlines:])+'\n')

            
    return sents, eseqs_list


def node2vec_sample_edge(R, curr_idx, prev_idx, q, randgen):
    """Sampling an edge in a node2vec style, starting from
    a current node `curr_idx` and given the previous node `prev_idx`
    with `p` and `q` the return and in-out parameters, respectively
    """

    N0 = R[:,curr_idx].indices
    
    # regular sampling in the first step
    if prev_idx is None:
        edge_weights = R[:,curr_idx].data   # this is an np.array
    else:
        # see which papers in N0 include previous node too (N0 intersect. N_{-1})
        edge_weights = np.ones(len(N0))
        N1 = R[:,prev_idx].indices
        edge_weights[np.isin(N0,N1)] = 1      # d_tx=1 in node2vec
        edge_weights[~np.isin(N0,N1)] = 1/q   # d_tx=2 in node2vec

    eind = (edge_weights/edge_weights.sum()).cumsum().searchsorted(randgen())
    e = N0[eind]
    
    return e
    
    
def node_weighting_ent(data, block_types):
    """Weighting nodes such that only chemicals are sampled; if there is
    no chemical is selected among the nodes, an all-zero vector will be returned 
    (i.e., random walk will be terminated)

    *Paramters*:

    ** data: array
       one row of the vertex matri

    ** block_types: dict
       types and size of each block of columns in the vertex matrix

    """

    nA = block_types['author']
    data[:nA] = 0
    if np.any(data>0):
        data = data/np.sum(data)

    return data

def node_weighting_waff(data, pies):
    """Weighting nodes in different groups 

    Group indices are hard-coded in this function, make them
    variable if needed (also the keyword noded is counted as a 
    chemical here)

    Here, we also assume that `data` is a 1D binary vector (values
    are either zero or one)
    """
    
    if ~np.any(data>0):
        return data
    
    assert np.sum(pies)==1., 'Mixture coefficients (pies) should sum to one'

    pies = np.array(pies)
    
    nA = 1739453
    nC = 107466
    nAff = 121267
    
    # renormalization
    GNNZ = np.array([np.sum(data[:nA]>0) + (data[-1]>0),
                     np.sum(data[nA:nA+nC]),
                     np.sum(data[nA+nC:-1]>0)])
    pies = pies / np.sum(pies[GNNZ>0])

    if GNNZ[0]>0:
        data[:nA] = data[:nA] * pies[0]/(np.sum(data[:nA]>0)+(data[-1]>0))
        data[-1] = data[-1] * pies[0]/(np.sum(data[:nA]>0)+(data[-1]>0))
    if GNNZ[1]>0:
        data[nA:nA+nC] = data[nA:nA+nC] * pies[1]/np.sum(data[nA:nA+nC]>0)
    if GNNZ[2]>0:
        data[nA+nC:-1] = data[nA+nC:-1] * pies[2]/np.sum(data[nA+nC:-1]>0)
        
    return data
    
    
def node_weighting_author(data, block_types):
    """Similar to node_weighting_chems but for authors

    *Paramters*:

    ** data: array
       one row of the vertex matri

    ** block_types: dict
       types and size of each block of columns in the vertex matrix
    """

    nA = block_types['author']
    data[nA:] = 0
    if np.any(data>0):
        data = data/np.sum(data)
        
    return data


def node_weighting_alpha(data, alpha, block_types):
    """Giving weights to existing nodes in a hyperedge  such that
    the probabiliy of choosing chemical nodes is alpha times the
    probability of choosing an author node in each random walk step

    *Parameters*:

    ** data: array
       one row of the vertex matri

    ** alpha: scalar
       P(sampling from entities) / P(sampling from authors)

    ** block_types: dict
       types and size of each block of columns in the vertex matrix (here,
       it should have only two types with keys `authors` and `entity`
    """

    assert len(block_types)==2, "node-weighting with alpha should be used " \
        "only with two types of vertex nodes, here we have {}".format(len(block_types))
    
    nA = block_types['author']
    nE = block_types['entity']
    
    A = np.sum(data[:nA]) + data[-1]  # assume data[-1]=KW
    E = np.sum(data[nA:nA+nE])
    if A>0 and E>0:
        data[:nA] = data[:nA] / ((alpha+1)*A)
        data[-1] = data[-1] / ((alpha+1)*A)
        data[nA:nA+nE] = alpha*data[nA:nA+nE] /  ((alpha+1)*E)
    elif A>0 and E==0:
        data[:nA] = data[:nA]/A
    elif A==0 and E>0:
        data[nA:nA+nE] = data[nA:nA+nE]/E
        
    return data


def random_chem_select(E, P2C_dict):
    """Randomly selecting chemicals from a set of given papers
    (hyperedges)

    *Parameters:*

    * E: a list of strings, each of which contains a set of paper IDs
    * P2C_dict: dictionary mapping each paper ID to the set of chemicals 
      that it contains
    """

    pids = [[int(x) for x in e.split(' ')] for e in E]
    pids_chems = [[[] if x not in P2C_dict else P2C_dict[x] for x in pid]
                  for pid in pids]
    #choices = [['' if len(x)==0 else random.choice(x) for x in pchems]
    #           for pchems in pids_chems]

    choices = []
    for j,pchems in enumerate(pids_chems):
        seq_choices = []
        for i,x in enumerate(pchems):
            if i==0:
                rem_chems = x
            elif pids[j][i-1]==pids[j][i]:
                rem_chems = list(filter(lambda xx: xx != seq_choices[i-1], rem_chems))
            elif pids[j][i-1]!=pids[j][i]:
                rem_chems = x
                
            if len(rem_chems)==0:
                seq_choices += ['']
            else:
                seq_choices += [random.choice(rem_chems)]
        choices += [seq_choices]
    
    # removing empty sets
    choices = [list(filter(lambda x:x!='', choice)) for choice in choices]
    # making strings
    choices = [' '.join(choice) for choice in choices]

    return choices


def compute_av_first_passage_distance(sents, node_1, nodes_2,
                                      nworkers=None,
                                      return_indiv_dists=False):
    """Computing average-first-passage distance metric from a given 
    node (`node_1`) and a set of other nodes (`nodes_2`)

    The assumption here is that all the sentences starat with the source node 
    (`node_1`) hence it always occurs at least once before the occurences of any
    given target node (those in `nodes_2`)
    """

    # S: subset of nodes_2
    def f(S):
        
        dists_dict = {x:[] for x in S}
        for ii,sent in enumerate(sents):
            toks = np.array(sent.split(' '))
            locs1 = np.where(toks==node_1)[0]

            for tarnode in S:
                if tarnode not in toks: continue

                locs2 = np.where(toks==tarnode)[0]

                # distances of all occurrences of node_2 to
                # node_1 in this sentence: for each occurrence of
                # node_2, the distance is defined as the number of steps
                # taken from the "last node_1 occurred before it".
                pdists = [x-locs1 for x in locs2]

                dists_to1 = {x: [] for x in range(len(locs1))}
                for i,pd in enumerate(pdists):
                    dists_to1[np.max(np.where(pd>0)[0])] += [np.min(pd[pd>0])]

                dists_dict[tarnode] += [np.min(x) for x in dists_to1.values() if len(x)>0]

            #if not((ii/len(sents))%0.25):
            #    logger.info('%{} is completed..'.format(100*ii/len(sents)))

        return dists_dict

    if nworkers is None:
        dists_dict = f(nodes_2)
    else:
        # THIS CANNOT BE DONE FOR NOW
        # since a local function cannot be pickled by parallelizer
        dists_dict = {}
        chunksize = int(len(nodes_2)/nworkers)
        with Pool(nworkers) as pool:
            for res in pool.imap(f, nodes_2, chunksize):
                dists_dict.update(res)

    
    dists = np.array([np.mean(dists_dict[x]) if len(dists_dict[x])>0 else np.nan
                      for x in nodes_2])
    #sims = np.array([np.sum([1./x for x in dists_dict[y]]) if len(dists_dict[y])>0 else np.nan
    #                 for y in nodes_2])

    if return_indiv_dists:
        return dists, dists_dict
    else:
        return dists
            
