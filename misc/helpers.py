import os
import sys
import pdb
import copy
import json
import logging
import numpy as np
from tqdm import tqdm
from scipy import sparse
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from pymatgen.core.composition import Composition

ELEMENTS_PATH = '/home/jamshid/codes/data/GNN/data/elements.txt'


def set_up_logger(log_name, logfile_path, logger_disable, file_mode='w'):
    """Setting up handler of the "root" logger as the single main logger
    """
    
    logger = logging.getLogger(log_name)
    if logger_disable:
        handler = logging.NullHandler()
    elif logfile_path is None:
        handler = logging.StreamHandler()
    else:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename=logfile_path,
                                      encoding='utf-8',
                                      mode=file_mode)
    handler.setFormatter(logging.Formatter("%(asctime)s : %(levelname)s : %(message)s"))
    logger.handlers = []
    logger.addHandler(handler)

    return logger

def locate_array_in_array(moving_arr, fixed_arr):
    """For each overlapping element in moving_arr, find its location 
    index in fixed_arr

    The assumption is that moving the array is a subset of the fixed array
    """

    assert np.sum(np.isin(moving_arr, fixed_arr))==len(moving_arr), \
        'Moving array should be a subset of fixed array.'
    
    sinds = np.argsort(fixed_arr)
    locs_in_sorted = np.searchsorted(fixed_arr[sinds], moving_arr)

    return sinds[locs_in_sorted]


def find_studied_ents_VW(ents,VW,row_yrs,yr):
    """Generating entities that have been studied prior to the input 
    year based on a given vertex-weight matrix 

    Here, it is assumed that VW (vertex weight matrix) contains only
    the entity+property columns (i.e., author columns are excluded).
    """

    assert len(ents)==(VW.shape[1]-1), 'Number of columns in the vertex weight ' +\
        'matrix should equal the number of given entities.'

    assert len(row_yrs)==VW.shape[0], 'Number of rows in the vertex weight ' +\
        'matrix should equal the number of given years.'

    sub_VW = VW[row_yrs<yr,:]
    studied_bin = np.asarray(np.sum(sub_VW[:,:-1].multiply(sub_VW[:,-1]), axis=0)>0)[0,:]
    return ents[studied_bin]
    

def find_studied_ents_linksdict(file_or_dict,yr):
    """Generating entities that have been studied prior to the input 
    year (excluding that year) based on a dictionary of the form:
    {E1:Y1, E2:Y2, ...}
    where Ei's are the entities and Yi's are the corresponding years that the 
    relationship between Ei's and the property are obtained (curated).
    """

    if isinstance(file_or_dict,str):
        link_dict = json.load(open(file_or_dict,'r'))
    elif isinstance(file_or_dict, dict):
        link_dict = file_or_dict
    else:
        raise ValueError('The first input should be either a string (path ' +\
                         'to file) or a dictionary.')

    return np.array([x for x,y in link_dict.items() if y<yr])


def year_discoveries(ents, VW, row_yrs, year, return_papers=False):
    """Finding cooccurrences between the set of entities with at least 
    one of the property-related keywords that happened for the first 
    time in a given year
    """
        
    # entities unstudied in the previous years
    KW_pubs = VW[row_yrs < year,-1]
    E_pubs  = VW[row_yrs < year, :-1]
    unstudied_ents = np.asarray(np.sum(E_pubs.multiply(KW_pubs),axis=0)==0)[0,:]

    # entities studied this year
    KW_pubs = VW[row_yrs==year,-1]
    E_pubs  = VW[row_yrs==year,:-1]
    EKW_pubs = E_pubs.multiply(KW_pubs).tocsc()
    yr_studied_ents = np.asarray(np.sum(EKW_pubs,axis=0)>0)[0,:]

    new_studied_ents = ents[unstudied_ents * yr_studied_ents]
    
    if return_papers:
        # we explicitly need row indices associated with the discovery year, so
        # that discovery papers can be returned through their IDs 
        year_pids = np.where(row_yrs==year)[0]

        new_studied_ents_inds = np.where(unstudied_ents * yr_studied_ents)[0]
        new_studies_papers = {}
        for idx in new_studied_ents_inds:
            new_studies_papers[ents[idx]] = year_pids[
                np.where((EKW_pubs[:,idx]>0).toarray())[0]]

        return new_studied_ents, new_studies_papers
    else:
        return new_studied_ents 

    
def year_discoverers(ents, VW, row_yrs, year):

    disc_ents, papers = year_discoveries(ents, VW[:,-len(ents)-1:], row_yrs, year, True)
    paper_ids = np.concatenate([pids for _,pids in papers.items()])
    # extracting the authors
    auids = np.unique(VW[paper_ids,:-len(ents)-1].tocsr().indices)

    return auids


def gt_discoveries(ents,VW,row_yrs,constraint_func=None):
    """Generating ground truth discoveries in a given year

    VM corresponds only to that part of vertex-weight matrix that 
    contains the entity columns
    """

    assert len(ents)==(VW.shape[1]-1), 'Number of columns in the vertex weight ' +\
                'matrix should equal the number of given entities.'

    assert len(row_yrs)==VW.shape[0], 'Number of rows in the vertex weight ' +\
        'matrix should equal the number of given years.'

    def gt_disc_func(year_of_gt):

        sub_VW = VW[row_yrs==year_of_gt,:]
        studied_bin = np.asarray(np.sum(sub_VW[:,:-1].multiply(sub_VW[:,-1]), axis=0)>0)[0,:]
        all_studied_ents = ents[studied_bin]
        # remove already studied ones
        prev_studied_ents = find_studied_ents_VW(ents,VW,row_yrs,year_of_gt)

        disc_ents = all_studied_ents[~np.isin(all_studied_ents,prev_studied_ents)]

        if constraint_func is not None:
            disc_ents = constraint_func(disc_ents)
            
        return disc_ents

    return gt_disc_func


def gt_discoveries_4CTD(disease):

    ds_dr_path = '/home/jamshid/codes/data/CTD/diseases_drugs.json'
    target_ds_dr = json.load(open(ds_dr_path, 'r'))
    rel_drugs = target_ds_dr[disease]

    def gt_disc_func(yr):
        gt = np.array([x.lower() for x,y in rel_drugs.items() if int(y)==yr])
        gt = np.array([x.replace(' ','_') for x in gt])
        return gt

    return gt_disc_func


def gt_discoverers(ents, VW, row_yrs, **kwargs):
    """Ground-truth discoverer function

    Here the input vertex-weight matrix `VW` includes the whole columns, that's
    why we also need the number of authors `nA`
    """

    assert len(row_yrs)==VW.shape[0], 'Number of rows in the vertex weight ' +\
        'matrix should equal the number of given years.'
    
    def gt_discoverers_func(year_of_pred):
        auids = year_discoverers(ents, VW,row_yrs,year_of_pred)
        return auids

    return gt_discoverers_func



def prune_deepwalk_sentences(sents, remove='author'):

    # removing authors or chemicals
    if remove=='author':
        hl = [[s for s in h.split(' ') if 'a_' not in s] for h in sents]
    elif remove=='chemical':
        hl = [[s for s in h.split(' ') if ('a_' in s) or ('thermoelectric' in s)]
              for h in sents]
    elif remove=='author_affiliation':
        hl = [[s for s in h.split(' ') if '_' not in s] for h in sents]

    # rejoining the split terms and ignoring those with singular terms
    hl = [' '.join(h) for h in hl if len(h)>1]

    # removing dots
    hl = [h.split('.')[0] for h in hl]

    # removing those sentences only containing the keyword
    hl = [h for h in hl if len(np.unique(h.split(' ')))>1]

    return hl


def random_walk_from_transprob(transprob,
                               start_idx,
                               L):
    """Doing a short random walk with length `L` for regular graphs using a 
    pre-computed transition probability matrix
    """

    nodes = [start_idx]
    randgen = np.random.sample
    for i in range(L-1):
        try:
            if transprob[nodes[-1],:].nnz==0:
                return np.array(nodes)
            pmf = np.array(transprob[nodes[-1],:].todense())[0,:]
        except:
            pdb.set_trace()
        rnd = randgen()
        try:
            cdf = pmf.cumsum()
            cdf[-1] = 1.
            nodes += [cdf.searchsorted(rnd)]
        except:
            pdb.set_trace()

    return  np.array(nodes)


def transprob_from_scores(sims_mat,SD_mat,beta,scale=10):

    transprobs = sparse.lil_matrix(sims_mat.shape,dtype=np.float32)
    sims_mat = sims_mat.tocsr()
    SD_mat = SD_mat.tocsr()
    for i in range(sims_mat.shape[0]):
        inds = sims_mat[i,:].indices
        dat = np.array(sims_mat[i,:].todense())[0,:]

        # bringing cosines to [0,1]
        dat[inds] = (dat[inds]+1)/2

        # linear combination with SD scores
        dat[inds] = dat[inds] - beta*np.array(SD_mat[i,inds].todense())[0,:]

        # passing through a scaled sigmoid to make them probabilities
        if np.sum(np.exp(scale*dat[inds]))==0:
            continue
        probs = np.exp(scale*dat[inds]) / np.sum(np.exp(scale*dat[inds]))

        rows = np.ones(len(inds))*i
        transprobs[rows,inds] = probs

    return transprobs


def atomic_features(path_to_elements=None):

    if path_to_elements is None:
        path_to_elements=ELEMENTS_PATH
    E = np.array(open(ELEMENTS_PATH, 'r').read().splitlines())

    def give_features(frml):
        f = np.zeros(len(E))
        comp = Composition(frml)
        for el,cnt in comp.element_composition.as_dict().items():
            f[E==el] = cnt
        return f

    return give_features


def unadjusted_words2sents(sents, w2v_model, alpha):
    """Computing unadjusted sentence embeddings of in a set
    of documents using the smoothened weighted average of tokens'
    embeddings
    """

    stop_tokens = ['(' ,')', '/', '\\', '?', '!', '+', '.', ',', ':',
                   ';', '-', '_', '&', '$', '~', '^', '\#', '@', '*',
                   '<nUm>', '\'', '\"']

    #n_jobs = min(20, cpu_count()-4)
    #parallel_processor = Parallel(n_jobs=n_jobs)

    def collect_embedding_one_abstract(i):
        tokens_list = [x.split(' ') for x in sents[i].split(' . ')]
        tokens_list = [np.array([y for y in x if y in w2v_model.wv.vocab])
                       for x in tokens_list]
        tokens_list = [x[~np.isin(x,stop_tokens)] for x in tokens_list]
        vectors_list = [np.stack([w2v_model.wv[tok]*alpha/(alpha+w2v_model.wv.vocab[tok].count) for tok in x], axis=0).mean(axis=0, keepdims=True)
                        for x in tokens_list if len(x)>0]
        vectors = np.concatenate(vectors_list, axis=0)

        return vectors
    
    #pbar_1 = tqdm(range(len(sents)), position=0, leave=True)
    #pbar_1.set_description('Grouping word vectors of sentences based on abstracts')
    #abstract_vectors = parallel_processor(delayed(process_one_abstract)(i)
    #                                      for i in pbar_1)

    pbar = tqdm(range(len(sents)), position=0, leave=True)
    pbar.set_description('Weighted Averaging')
    abstract_vectors = []
    for i in range(len(sents)):
        abstract_vectors += [collect_embedding_one_abstract(i)]
        pbar.update(1)
    pbar.close()

    return abstract_vectors


def extract_chems_from_deepwalks(path_or_sents):
    """Extracting chemical terms of a set of deepwalk sentences,
    assuming that the deepwalks have been generated starting from a 
    single keyword node

    *Returns:*

    * unique values in the deepwalk sentences (excluding the keyword term)
    * counts of the unique values
    """

    if isinstance(path_or_sents, str):
        with open(path_or_sents, 'r') as f:
            sents = f.read().splitlines()
    else:
        sents = path_or_sents

    sents = prune_deepwalk_sentences(sents)
    kw = sents[0].split(' ')[0]
    chems = ' '.join(sents)
    chems = chems.replace(kw+' ', '')
    chems = chems.split(' ')

    return np.unique(chems, return_counts=True)



def lighten_color(color, amount=0.5):
    """
    Downloaded
    -----------
    This piece of code is downloaded from
    https://stackoverflow.com/a/49601444/8802212

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



class RestartableIterator(object):
    def __init__(self, n, size, arr=None, fill=False):
        self.n = n
        self.size = size
        self.array = arr
        self.fill = fill
        self._randinds = np.random.permutation(self.n)
        self.reset()

    def next(self):
        next_chunk = self.array[self._chunks.pop(0)] if self.array is not None else self._chunks.pop(0)
        if len(self._chunks)==0:
            self.reset()
        return next_chunk

    def reset(self):
        self._chunks = [self._randinds[x:x+self.size]
                        for x in range(0,len(self._randinds),self.size)]
        self._randinds = np.random.permutation(self.n)

        # if all chunks need to have the same size, fill the last item
        # with random indices that is generated for the next iteration
        pdb.set_trace()
        if self.fill:
            if len(self._chunks[-1])<self.size:
                # avoiding repitition when filling the last chunk
                filling_arr = self._randinds[~np.isin(self._randinds, self._chunks[-1])][
                    :self.size-len(self._chunks[-1])]
                self._chunks[-1] = np.append(self._chunks[-1], filling_arr)
                # remove the selected fillig array from the new random indices
                self._randinds = self._randinds[~np.isin(self._randinds, filling_arr)]
