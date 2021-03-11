import os
import os.path as osp
import pdb
import sys
import numpy as np
from tqdm import tqdm
from scipy import sparse
import scipy.sparse as sp

from gensim.models import Word2Vec
from sklearn.utils.extmath import randomized_svd

import torch
from torch_sparse import coalesce
from torch.distributions import categorical
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  NeighborSampler, extract_zip)

path = '/home/jamshid/codes/social-knowledge-analysis/'
sys.path.insert(0,path)
from misc.helpers import (atomic_features, gt_discoveries,
                          find_studied_ents_VW, unadjusted_words2sents)


class DataSet(InMemoryDataset):

    def __init__(self, root,
                 path_to_VM,
                 path_to_VMkw,
                 path_to_yrs,
                 yr, memory=5,
                 path_to_ents=None,
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.year = yr
        self.memory = memory
        self.path_to_VM = path_to_VM
        self.path_to_VMkw = path_to_VMkw
        self.path_to_yrs = path_to_yrs
        self.path_to_ents = path_to_ents
        super(DataSet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return []
        
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass


    def process(self):

        VM = sparse.load_npz(self.path_to_VM)
        VMkw = sparse.load_npz(self.path_to_VMkw)
        R = sparse.hstack((VM,VMkw))
        N0 = R.shape[1]

        row_years = np.loadtxt(self.path_to_yrs)
        
        R = R[(row_years>=self.year-self.memory)*(row_years<=self.year-1),:]
        
        # keeping only the nodes that are not isolated after filtering
        # the papers (hyperedges)
        non_isolated_nodes = np.unique(R.nonzero()[1])
        R = R[:,non_isolated_nodes]
        
        adj = R.T * R
        adj = adj.tocoo()

        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, _ = coalesce(edge_index, None, adj.shape[0], adj.shape[0])

        x = torch.from_numpy(np.ones(adj.shape[0])).to(torch.float)
        y = torch.from_numpy(non_isolated_nodes).to(torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

        

class GNNdata(DataSet):

    def __init__(self, feature_type, **kwargs):
        # popping feature-related inputs from the
        # keyword arguments (if any)
        self.fetch_features = kwargs.pop('fetch_features', False)
        self.prop = kwargs.pop('prop', None)                   # for w2v only
        self.alpha = kwargs.pop('alpha', None)                 # for w2v only
        self.path_to_w2v = kwargs.pop('path_to_w2v', None)     # for w2v only
        self.path_to_sents = kwargs.pop('path_to_sents', None) # for w2v only
        super(GNNdata, self).__init__(**kwargs)
        self.edge_index = self.data.edge_index
        self.n_x = len(self.data.y)
        self.selected_inds = self.data.y.numpy()
        self.feature_type = feature_type
        # no need for `data` created by the parent class 
        delattr(self, 'data')
        
        if self.feature_type=='atomic':
            self.dim = 121
        elif self.feature_type=='w2v':
            self.model = Word2Vec.load(self.path_to_w2v)
            self.dim = self.model.vector_size
        elif self.feature_type=='indicator':
            self.dim=self.n_x

        # load stuff like hypergraph incidence matrix, list of entities, etc
        self._load_hypergraph_materials()
        if self.fetch_features:
            self._fetch_node_features()

        
    def sample_positives(self):
        """TBA
        """
        pass


    def load_positive_pairs(self, path, include_nodes=None):
        """Loading positive samples that have been already computed
        as pairs of (x_target, x_nbr)
        """

        pos_samples = open(path, 'r').read().splitlines()
        pos_samples = [[int(x) for x in y.split(',')]
                       if len(y)>0 else [] for y in pos_samples]

        if include_nodes is None:
            pos_pairs = [[x,y] for x,line in enumerate(pos_samples) for y in line]
        else:
            pos_pairs = [[x,y] for x,line in enumerate(pos_samples)
                         if self.selected_inds[x] in include_nodes for y in line]
        
        self.pos_pairs = np.array(pos_pairs).T
        

    def get_negative_sampler(self, smooth_par=0.75):
        """
        """

        node_idx,node_degrees = np.unique(self.edge_index[0,:], return_counts=True)

        # there may be isolated nodes that are not present in edge_index
        all_degrees = np.zeros(self.n_x)
        all_degrees[node_idx] = node_degrees
        
        Pn = all_degrees ** smooth_par
        Pn = Pn / np.sum(Pn)

        self.neg_sampler = categorical.Categorical(torch.from_numpy(Pn))


    def get_neighbor_sampler(self, train_sizes, b1=50, b2=1000, nworkers=50):
        """Neighbor sampler for this data
        """

        # neighbor sampler for training
        self.train_loader = NeighborSampler(self.edge_index, node_idx=None,
                                            sizes=train_sizes, batch_size=b1,
                                            num_workers=nworkers, shuffle=False)

        # neighbor sampler for testing (single-layer sampler)
        self.test_loader = NeighborSampler(self.edge_index, node_idx=None,
                                           sizes=[-1], batch_size=b2,
                                           num_workers=nworkers, shuffle=False)


        
    def _load_hypergraph_materials(self):

        assert self.path_to_ents is not None, \
            'init_validation needs a path to the entites. ' + \
            'Set it via data.path_to_ents .'
        
        VM = sparse.load_npz(self.path_to_VM)
        VMkw = sparse.load_npz(self.path_to_VMkw)
        self.IMat = sparse.hstack((VM,VMkw))
        self.row_yrs = np.loadtxt(self.path_to_yrs)
        self.ents = np.array(open(self.path_to_ents, 'r').read().splitlines())
        self.nA = self.IMat.shape[1] - len(self.ents) - 1
        self.tags = np.array(['a_{}'.format(i) if i<self.nA else
                              ('prop' if i==self.IMat.shape[1]-1 else self.ents[i-self.nA])
                              for i in self.selected_inds])
        self.selected_ents = np.array([self.ents[i-self.nA]
                                       for i in self.selected_inds[:-1] if i>=self.nA])

        # the following is written specifically for materials science data set
        gt = gt_discoveries(self.ents,
                            self.IMat[:,self.nA:],
                            self.row_yrs)
        self.GT = np.concatenate([gt(yr) for yr in range(self.year,2019)])
        self.studied_ents = find_studied_ents_VW(self.ents,
                                                self.IMat[:,self.nA:],
                                                self.row_yrs,
                                                self.year)


    def _fetch_node_features(self):

        if self.feature_type=='atomic':
            self.x_all = load_atomic_features(self.selected_inds,
                                              self.ents,
                                              self.nA)

        elif self.feature_type=='w2v':
            # The model is suppsed to have been trained over sentences
            # before the prediction year --> (-inf, year)
            subIMat = self.IMat[self.row_yrs<self.year,:].tocsr()
            sub_row_yrs = self.row_yrs[self.row_yrs<self.year]
            # manually zeroing papers that don't belong to
            # the period [yr-memory, yr-1]
            for row in range(subIMat.shape[0]):
                if sub_row_yrs[row]<(self.year-self.memory):
                    subIMat.data[subIMat.indptr[row]:subIMat.indptr[row+1]]=0
            subIMat.eliminate_zeros()
            
            self.x_all = load_w2v_features(self.selected_inds,
                                           subIMat[:,:self.nA],
                                           self.ents,
                                           self.prop,
                                           self.path_to_w2v,
                                           self.path_to_sents,
                                           self.alpha)
        elif self.feature_type=='indicator':
            # don't need to pre-calculate the indicator feature vectors for all
            self.x_all=None
            

        
    def get_node_features(self, sample_inds, type='atomic'):
        """Loading feature vectors of some nodes into the memroy
        """


        if self.feature_type in ['atomic', 'w2v']:
            # these features required pre-calculated features
            batch_X = self.x_all[sample_inds,:]
            if np.ndim(batch_X)==1:
                batch_X = np.expand_dims(batch_X, axis=0)
        elif self.feature_type in ['indicator']:
            # for indicator features, the features will be generated on the spot
            batch_X = np.zeros((len(sample_inds), self.n_x), dtype=np.float32)
            batch_X[np.arange(len(sample_inds)), sample_inds] = 1

        return batch_X
                                          

def load_atomic_features(inds_all, chems, nA):
    """Load atomic feature from a given set of sample indices

    Each sample id is indeed an integer between 0, nA+nE+1, where
    nA is the number of authors and nE is the number of entities 
    (here, chemical compounds). 
    """

    f = atomic_features()
    
    # number of features = len(elements) + 2 = 119 + 2 = 121
    x_all = np.zeros((len(inds_all), 121), dtype=int)
    for i,idx in enumerate(inds_all):
        if idx<nA:
            x_all[i,0] = 1
        elif idx < (nA+len(chems)):
            x_all[i,1:-1] = f(chems[idx-nA])
        else:
            x_all[i,-1] = 1

    # normalization
    x_all = x_all / np.sqrt(np.sum(x_all**2,axis=1,keepdims=True))

    return x_all


def load_w2v_features(inds_all,
                      author_IMat,
                      ents,
                      prop,
                      path_to_w2v,
                      path_to_sents,
                      alpha):
    """Engineering node features by means of word2vec embeddings
    vectors. The entities and property nodes will be assigned their
    corresponding embedding vector, whereas the authors will be given
    the average of the sentence embeddings from the abstract of all 
    their papers. The sentence embedding is computed through a smoothened
    weighted average, which is also  adjusted by subtracting the first
    principal component of the sentences.

    Args:

    * inds_all: array-like
    Index of all the nodes that were selected in our dataset. The 
    indices that correspond to authors should be located in the first chunk
    of this array. Their size is equal to the the number of
    columns in `author_IMat`, i.e.
    `inds_all = [A_ids, E_inds, P_ind]` where
    A: auhors, E: entities, P: property

    * author_IMat: 2D sparse array
    Incidence matrix corresponding to all the author nodes (no matter
    if they are among the selected nodes or not). Number of papers (hyperedges)
    in this matrix should be equal to the number of abstracts (saved as
    sentences) whose path is given by `path_to_sents`. In case we would like 
    to see papers only in a specific time-window, the rows outside this
    window should be zero-ed out before feeding it to this function.

    * ents: array-like
    List of all entities (no matter if they are among the selected
    nodes or not)

    * prop: str or list of str
    Property keyword(s)

    * path_to_w2v: str
    Path to the Word2Vec model

    * path_to_sents: str
    Path to the sentences using which the Word2Vec model were trained

    * alpha: float scalar
    The smoothing parameter
    """

    # total number of authors
    nA = author_IMat.shape[1]
    # number of selected authors
    nA_selected = np.sum(inds_all<nA)

    # load the w2v model
    model = Word2Vec.load(path_to_w2v)

    # load sentences, and compute their embeddings
    sents = np.array(open(path_to_sents, 'r').read().splitlines())
    sents_embeds = unadjusted_words2sents(sents, model, alpha)

    x_all = np.zeros((len(inds_all), model.vector_size))

    # the easier task first (entities and property)
    for i,ind in enumerate(inds_all[nA_selected:]):
        # the last index is for the property
        if ind != inds_all[-1]:
            ent = ents[ind-nA]
            idx = model.wv.vocab[ent].index
            v = model.trainables.syn1neg[idx,:]
            x_all[nA_selected+i,:] = v/np.sqrt((v**2).sum())
        else:
            x_all[-1,:] = model.wv[prop]/np.sqrt((model.wv[prop]**2).sum())

    # now, the more demanding task (authors)
    author_IMat = author_IMat.tocsc()
    pbar = tqdm(range(nA_selected), position=0, leave=True)
    pbar.set_description('Words2Sents for Authors')
    for i,ind in enumerate(inds_all[:nA_selected]):
        pids = author_IMat[:,ind].indices
        
        # so far the matrices were row-wise, make it
        # column-wise to be more consistent with the formula
        V = np.concatenate([sents_embeds[j] for j in pids], axis=0).T
        u,_,_ = randomized_svd(V, n_components=1)

        # adjusted average vector
        avg = np.sum(V, axis=1)
        avg =  avg - np.dot(np.dot(u,u.T), avg)
        x_all[i,:] = avg.squeeze() / np.sqrt((avg**2).sum())

        pbar.update(1)

    pbar.close()
    return x_all
