import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch.distributions import categorical
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GAE, VGAE

from sklearn.manifold import TSNE
from gensim.models import Word2Vec

import os
import sys
import pdb
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy import sparse
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pymatgen.core.composition import Composition
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


path = '/home/jamshid/codes/social-knowledge-analysis/'
sys.path.insert(0, path)

import gnn
import hypergraphs
from misc import helpers
from data import pyg_datasets


yr = 2001
kw = 'thermoelectric'
alpha=1
pars = {'root': '/home/jamshid/codes/data/GNN/data/thermoelectric_2',
        'path_to_VM': '/home/jamshid/codes/data/materials_science/hypergraphs/vertex_matrix.npz',
        'path_to_VMkw': '/home/jamshid/codes/data/materials_science/hypergraphs/vertex_submatrix_thermoelectric_multiKW.npz',
        'path_to_yrs': '/home/jamshid/codes/data/materials_science/MSDB_row_years.txt',
        'path_to_ents': '/home/jamshid/codes/data/materials_science/MSDB_chems.txt',
        'yr': yr,
        'memory': 5,
        'path_to_w2v': '/home/jamshid/codes/data/training/models/abstracts_titles/msdb/before_{}/model'.format(yr),
        'path_to_sents': '/home/jamshid/codes/data/training/sentences/abstracts_titles/msdb/before_{}.txt'.format(yr),
        'prop': kw,
        'alpha': alpha,
        'fetch_features': False}

def read_data():
    data = pyg_datasets.GNNdata(feature_type='w2v', **pars)
    return data

def build_graph(data, prox):
    
    yr=2001
    memory=5
    subIMat = data.IMat[(data.row_yrs>=(yr-memory))*(data.row_yrs<=(yr-1)),:]

    nA = data.nA
    P = hypergraphs.compute_transprob(subIMat)
    Pe = P[nA:,:][:,nA:]

    if prox==1:
        P2 = hypergraphs.compute_multistep_transprob(P,
                                                     source_inds=np.arange(nA,P.shape[1]),
                                                     dest_inds=np.arange(nA,P.shape[1]),
                                                     interm_inds=np.arange(nA),
                                                     nstep=2)
        
        Pe = P2+Pe

    # removing diagoal vales before normalizing:
    Pe = Pe - sparse.diags(Pe.diagonal())

    # normalizing
    # normalizing the rows of the mixed transition matrix
    Pev2 = sparse.lil_matrix(Pe.shape, dtype=Pe.dtype)
    for i in range(Pe.shape[1]):
        cols = Pe[i,:].indices
        if len(cols)>0:
            dat = Pe[i,:].data
            rows = np.ones(len(cols))*i
            Pev2[rows,cols] = dat/dat.sum()
    Pev2 = Pev2.tocsr()

    # removing isolated nodes
    selected_inds_v2 = np.unique(Pev2.nonzero()[1])
    Pev2 = Pev2[selected_inds_v2,:][:,selected_inds_v2]
    selected_ents_v2 = data.ents[selected_inds_v2[:-1]]
    selected_ents_v2 = np.append(selected_ents_v2,['prop'])

    # return the transition matrix and the nodes that have been kept
    return Pev2, selected_inds_v2, selected_ents_v2


def positive_samples(P, M=20, L=10):

    walks = []
    n = P.shape[0]
    
    pbar = tqdm(total=n, position=0, leave=True)
    pbar.set_description('Positive Sampling..')
    for i in range(n):
        node_walks = []
        for j in range(M):
            node_walks += [helpers.random_walk_from_transprob(P,i,L)]
        walks += [node_walks]
        pbar.update(1)
    pbar.close()

    pos_pairs = []
    for i in range(len(walks)):
        dests = np.unique(np.concatenate(walks[i])).tolist()
        dests.remove(i)
        pos_pairs += [[i,x] for x in dests]
    pos_pairs = np.array(pos_pairs).T

    return pos_pairs


def form_edges(P):

    # forming the connectivities
    row = torch.from_numpy(P.tocoo().row).to(torch.long)
    col = torch.from_numpy(P.tocoo().col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, P.shape[0], P.shape[0])

    return edge_index


def load_w2v_features(data,selected_inds):

    model = Word2Vec.load(pars['path_to_w2v'])
    x_all = np.zeros((len(selected_inds), model.vector_size))
    for i, idx in enumerate(selected_inds):
        if idx==len(data.ents):
            x_all[i,:] = model.wv[kw]/np.sqrt((model.wv[kw]**2).sum())
        else:
            ent = data.ents[idx]
            ent_idx = model.wv.vocab[ent].index
            v = model.trainables.syn1neg[ent_idx,:]
            x_all[i,:] = v/np.sqrt((v**2).sum())

    return x_all
    


def train(args):

    prox = args.prox
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    path_to_init_model = args.init_model
    optimizer = args.optimizer
    neg_num = args.neg_num
    M = args.rw_size
    L = args.rw_length
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    save_path = args.save_path

    data = read_data()
    P, selected_inds, selected_ents = build_graph(data, prox)
    if not(hasattr(args,'pos_pairs')):
        pos_pairs = positive_samples(P,M,L)
    else:
        pos_pairs = args.pos_pairs
    edge_index = form_edges(P)
    x_all = load_w2v_features(data,selected_inds)

    data.x_all = x_all
    data.edge_index = edge_index
    data.pos_pairs = pos_pairs
    data.tags = selected_ents
    data.selected_ents = selected_ents
    data.selected_inds = selected_inds+data.nA
    data.n_x = len(selected_inds)


    data.get_neighbor_sampler([25,10])

    # building the GAE model
    gae = gnn.UnsGAE(data,embed_dim=embed_dim,hidden_dim=hidden_dim)
    gae.init_model([25,10])
    gae.init_validation()
    if path_to_init_model is not 'NA':
        gae.model.load_state_dict(torch.load(path_to_init_model))
    gae.init_training(neg_num, optim=optimizer, lr=lr)
    
    losses = []
    precs = []
    prec = gae.validate()
    precs += [prec]

    torch.save(gae.model.state_dict(), '{}/thermo_model_0.pars'.format(save_path))
    np.savetxt('{}/precisions.txt'.format(save_path),precs, fmt='%.3f')
    print('Initial precision: {}'.format(prec))
    for ep in np.arange(1,epochs+1):
        loss = gae.train_one_epoch(ep, batch_size=batch_size)
        torch.save(gae.model.state_dict(), '{}/thermo_model_{}.pars'.format(save_path,ep))
        
        losses += [loss]
        prec = gae.validate()
        precs += [prec]
        np.savetxt('{}/losses.txt'.format(save_path),losses, fmt='%.5f')
        np.savetxt('{}/precisions.txt'.format(save_path),precs, fmt='%.3f')
        print('Precision @ {}: {:.2f}'.format(ep, prec))
    


def main():
    parser = ArgumentParser("gae_trainer",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    
    parser.add_argument('--prox', default=0, type=int,
                        help='Proximity order when computing of the transition matrix')

    parser.add_argument('--embed_dim', default=200, type=int,
                        help='Dimension of the output embedding space')

    parser.add_argument('--hidden_dim', default=200, type=int,
                        help='Dimension of the hidden layer')

    parser.add_argument('--init_model', default='NA',
                        help='Path to the initialized model')

    parser.add_argument('--optimizer', default='SGD',
                        help='Optimizer to be used in the training')
    
    parser.add_argument('--neg_num', default=2, type=int,
                        help='Number of negative samples')

    parser.add_argument('--rw_length', default=5, type=int,
                        help='Length of random walk for positive sampling')

    parser.add_argument('--rw_size', default=20, type=int,
                        help='Size of random walk per node when positive sampling')

    parser.add_argument('--lr', required=True, type=float,
                        help='Learning rate')

    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of training epochs')

    parser.add_argument('--batch_size', default=5000, type=int,
                        help='Size of training mini-batches')

    parser.add_argument('--save_path', required=True, 
                        help='Directory path where to save the results')

    args = parser.parse_args()
    train(args)

    
    
if __name__ == '__main__':
    sys.exit(main())
                



    

    
