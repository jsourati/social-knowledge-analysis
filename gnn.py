import torch
import torch.nn.functional as F
from torch.distributions import categorical
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GAE, VGAE

import os
import sys
import pdb
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy import sparse

path = '/home/jamshid/codes/social-knowledge-analysis/'
sys.path.insert(0, path)

import hypergraphs
from misc import helpers
from data import pyg_datasets

class UnsGAE(object):

    def __init__(self, data, embed_dim, **kwargs):
        super(UnsGAE, self).__init__()
        self.data = data
        self.input_dim = self.data.dim
        self.embed_dim = embed_dim

        # for now, we only work with 2-layer encoders
        self.hidden_dim = kwargs.get('hidden_dim', 2*embed_dim)
        self.encoder = kwargs.get('encoder', batched_SAGEEncoder)
        self.encoder = self.encoder(self.input_dim,
                                    self.hidden_dim,
                                    self.embed_dim)
        self.model = GAE(self.encoder)

        # preparing the device 
        device = kwargs.get('device', 'cuda')
        if device=='gpu' and not(torch.cuda.is_available()):
            print('CUDA is not available in PyTorch. the model ' +\
                  'will be initiated on CPU.')
            device = 'cpu'
        self.device = torch.device(device)

        
    def init_model(self, sizes, weights_path=None):
        self.model = self.model.to(self.device)
        
        # sizes are directly used for initializing the model
        # but it will be used for every feed-forward as the
        # sampling size of the neighbors
        assert len(sizes)==self.model.encoder.num_layers, \
            'Number of sizes should be equal to the number of layers in the encoder.'
        self.sizes = sizes
        if not(hasattr(self.data, 'loader')):
            self.data.get_neighbor_sampler(self.sizes)

        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            
        
    def init_training(self, neg_num, optim='Adam', lr=1e-5, smooth_par=0.75):
        if optim=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim=='SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.train_one_epoch = self._train_edge_batching
        self.neg_num = neg_num

        if not(hasattr(self.data, 'pos_pairs')):
            assert 'pos_samples_path' in kwargs, 'The provided data does ' +\
                'not come with positive pairs, and we need a path to the ' +\
                'already selected positive samples. You can provide it through ' +\
                'input pos_samples_path .'
            include_nodes = kwargs.get('include_nodes', None)
            self.data.load_positive_pairs(kwargs['pos_samples_path'], include_nodes)
            
        if not(hasattr(self.data, 'neg_sampler')):
            #smooth_par = kwargs.get('smooth_par', 0.75)
            self.data.get_negative_sampler(smooth_par)

        if not(hasattr(self.data, 'x_all')):
            self.data._fetch_node_features()

        
    def init_validation(self):
        if not(hasattr(self.data, 'x_all')):
            self.data._fetch_node_features()
        

    def embed_some(self, sample_inds, b=100):
        """This will be used in the training, when the
        embedding of a batch of samples are needed
        """

        quot, rem = np.divmod(len(sample_inds), b)

        Z = []
        for i in range(quot+1):
            if i<quot:
                b_ids = sample_inds[i*b:(i+1)*b]
            elif rem>0:
                b_ids = sample_inds[i*b:]
                
            # neighbor-sampling for each sample
            _, n_id, adjs = self.data.train_loader.sample(b_ids)
            adjs = [adj.to(self.device) for adj in adjs]

            # get feature vectors through the neighbors sampled above
            batch_X = torch.from_numpy(self.data.get_node_features(n_id))
            batch_X = batch_X.to(torch.float).to(self.device)

            # the encoder's output as the embedding
            try:
                batch_Z = self.model.encoder(batch_X, adjs)
            except:
                pdb.set_trace()
            Z += [batch_Z]

        Z = torch.cat(Z, dim=0)
        return Z

    def embed_all(self):

        L = self.model.encoder.num_layers
        pbar = tqdm(total=self.data.n_x * L, position=0, leave=True)
        pbar.set_description('Evaluating')
        
        self.model.encoder.eval()
        # inference is used in the evaluation stage (not in training) when
        # the embeddings for "all" nodes will be computed. It's written in a way
        # that is faster than the foward-passing function which is mostly used
        # for single batches in the training
        with torch.no_grad():
            for i in range(L):
                xs = []
                for batch_size, n_id, adj in self.data.test_loader:
                    edge_index, _, size = adj.to(self.device)
                    if i==0:
                        x = torch.from_numpy(self.data.get_node_features(n_id))
                        x = x.to(torch.float).to(self.device)
                    else:
                        x = x_all[n_id,:].to(self.device)

                    x_target = x[:size[1]]
                    x = self.model.encoder.convs[i]((x,x_target), edge_index)
                    if i != L-1:
                        x = F.relu(x)

                    xs.append(x[:batch_size,:].cpu())
                    
                    pbar.update(batch_size)

                x_all = torch.cat(xs, dim=0)
                
        pbar.close()
        
        return x_all
    
            
    def _train_edge_batching(self, ep, batch_size=5000):

        assert hasattr(self.data, 'pos_pairs'), 'Positive and negative ' + \
            'samples must be generated before starting the training'
        
        self.model.train()
        neg_num = self.neg_num

        torch.multiprocessing.set_sharing_strategy('file_system')
        pbar = tqdm(total=self.data.pos_pairs.shape[1], position=0, leave=True)
        pbar.set_description(f'Epoch {ep:02d}')
        
        total_loss = 0
        np.random.shuffle(self.data.pos_pairs.T)
        quot, rem = np.divmod(self.data.pos_pairs.shape[1], batch_size)

        for i in range(quot+1):

            # positive mini-batch
            # (#: batch size)
            if i<quot:
                batch_pos_pairs = self.data.pos_pairs[:,i*batch_size:(i+1)*batch_size]
            else:
                batch_pos_pairs = self.data.pos_pairs[:,i*batch_size:]
            batch_pos_samples, pos_edge_index = np.unique(batch_pos_pairs,
                                                          return_inverse=True)
            pos_edge_index = pos_edge_index.reshape(batch_pos_pairs.shape)

            # negative mini-batch
            # (#: batch_size * neg_num)
            batch_neg_samples = self.data.neg_sampler.sample(
                torch.Size([neg_num*batch_pos_pairs.shape[1]]))
            neg_edge_index = np.array([np.repeat(pos_edge_index[0,:],neg_num),
                                       np.arange(pos_edge_index.max()+1,
                                                 pos_edge_index.max()+len(batch_neg_samples)+1)])

            # embeddings of the nodes involved in + and - edges
            self.optimizer.zero_grad()
            unodes = batch_pos_samples.tolist() + batch_neg_samples.tolist()
            Z = self.embed_some(unodes)

            # reconstruction loss
            pos_edge_index = torch.from_numpy(pos_edge_index).to(self.device)
            neg_edge_index = torch.from_numpy(neg_edge_index).to(self.device)
            loss = self.model.recon_loss(Z, pos_edge_index, neg_edge_index)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss)

            pbar.update(batch_size)

        pbar.close()

        loss = total_loss / (quot+1)
        return loss


    def validate(self):

        self.model.eval()

        Z = self.embed_all()
        ents_Z = Z[:-1,:][self.data.selected_inds[:-1]>=self.data.nA,:].detach().numpy()
        prop_Z = Z[self.data.tags=='prop',:].detach().numpy()
        scores = np.dot(ents_Z, prop_Z.T).squeeze()
        
        sorted_ents = self.data.selected_ents[np.argsort(-scores)]
        unstudied_sorted_ents = np.array([x for x in sorted_ents
                                           if x not in self.data.studied_ents])
        preds = unstudied_sorted_ents[:50]

        prec = np.isin(preds,self.data.GT).sum() / len(preds)

        return prec

class batched_SAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(batched_SAGEEncoder, self).__init__()
        
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        # do NOT use cached=True with mini-batching
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=True))

    def forward(self, x, adjs):
        # the inputs to the farward function is not just the whole graph,
        # but a batch of nodes and their layer-wise adjacencies
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                x = F.relu(x)
                #x = F.dropout(x, p=0.5, training=self.training)
        return x
                                                                                                                                                
