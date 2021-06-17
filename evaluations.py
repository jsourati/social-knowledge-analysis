import os
import sys
import pdb
import copy
import umap
import json
import logging
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse

from gensim.models import Word2Vec

from sklearn import metrics
from sklearn.metrics import roc_auc_score

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

pr = utils.MaterialsTextProcessor()


def average_accs_dict(accs_dict, stat='mean'):

    yrs = list(accs_dict.keys())
    stats = np.zeros(len(accs_dict[np.min(yrs)]))
    for i in range(len(stats)):
        vals = [accs[i] for _,accs in accs_dict.items() if len(accs)>i]
        stats[i] = np.mean(vals) if stat=='mean' else np.std(vals)
    return np.array(stats)
        
        
def eval_predictor(predictor_func,
                   gt_func,
                   year_of_pred,
                   **kwargs):
    """Evaluating a given predictor function in how accurate its predictions
    match the actual discoveries returned by a given ground-truth function

    The evaluations are done for individual years strating from a given year 
    of prediction to 2018.
    """

    metric = kwargs.get('metric', 'cumul_precision')
    last_year = kwargs.get('last_year', 2019)
    save_path = kwargs.get('save_path', None)
    return_preds = kwargs.get('return_preds', False)
    logfile_path = kwargs.get('logfile_path', None)
    logger_disable = kwargs.get('logger_disable',False)
    logger = helpers.set_up_logger(__name__, logfile_path, logger_disable)
    

    """ Generating the Prediction """
    preds = predictor_func(year_of_pred)
    logger.info('Number of actual predictions: {}'.format(len(preds)))
    if metric=='auc':
        if len(preds)!=2:
            raise ValueError('When asking for AUC metric, predictor should return score array too.')
        scores = preds[1]
        preds = preds[0]

    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write('\n'.join(preds)+'\n')
    
    """ Evaluating the Predictions for the Upcoming Years """
    years_of_eval = np.arange(year_of_pred, last_year)
    iter_list = []  # to be the prec. values or actul disc. (for AUC)   
    for i, yr in enumerate(years_of_eval):
        gt = gt_func(yr)

        if metric=='cumul_precision':      # Cumulative Precision
            iter_list += [np.sum(np.in1d(gt, preds)) / len(preds)]
        elif metric=='auc':    # Area Under Curve
            iter_list += gt.tolist()

    if metric == 'cumul_precision':
        res = np.cumsum(iter_list)
    elif metric == 'auc':
        y = np.zeros(len(preds))
        y[np.isin(preds,iter_list)] = 1
        res = roc_auc_score(y, scores)

    if return_preds:
        return res, preds
    else:
        return res


def eval_author_predictor(discoverers_predictor_func,
                          gt_discoverers_func,
                          year_of_pred,
                          **kwargs):
    """If return_yrs is True, then the function returns a dictionary
    that contains the years which true positives happened
    """

    last_year = kwargs.get('last_year', 2019)
    return_yrs = kwargs.get('return_yrs', False)
    
    preds = discoverers_predictor_func(year_of_pred)

    years_of_eval = np.arange(year_of_pred, last_year)
    precs = np.zeros(len(years_of_eval))
    inclusive_preds = preds
    yrs_dict = {}
    for i, yr in enumerate(years_of_eval):
        gt = gt_discoverers_func(yr)

        tpbin = np.isin(preds, gt)
        precs[i] = np.sum(tpbin)

        if return_yrs:
            tp = inclusive_preds[np.isin(inclusive_preds,gt)]
            for au in tp:
                # int() function is added below to keep the year types
                # int and not numpy.int64 which is not JSON serializable
                if au in yrs_dict:
                    yrs_dict[au] += [int(yr)]
                else:
                    yrs_dict[au] = [int(yr)]

        # exclude the ones that have already been labeled as a discoverer
        #preds = preds[~tpbin]

    if return_yrs:
        return precs/len(inclusive_preds), yrs_dict
    else:
        return precs/len(inclusive_preds)


def get_drug_scores(drgs, scores, **args):
    '''`scores` includes results of a scoring functions for the
    drugs with the same order in `drugs` (possible with NAN values)
    '''

    correct_order = args.get('correct_order', False)
    nonan = args.get('nonan', False)
    drugs = args.get('drugs', None)

    if drugs is None:
        path = '/home/jamshid/codes/data/pubmed_covid19/drugs.txt'
        drugs = np.array(open(path,'r').read().splitlines())

        
    # the following line return the scores in the order of the given drugs
    # in variable "drugs" (and not drgs)
    if correct_order:
        drg_scores = np.array([scores[drugs==x][0] for x in drgs])
    else:
        drg_scores = scores[np.isin(drugs,drgs)]

    if nonan:
        drg_scores = drg_scores[~np.isnan(drg_scores)]

    return drg_scores


def SP_diseases_full(ds, drugs_len, subVM, sub_rows, save_dir=None):
    ''' For structure in the form:
    
    R = [A_1,...,A_nA, Dr1,...,DrN, P]
    '''
    
    base_dir = '/home/jamshid/codes/data/CTD/disease_vertex_weight_submatrices'
    subVMkw = sparse.load_npz('{}/{}.npz'.format(base_dir,ds))[sub_rows,:]
    subR = sparse.hstack((subVM,subVMkw))

    # create NX graph
    subP = hypergraphs.compute_transprob(subR)
    subP[subP.nonzero()] = 1.
    G = nx.from_scipy_sparse_matrix(subP)

    spds = np.zeros(drugs_len)
    nA = subVM.shape[1]-drugs_len
    for i in range(drugs_len):
        try:
            spds[i] = nx.shortest_path_length(G,source=subR.shape[1]-1, target=nA+i)
        except Exception as ex:
            if type(ex).__name__ == 'NetworkXNoPath':
                spds[i] = -1

    if save_dir is not None:
        np.savetxt('{}/{}.txt'.format(save_dir,ds), spds,fmt='%d')

    return spds


def SP_diseases_authorgraph(ds, drugs_len, subVM, sub_rows, save_dir=None):
    '''Computing SP distances from authorship graph (excluding all
    other drug nodes
    '''

    base_dir = '/home/jamshid/codes/data/CTD/disease_vertex_weight_submatrices'
    subVMkw = sparse.load_npz('{}/{}.npz'.format(base_dir,ds))[sub_rows,:]
    subR = sparse.hstack((subVM,subVMkw))

    subP = hypergraphs.compute_transprob(subR)
    subP[subP.nonzero()] = 1.

    # create NX authorship graph with the porperty node
    nA = subVM.shape[1] - drugs_len
    AsubR = sparse.hstack((subR[:,:nA], subR[:,-1]))
    AsubP = hypergraphs.compute_transprob(AsubR)
    AsubP[AsubP.nonzero()] = 1.
    G = nx.from_scipy_sparse_matrix(AsubP)

    spds = np.zeros(drugs_len)
    new_node_idx = AsubP.shape[1]   # to be added to the graph for each drug
    for i in range(drugs_len):
        # forming edges from the drug to the author and the property (if any)
        nbrs = subP[nA+i,:].indices
        nbrs = nbrs[(nbrs<nA)|(nbrs==subP.shape[0]-1)]
        nbrs[nbrs==(subP.shape[0]-1)] = AsubP.shape[0]-1

        G.add_node(new_node_idx)
        edges = [(new_node_idx,x) for x in nbrs] + [(x,new_node_idx) for x in nbrs]
        G.add_edges_from(edges)

        try:
            spds[i] = nx.shortest_path_length(G, source=AsubP.shape[1]-1, target=new_node_idx)
        except Exception as ex:
            if type(ex).__name__ == 'NetworkXNoPath':
                spds[i] = -1


        G.remove_node(new_node_idx)

        if not(i%10):
            print(i,end=',')

    if save_dir is not None:
        np.savetxt('{}/{}_authorgraph.txt'.format(save_dir,ds), spds,fmt='%d')

    return spds

    

def get_AAI_orbits(drugs, txt_model, diseases_with_codes, **kwargs):
    '''Computing and saving the information on evauation orbitals
    for different diseases tailored for the data shared by Gysi et al., 2021
    '''

    metric = kwargs.get('metric', 'emb_cosine')
    row_years = kwargs.get('row_years', 'None')
    silent = kwargs.get('silent', True)
    save_dir = kwargs.get('save_dir', None)
    spds = kwargs.get('spds', None)

    if row_years is None:
        row_years = np.loadtxt('/home/jamshid/codes/data/pubmed_covid19/OVRLP_yrs.txt')

    # Getting IDs for the drugs in `drugs`
    # (this will be needed when computing drug-disease similarity scores)
    df = pd.read_csv('/home/jamshid/codes/data/pubmed_covid19/DrugBank_cRanks_Pipelines.tsv',
                     usecols=['DrugBank_ID', 'DrugBank_name', 'DrugBank_cRank'],
                     delimiter='\t')
    df.columns = ['DrugBank_ID', 'DrugBank_name', 'DrugBank_rank']

    df = df[~df['DrugBank_name'].isnull()]
    df_drugs = np.array(df.DrugBank_name)
    df_drugs[df_drugs=='Hybrid Between B and C Type Hemes (Protoporphyrin Ixcontaining Fe)'] = 'Protoporphyrin Ix containing Fe'
    df_drugs = np.array([x.replace(' ','_').lower() for x in df_drugs])

    drugs_ids = [df[df_drugs==x].DrugBank_ID.values[0] for x in drugs]

    
    # preparing embeddings for drugs and diseases
    drugs_emb = pd.read_csv('/home/jamshid/codes/data/pubmed_covid19/Gysi_drugs_emb.csv', header=None)
    dis_emb = pd.read_csv('/home/jamshid/codes/data/pubmed_covid19/Gysi_diseases_emb.csv', header=None)
    dr_ids = drugs_emb.iloc[:,0].values
    ds_ids = dis_emb.iloc[:,0].values
    drugs_emb_mat = drugs_emb.iloc[:,1:].values


    # getting only those drugs in the network
    # (the ones outside does not have SPds computed for them, and
    # it doesn't make sense to make their SPds infinity either, because
    # they don't have any nodes in the network at all)
    VM = sparse.load_npz('/home/jamshid/codes/data/pubmed_covid19/OVRLP_vertex_weight_matrix.npz')
    nA = VM.shape[1] - len(drugs)
    subVM = VM[(row_years>=1996)*(row_years<=2000),:]
    target_ents = drugs  # drugs[np.unique(subVM[:,nA:].nonzero()[1])]

    
    # get SPd mapping: (f(x) = x if x<5
    #                          5 if 5 <= x < inf_val
    #                          6 if x=inf_val (i.e., inf)
    def spds_map(x, inf_val):
        if x<=4:
            return x
        elif (4<x) and (x<inf_val):
            return 5
        elif x==inf_val:
            return 6
        else:
            raise ValueError('SPd values should be a bounded integer >-1')
        
    results_per_disease = {}
    for ds,code in diseases_with_codes.items():
        if not(silent):
            print('Working on {}...'.format(ds))

        if code not in ds_ids:
            continue
            
        # computing and matching drugs-disease similarities
        if metric=='emb_cosine':
            ds_emb = dis_emb[ds_ids==code].values[0,1:]
            emb_scores = np.dot(drugs_emb_mat, ds_emb)
            drnorms  = np.linalg.norm(drugs_emb_mat, axis=1)
            dsnorms = np.linalg.norm(ds_emb)
            
            all_scores = emb_scores / (dsnorms*drnorms)
        elif metric=='UMAP_Euc':
            coords = np.loadtxt('/home/jamshid/codes/data/CTD/AAI/UMAP_dsdr_coords.txt')
            # computed through:
            # embs = np.concatenate((drugs_emb_mat, dis_emb.iloc[:,1:]),axis=0)
            # coords = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.8,
            #                    metric='cosine', random_state=0).fit_transform(embs)
            dr_coords = coords[:len(dr_ids)]
            ds_coord  = coords[len(dr_ids)+np.where(ds_ids==code)[0],:]
            all_dists = np.squeeze(metrics.pairwise_distances(X=dr_coords,
                                                              Y=ds_coord,
                                                              metric='euclidean'))

            all_scores = np.exp(-all_dists)

        # these are the scores:
        # (NaN for drugs that don't exist in the protein-protein embegging)
        drugs_scores = np.array([all_scores[dr_ids==x][0]
                                 if x in dr_ids else np.nan for x in drugs_ids])

        gt_func = helpers.gt_discoveries_4CTD(ds)
        GT = np.concatenate([gt_func(y) for y in range(2001,2020)])

        # getting the alienness (SPds) and the plausability (woord2vec) scores
        if spds is None:
            try:
                spds = np.loadtxt('/home/jamshid/codes/data/CTD/AAI/SPDs/{}.txt'.format(ds))
            except:
                continue
        else:
            # copy of the original vector to preserve its -1 values
            cp_spds = copy.deepcopy(spds)
        inf_val = spds.max()+1
        cp_spds[cp_spds==-1] = inf_val
        
        txt_sims = measures.cosine_sims(txt_model,target_ents,ds.lower())
        txt_sims[np.isnan(txt_sims)] = -np.inf

        # running alien AI for different beta values
        S1 = np.array([cp_spds[drugs==x][0] for x in target_ents])
        S2 = np.exp(txt_sims)
        
        # if True, only consider those drugs with valid embedding score
        if True:
            target_scores = get_drug_scores(target_ents, drugs_scores, correct_order=True)
            nonan_target_ents = target_ents[~np.isnan(target_scores)]
            nonan_S1 = S1[~np.isnan(target_scores)]
            nonan_S2 = S2[~np.isnan(target_scores)]
            # change the follwing to nonan_...
                                

        betas = np.arange(-10,11) * 0.1
        spd_levels = np.arange(1,7)
        lai_percs = np.zeros((6,len(betas)))
        lai_scores = np.zeros((6,len(betas)))
        precs = np.zeros(len(betas))
        ovscores = np.zeros(len(betas))
        for i,beta in enumerate(betas):
            S = measures.combine_scores(nonan_S1, nonan_S2, beta=beta, method='van-der-waerden')
            preds = nonan_target_ents[np.argsort(-S)][:50]
            preds_spds = cp_spds[np.isin(drugs,preds)]
            preds_spds = np.array([spds_map(x,inf_val) for x in preds_spds])
            preds_scores = get_drug_scores(preds,drugs_scores,correct_order=True)

            lai_percs[:,i] = helpers.logperc([(preds_spds==x).sum()/len(preds) for x in spd_levels])

            lai_scores_wnan = [preds_scores[preds_spds==x]
                               if x in preds_spds else 0 for x in spd_levels]
            # this is to zero out the all-nan vectors:
            lai_scores_wnan = [0 if np.all(np.isnan(x)) else x for x in lai_scores_wnan]
            lai_scores[:,i] = [np.mean(x[~np.isnan(x)]) if np.ndim(x)>0 else x
                               for x in lai_scores_wnan]

            precs[i] = np.sum(np.isin(preds,GT))/len(preds)
            ovscores[i] = np.mean(preds_scores[~np.isnan(preds_scores)])
            

        results_per_disease[ds] = [lai_percs, lai_scores, precs, ovscores]

        if save_dir is not None:
            ds_save_dir = os.path.join(save_dir,ds)
            if not(os.path.exists(ds_save_dir)):
                os.mkdir(ds_save_dir)
            np.savetxt(os.path.join(ds_save_dir,'Embslogpercs_whole.csv'),lai_percs,fmt='%.3f', delimiter=',')
            np.savetxt(os.path.join(ds_save_dir,'EmbsOvScores_whole.csv'),
                       np.expand_dims(ovscores,axis=0) ,fmt='%.5f', delimiter=',')
            np.savetxt(os.path.join(ds_save_dir,'Embsprecs_whole.csv'),
                       np.expand_dims(precs,axis=0), fmt='%.2f', delimiter=',')
    
    return results_per_disease

        
        
        
        
