import os
import sys
import pdb
import json
import logging
import numpy as np
from scipy import sparse
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec

path = '/home/jamshid/codes/social-knowledge-analysis/'
sys.path.insert(0, path)
path = '/home/jamshid/codes/mat2vec/'
sys.path.insert(0, path)

from data import utils
from misc import helpers
from training.utils import keep_simple_formula

from mat2vec.training.phrase2vec import wordgrams
from mat2vec.training.helpers.utils import COMMON_TERMS,EXCLUDE_PUNCT, INCLUDE_PHRASES

pr = utils.MatTextProcessor()

DEFAULT_PARS_PATH = '/home/jamshid/codes/data/training/default_params.json'
with open(DEFAULT_PARS_PATH, 'r') as f:
    DEFAULT_PARS = json.load(f)
DEFAULT_PARS['common_terms'] = COMMON_TERMS
DEFAULT_PARS['exclude_punct'] = EXCLUDE_PUNCT
DEFAULT_PARS['include_phrases'] = INCLUDE_PHRASES



class W2V(object):

    def __init__(self, path_to_data, **kwargs):


        self.path_to_data = path_to_data
        self.pars = {}
        for key, def_val in DEFAULT_PARS.items():
            self.pars[key] = kwargs.get(key, def_val)

        # setting up the logger
        logger_disable = kwargs.get('logger_disable', False)
        self.logfile_path =   kwargs.get('logfile_path', None)
        self.logger = helpers.set_up_logger(__name__, self.logfile_path, logger_disable)

    def build_sentences_phrases(self, phrasing=True):

        self.logger.info('Parsing lines (sentences) in: {}: '.format(self.path_to_data))
        self.logger.info('Parameters for parsing phrases are as follows:')
        for key in ['depth', 'phrase_count', 'phrase_threshold',
                    'common_terms', 'exclude_punct', 'include_phrases']:
            self.logger.info('\t{}: {}'.format(key, self.pars[key]))

        self.logger.info('='*20 + 'Gensim logging starts' + '='*20)

        
        self.sentences = LineSentence(self.path_to_data)

        if phrasing:
            self.sentences, self.phrases = wordgrams(self.sentences,
                                                     self.pars['depth'],
                                                     self.pars['phrase_count'],
                                                     self.pars['phrase_threshold'],
                                                     self.pars['common_terms'],
                                                     self.pars['exclude_punct'],
                                                     self.pars['include_phrases'])

        self.logger.info('='*20 + 'Gensim logging ends' + '='*20)
        

    def train(self, **kwargs):

        self.trim_rule = kwargs.get('trim_rule', keep_simple_formula)
        self.model_save_path = kwargs.get('model_save_path', None)
        brkpnt = kwargs.get('brkpnt', 1)
        
        callbacks = [MyCallBack(brkpnt, self.model_save_path, self.logger)]

        self.logger.info('Training a model from scratch using the following parameters:')
        for key, val in self.pars.items():
            if key in ['depth', 'phrase_count', 'phrase_threshold',
                       'common_terms', 'exclude_punct', 'include_phrases']: continue
            self.logger.info('\t{}: {}'.format(key, val))
        if self.trim_rule is not None:
            self.logger.info('\ttrim rule: {}'.format(self.trim_rule.__name__))
        self.logger.info('The model will be saved in {}'.format(self.model_save_path))
        
        self.model = Word2Vec(self.sentences,
                              size=self.pars['size'],
                              window=self.pars['window'],
                              min_count=self.pars['min_count'],
                              sg=self.pars['sg'],
                              hs=self.pars['hs'],
                              workers=self.pars['workers'],
                              alpha=self.pars['alpha'],
                              sample=self.pars['subsample'],
                              negative=self.pars['negative'],
                              trim_rule=self.trim_rule,
                              compute_loss=True,
                              sorted_vocab=True,
                              batch_words=self.pars['batch'],
                              iter=self.pars['epochs'],
                              callbacks=callbacks)

        self.logger.info('='*20 + 'Gensim logging ends' + '='*20)

    def erase_logger(self):
        assert self.logfile_path==self.logger.handlers[0].baseFilename, \
            'For some reason, logger filename attribute of the trainer is \
             different than the file-name of the handler.'
        with open(self.logfile_path, 'r+') as f:
            f.truncate(0)

    def put_log_separator(self):
        assert self.logfile_path==self.logger.handlers[0].baseFilename, \
            'For some reason, logger filename attribute of the trainer is \
             different than the file-name of the handler.'
        with open(self.logfile_path, 'a') as f:
            f.write('='*50+'\n\n\n')
            

        
class MyCallBack(CallbackAny2Vec):

    """Callback to save model after every epoch."""
    def __init__(self, brkpnt=10, model_save_path=None, logger=None):
        self.epoch = 0
        self.losses = []
        #self.man_acc = []
        self.brkpnt = brkpnt
        self.logger = logger
        self.model_save_path = model_save_path

    def on_epoch_end(self, model):
        self.epoch += 1
        if not(self.epoch%self.brkpnt):
            if self.epoch==1:
                self.losses += [model.get_latest_training_loss()]
            else:
                self.losses += [model.get_latest_training_loss() - self.last_loss]

            self.last_loss = model.get_latest_training_loss()
            # manually added evaluator
            #self.man_acc += [self.man_eval(model)]

            if self.model_save_path is not None:
                if self.epoch==1:
                    model.save(self.model_save_path)
                else:
                    if self.losses[-1] < np.min(self.losses[:-1]):
                        model.save(self.model_save_path)

            if self.logger is not None:
                self.logger.info('{} Epoch(s) done. Loss: {}, LR: {}'.format(self.epoch,
                                                                             self.losses[-1],
                                                                             model.min_alpha_yet_reached))


                
class PPMI(object):

    def __init__(self, path_to_sents, **kwargs):

        self.sents = open(path_to_sents,'r').read().splitlines()
        
        # for now we are skipping the preprocessing steps as we
        # will be using this for deepwalk sentences that do not need
        # preprocessing
        # ... PREPROCESSING GOES HERE
        self.to_be_removed = []

        # setting up the logger
        logger_disable = kwargs.get('silent', False)
        self.logger = helpers.set_up_logger(__name__, None, logger_disable)

        
    def form_unigrams(self):

        self.uni_counts = Counter()
        for iD, D in enumerate(self.sents):
            tokens = [d for d in D.split(' ') if d not in self.to_be_removed]
            for i, tok in enumerate(tokens):
                # unigram count
                self.uni_counts[tok] += 1

            if not(iD%10000):
                self.logger.info(iD)

        self.tok2ind = {tok: ind for ind, tok in enumerate(self.uni_counts.keys())}
        self.ind2tok = {ind: tok for tok,ind in self.tok2ind.items()}
                
                
    def form_skipgrams(self, w):

        self.w = w
        win = int((w-1)/2)

        self.skip_counts = Counter()
        for iD, D in enumerate(self.sents):
            tokens = [d for d in D.split(' ') if d in self.uni_counts]

            for i, tok in enumerate(tokens):
                # pairwise count
                start_window = max(i-win, 0)
                end_window   = min(i+win, len(tokens)-1)
                context = [tokens[ii] for ii in np.arange(start_window, end_window+1) if ii!=i]
                for c in context:
                    self.skip_counts[tok, c] += 1

            if not(iD%10000):
                self.logger.info(iD)


    def form_sparse_cnt_matrix(self, save_path=None):

        nV = len(self.uni_counts)
        self.cmat = sparse.lil_matrix((nV,nV), dtype=np.uint32)
        rows=[]
        cols=[]
        vals=[]
        for i,skp in enumerate(self.skip_counts.items()):
            # each element of the skipgram counter is like
            # [..., (('center','cntxt'), count), ...]
            # index of the center (=row)
            rows += [self.tok2ind[skp[0][0]]]
            # index of the context (=column)
            cols += [self.tok2ind[skp[0][1]]]
            # pairwise counts of center and context tokens
            vals += [skp[1]]

            if not(i%1000000) or (i==len(self.skip_counts)-1):
                self.logger.info(i)
                self.cmat[rows,cols] = vals
                rows=[]
                cols=[]
                vals=[]

        self.cmat = self.cmat.tocsr()
        
        if save_path is not None:
            sparse.save_npz(self.cmat, save_path)


    def form_sppmi_matrix(self):
                                                                                
        nV = len(self.uni_counts)
        self.sppmi = sparse.lil_matrix((nV,nV), dtype=np.float32)

        word_cnts = np.array(self.cmat.sum(1)).flatten()
        ncorp = word_cnts.sum()
        shift = np.log(self.w)

        rows = []
        cols = []
        vals = []
        for ind,tok in self.ind2tok.items():
            nnz_cols = self.cmat[ind,:].indices
            wj_cnts = word_cnts[nnz_cols]
            wi_cnt  = word_cnts[ind]

            # associatedness
            assocs = ncorp*self.cmat[ind,:].data / (wi_cnt*wj_cnts)
            # shifted ppmi
            sppmi_vals = np.log(assocs) - shift
            p_cols = nnz_cols[sppmi_vals>0]

            rows += list(np.ones(len(p_cols), dtype=np.int32)*ind)
            cols += list(p_cols)
            vals += list(sppmi_vals[sppmi_vals>0])

            if not(ind%10000) or (ind==nV-1):
                self.logger.info(ind)
                self.sppmi[rows,cols] = vals
                rows = []
                cols = []
                vals = []


    def get_most_similar_terms(self, tok, words=None, ntop=10, thrld=0, use_SVD=False):

        if words is None:
            words_inds = np.array([y for x,y in self.tok2ind.items()
                                   if self.uni_counts[x]>thrld])
        else:
            words_inds = np.array([self.tok2ind[x] for x in words
                                   if self.uni_counts[x]>thrld])

        cosine_sims = cosine_similarity(self.cmat[[self.tok2ind[tok]],:],
                                        self.cmat[words_inds,:]).flatten()
        most_similar_inds = words_inds[np.argsort(-cosine_sims)[:ntop]]

        return [(self.ind2tok[x], cosine_sims[x]) for x in most_similar_inds]
