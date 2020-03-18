import os
import sys
import pdb
import json
import logging
import numpy as np

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

    def build_sentences_phrases(self):

        self.logger.info('Parsing lines (sentences) in: {}: '.format(self.path_to_data))
        self.logger.info('Parameters for parsing phrases are as follows:')
        for key in ['depth', 'phrase_count', 'phrase_threshold',
                    'common_terms', 'exclude_punct', 'include_phrases']:
            self.logger.info('\t{}: {}'.format(key, self.pars[key]))

        self.logger.info('='*20 + 'Gensim logging starts' + '='*20)

        
        sentences = LineSentence(self.path_to_data)
            
        self.sentences, self.phrases = wordgrams(sentences,
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
        self.logger.info('\ttrim rule: {}'.format(self.trim_rule.__name__))
        self.logger.info('The model will be saved in {}'.format(self.model_save_path))

        self.logger.info('='*20 + 'Gensim logging starts' + '='*20)
        
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

            if self.model_save_path is not None:
                model.save(self.model_save_path)

            if self.logger is not None:
                self.logger.info('{} Epoch(s) done. Loss: {}, LR: {}'.format(self.epoch,
                                                                           self.losses[-1],
                                                                           model.min_alpha_yet_reached))


class SDclassifier(object):

    def __init__(self,cocrs_path,
                 yr_SDs_path,
                 full_chems_path,
                 yrs_path):

        self.cocrs = np.loadtxt(cocrs_path)
        self.yr_SDs = np.loadtxt(yr_SDs_path)
        self.yrs = np.loadtxt(yrs_path)
        with open(full_chems_path, 'r') as f:
            full_chems = f.read().splitlines()
        self.full_chems = np.array(full_chems)

        
    def prepare_feature_vectors(self, train_yr,
                                memory,
                                neg_margin_yr,
                                w2v_model_path=None,
                                poly_deg=1,
                                start_yr=None,
                                logger=None):
        """Returning feature vectors for training a learning model to predict
        discoveries based on SD values of the previous years

        :parameters:

        * train_yr: int
            year of the training (the training vectors will be prepared
            using data before this year)

        * memory: int
            the memory to be used in SD-based prediction

        * neg_margin_yr: int
            number of years to consider before the training year
            as a margin for negative samples so that the data that 
            are too close to the prediction year won't make the 
            training biased as we are not aware of the real class
            of the unstudied materials

        * w2v_model_path: str (default is NONE)
            path to the word2vec model whose vocabulary will be used
            to extract the chemicals. If no path is given, the full set
            of chemicals will be used.

        * poly_deg: int (default is 1)
            polynomial degree of the features transformation.
            If no degree is given, no transformation will be used

        * start_yr: int (default is None)
            The year to start collecting positive features, used to
            avoid unreliable very early years.If not provided, the
            earliest year in self.yrs will be used

        * logger: object of type logging.RootLogger
            If no logger is provided, a null handler will be used 
            (hence no logging will be done)

        """

        self.train_yr = train_yr
        self.memory = memory
        
        # if no logger is provided simply use a NullHandler
        if logger is None:
            logger = helpers.set_up_logger(None, None, logger_disable=True)

        # get the vocabulary's chemicals OR    
        # use the full chemicals
        if w2v_model_path is None:
            self.model_chems = self.full_chems
            self.sub_cocrs = self.cocrs
            self.sub_yr_SDs = self.yr_SDs
        else:
            model = Word2Vec.load(w2v_model_path)
            model_chems = []
            for w in model.wv.index2word:
                if pr.is_simple_formula(w) and model.wv.vocab[w].count>3:
                    if (pr.normalized_formula(w)==w) or (w in ['H2','O2','N2']):
                        model_chems += [w]
            model_chems_indic_in_full = np.in1d(self.full_chems, np.array(model_chems))
            self.model_chems = self.full_chems[model_chems_indic_in_full]
            self.sub_cocrs  = self.cocrs[model_chems_indic_in_full,:]
            self.sub_yr_SDs = self.yr_SDs[model_chems_indic_in_full,:]


        """ preparing positive features """
        # ignore very early years
        if (start_yr is None) or (start_yr<self.yrs[0]):
            consider_data_after_yr = self.yrs[0]
        else:
            consider_data_after_yr = start_yr
            
        start_yr_loc = np.where(self.yrs==consider_data_after_yr)[0][0]
        yr_loc = np.where(self.yrs==train_yr)[0][0]
        pos_features = []
        # collect new discoveries in all the years preceding train_yr
        for y in np.arange(start_yr_loc,yr_loc):
            discoveries_before_yr = helpers.find_first_time_cocrs(self.sub_cocrs, y)
            features = self.sub_yr_SDs[discoveries_before_yr,y-memory:y]
            pos_features += [features]
        pos_features = np.concatenate(pos_features, axis=0)
        pos_features = pos_features[np.sum(pos_features,axis=1)>0,:]
        pos_n = pos_features.shape[0]
        logger.info('PROGRESS for year {}: {} positive features are collected'.format(
            train_yr, pos_n))

        """ preparing negative features """
        # take unstudied materials with existing signals in the period
        #    yr_start-neg_margin_yr-memory  -->  yr_start-neg_margin_yr
        unstudied_ents = np.sum(self.sub_cocrs[:,:yr_loc],axis=1)==0
        neg_idx = np.where(unstudied_ents)[0]
        negs_wsignals = neg_idx[
            np.sum(self.sub_yr_SDs[unstudied_ents,
                                   yr_loc-neg_margin_yr-memory : yr_loc-neg_margin_yr],
                   axis=1)>0
        ]
        # select roughly the same number of negative samples as the size of positive class
        neg_n = pos_n
        # if neg_n<pos_n it will take whatever it has 
        sub_negs_wsignals = negs_wsignals[np.random.permutation(len(negs_wsignals))[:neg_n]]
        neg_features = self.sub_yr_SDs[sub_negs_wsignals,
                                       yr_loc-neg_margin_yr-memory : yr_loc-neg_margin_yr]
        logger.info('PROGRESS for year {}: {} negative features are collected'.format(
            train_yr, neg_n))

        logger.info('PROGRESS for year {}: Polynomial of degree {} is used to transform the features.'.format(
            train_yr, poly_deg))
        self.poly = PolynomialFeatures(poly_deg)
        Xtrain = np.concatenate((neg_features, pos_features), axis=0)
        self.Xtrain = self.poly.fit_transform(Xtrain)
        self.ytrain = np.ones(self.Xtrain.shape[0])
        self.ytrain[:len(neg_features)] = -1

        
    def train(self, cv_folds=5):
        self.model = LogisticRegressionCV(cv=cv_folds).fit(self.Xtrain, self.ytrain)

        
    def predict(self, pred_yr, pred_size=50):
        
        assert self.train_yr <= pred_yr, 'Prediction year cannot be before \
                                          the training year.'

        pred_yr_loc = np.where(self.yrs==pred_yr)[0][0]
        unstudied_ents = np.sum(self.sub_cocrs[:,:pred_yr_loc],axis=1)==0
        neg_idx = np.where(unstudied_ents)[0]
        
        # use the same co-occurrence and SD matrices already
        # saved in the class to extract the unstudied materials
        # and doin the prediction
        # -------------
        # throw away materials without signal in the considered period
        # and do the prediction over those with non-zero SDs
        self.unstudied_wsignal = neg_idx[np.sum(
            self.sub_yr_SDs[unstudied_ents,
                            pred_yr_loc-self.memory : pred_yr_loc], axis=1)>0]
        Xtest = self.sub_yr_SDs[self.unstudied_wsignal,pred_yr_loc-self.memory:pred_yr_loc]
        self.Xtest = self.poly.fit_transform(Xtest)
        ytest_scores = self.model.decision_function(self.Xtest)
        # ranking based on the scores
        preds = self.unstudied_wsignal[np.argsort(-ytest_scores)[:pred_size]]

        return ytest_scores, preds
        
