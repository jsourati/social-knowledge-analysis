import os
import sys
import pdb
import json
import logging
import numpy as np

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
                self.logger.info('{} Epoch(s) done. Loss: {}'.format(self.epoch,
                                                                     self.losses[-1]))
