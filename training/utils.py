import os
import sys
import numpy as np

import gensim

path = '/home/jamshid/codes/social-knowledge-analysis/'
sys.path.insert(0, path)
from data import utils

pr = utils.MatTextProcessor()


def keep_simple_formula(word, count, min_count):
    if pr.is_simple_formula(word):
        return gensim.utils.RULE_KEEP
    else:
        return gensim.utils.RULE_DEFAULT

    
def keep_certain_terms(word, count, min_count, terms):
    if word in terms:
        return gensim.utils.RULE_KEEP
    else:
        return gensim.utils.RULE_DEFAULT
    

def trim_for_deepwalks(word, count, min_count):
    if 'a_' in word:
        return gensim.utils.RULE_DISCARD 
    else:
        return gensim.utils.RULE_KEEP

    
def threshold_chems_by_counts(chems, model, count_threshold):
    """Filtering a set of chemicals by removing those whose frequency in 
    the vocabulary of the given model is less than a threshold
    """

    model_chems = []
    for w in model.wv.index2word:
        if pr.is_simple_formula(w) and model.wv.vocab[w].count>count_threshold:
            if (pr.normalized_formula(w)==w) or (w in ['H2','O2','N2']):
                model_chems += [w]

    # get the intersection between the extracted chemicals and the full set
    model_chems_indic_in_full = np.in1d(chems, model_chems)
    sub_chems = chems[model_chems_indic_in_full]

    return sub_chems


def preprocess_corpus(C, normalize_materials=False):
    """Preparing a corpus of texts, e.g. set of abstracts, for
    training a word2vec model

    This preprocessing function is mainly using the "process" function of 
    `MaterialsTextProcessor` class developed by `mat2vec` team, and
    it includes removing numbers and punctuations, and normalizing the
    materials formula if needed. 

    The corpus should be given in form of a list of strings.
    """

    prep_C = []
    for text in C:
        prep_C += [' '.join(pr.process(text.replace('Inf','inf'),
                                       exclude_punct=True,
                                       convert_num=True,
                                       normalize_materials=normalize_materials,
                                       make_phrases=False)[0])]

    return prep_C
    
