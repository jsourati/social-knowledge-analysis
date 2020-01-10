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


    
    
