import os
import sys
import pdb

import numpy as np
import pandas as pd

# This module uses a text-processor class provided by mat2vec project
# specifically designed for materials science articles
mat2vec_path = '~/scratch-midway2/repos/mat2vec'

from pybliometrics.scopus import AbstractRetrieval
from mat2vec.processing.process import MaterialsTextProcessor

class MatTextProcessor(MaterialsTextProcessor):

    def mat_preprocess(self, text):
        """Pre-processing a given text using tools provided by 
        the repository mat2vec 


        """

        tokens = self.tokenize(text)
        ptokens = [self.process(token, make_phrases=True)[0] for token in tokens]

        return ptokens   

    def make_training_file(self, dois, save_dir):
        """Downloading, pre-processsing and storing abstracts of a set
        of DOIs in a text file which can be later used as the training data
        for tuning models like word2vec

        Each line of the saved file corresponds to one article and shows
        its title followed by the abstract

        ** Parameters:
            * dois : *(list)* list of DOIs
            * saved_dir : *(str)* directory to save the files
        """

        # list of lists (each list = one line = title + abstract)
        save_path_abst = os.path.join(save_dir, 'abstracts')
        save_path_dois = os.path.join(save_dir, 'saved_DOIs')
        saved_dois = []
        for doi in dois:
            try:
                r = AbstractRetrieval(doi)
            except:
                continue

            tokens = self.mat_preprocess(r.title) + self.mat_preprocess(r.description)
            line = ' '.join(sum(tokens,[]))
            doi_line = doi
            if doi!=dois[-1]:
                line += '\n'
                doi_line += '\n'

            # saving the texts
            with open(save_path_abst, 'a+', encoding='utf-8') as f:
                f.write(line)
            with open(save_path_dois, 'a+') as f:
                f.write(doi_line)
