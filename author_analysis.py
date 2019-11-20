import os
import sys
import pdb

import numpy as np
import pandas as pd

from pybliometrics.scopus import AbstractRetrieval
from data import utils


def find_authors(abstracts_path, dois_path, entries):
    """Listing all the authors who have at least one publlication which
    contains at least one of the given entries
    """

    p = utils.MatTextProcessor()

    # domain of the search (DOIs)
    doi_list = pd.read_csv(dois_path, header=None)

    auids = []
    dois = []
    with open(abstracts_path, 'r', encoding='utf-8') as f:
        for i,line in enumerate(f):
            if np.any([e in line for e in entries]):
                abst = line.split(' ')
                if np.any([p.normalized_formula(e) in abst 
                           for e in entries]):
                    dois += [doi_list.iloc[i][0]]
                    doc = AbstractRetrieval(dois[-1]) 
                    auids += [[a.auid for a in doc.authors]]
                

    # unique authors and their documents
    u_auids = list(np.unique(np.array(sum(auids,[]))))
    au_dois = [[dois[j] for j in range(len(dois)) if au in auids[j]] 
               for au in u_auids]

    return u_auids, au_dois

