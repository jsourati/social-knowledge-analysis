import os
import sys
import pdb

import numpy as np
import pandas as pd

from pybliometrics.scopus import AbstractRetrieval
from data import utils


def find_authors(abst_path, dois_path, entries):
    """Listing all the authors who have at least one publlication which
    contains at least one of the given entries

    **Parameters:**

    * `abst_paths`: (str) path to abstracts file
    * `dois_path`: (str) path to the list of DOIs
    * `entries`: (list) list of strings, each string is an entry

    ** Returns:

    * `u_auids`: (list) set of author IDs for those who had at least
                published one paper that contained one of the entries
    * `au_dois`: (list) list of DOIs of the authors that were identified
               (same length as `u_auids)`
    """

    p = utils.MatTextProcessor()

    # domain of the search (DOIs)
    doi_list = pd.read_csv(dois_path, header=None)

    auids = []
    dois = []
    with open(abst_path, 'r', encoding='utf-8') as f:
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


def JSD(abst_path, dois_path, E1, E2, silent=False):
    """Calculating Jaccardian Social Density (JSD) between
    two sets of strings, possibly a molecule and a property

    **Parameters:**

    * `abst_paths`: (str) path to abstracts file
    * `dois_path`: (str) path to the list of DOIs
    * `E1`: (list) set of strings representing the first entry
    * `E2`: (list) set of strings representing the second entry
    * `silent`: (bool) flag for printing on screen or being silent
    """

    if type(E1)=='str':
        E1 = [E1]
    if type(E2)=='str':
        E2 = [E2]

    A1,_ = find_authors(abst_path, dois_path, E1)
    if not(silent):
        print('{} number of authors are identified for entries {}'.format(len(A1),E1))
    A2,_ = find_authors(abst_path, dois_path, E2)
    if not(silent):
        print('{} number of authors are identified for entries {}'.format(len(A2),E2))

    if len(A1)==0 or len(A2)==0:
        JSD = 0
    else:
        overlap = set(A1).intersection(set(A2))
        JSD =  len(overlap) / (len(A1) + len(A2))

    if not(silent):
        print('JSD is {}'.format(JSD))

    return JSD

    
