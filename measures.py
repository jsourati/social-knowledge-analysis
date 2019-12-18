import os
import sys
import pdb

import json
import pymysql
import numpy as np

def naive_authors_listing_from_terms(term, db, crsr):
    """Getting list of authors who have used the given term
    in the abstract of at least one of their papers
    """

    #scomm = 'SELECT paper_author_mapping.author_id FROM \
    #         paper, paper_author_mapping WHERE \
    #         paper.paper_id=paper_author_mapping.paper_id AND \
    #         (paper.abstract LIKE BINARY "%{}%" OR \
    #         paper.title LIKE BINARY "%{}%")'.format(term,term)
    scomm = 'SELECT P2A.author_id \
             FROM paper_author_mapping P2A \
             INNER JOIN paper P \
             ON P2A.paper_id=P.paper_id \
             WHERE (P.abstract LIKE BINARY "%{}%" OR \
             P.title LIKE BINARY "%{}%")'.format(term,term)

    crsr.execute(scomm)
    A = crsr.fetchall()
    return [a[0] for a in A]
    
