import os
import sys
import pdb

import numpy as np
import pandas as pd
from itertools import groupby

# This module uses a text-processor class provided by mat2vec project
# specifically designed for materials science articles
mat2vec_path = '~/scratch-midway2/repos/mat2vec'

from pybliometrics.scopus import AbstractRetrieval
from pybliometrics.scopus.exception import Scopus429Error
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
        save_path_misses = os.path.join(save_dir, 'missed_DIOs')
        missed_dois = []
        for doi in dois:
            try:
                r = AbstractRetrieval(doi)
                tokens = self.mat_preprocess(r.title) + self.mat_preprocess(r.description)
            except:
                #pdb.set_trace()
                with open(save_path_misses, 'a+', encoding='utf-8') as f:
                    f.write(doi+'\n')
                continue
            
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

def Scopus_to_SQLtable(dois,
                       sql_db, 
                       sql_cursor, 
                       bad_dois_save_path=None):

    # get the last primary paper/author IDs
    sql_cursor.execute('SELECT paper_id FROM paper;')
    all_paper_PKs = sql_cursor.fetchall()
    if len(all_paper_PKs)==0:
        paper_PK = 0
    else:
        paper_PK = max([a[0] for a in all_paper_PKs]) + 1

    sql_cursor.execute('SELECT author_id FROM author;')
    all_author_PKs = sql_cursor.fetchall()
    if len(all_author_PKs)==0:
        author_PK = 0
    else:
        author_PK = max([a[0] for a in all_author_PKs]) + 1
        


    # all previously entered paper DOIs to avoid repetition
    sql_cursor.execute('SELECT doi FROM paper;')
    all_dois = sql_cursor.fetchall()
    all_dois = [a[0] for a in all_dois]
    # ... same for authors
    sql_cursor.execute('SELECT author_scopus_ID FROM author')
    curr_scopus_id_list = [a[0] for a in sql_cursor.fetchall()]

    bad_dois = []
    for i,doi in enumerate(dois):
        if doi in all_dois:
            print('{} is already entered to the database'.format(doi))
            continue

        try:
            r = AbstractRetrieval(doi)
        except Scopus429Error:
            print('Scopus resource exhausted. Check your quota.')
            return
        except:
            bad_dois += [doi]
            if bad_dois_save_path is not None:
                with open(bad_dois_save_path, 'a+') as bad_f:
                    bad_f.write(doi+'\n')
            continue
            

        # ROW IN PAPER TABLE
        title = r.title.replace('\"','')
        if r.description is not None:
            abst = r.description.replace('\"','')
        else:
            abst = 'NA'
            
        scomm = """INSERT INTO paper VALUES({},"{}","{}","{}","{}");""".format(
            paper_PK,
            r.doi,
            r.coverDate,
            title,
            abst)
        # taking care of unicode characters
        #scomm = "{}".format(scomm.encode('utf-8'))
        #scomm = scomm[2:-1].replace('\\', '\\\\')

        sql_cursor.execute(scomm)


        # ROW IN AUTHOR TABLE
        # skip the rest if no auhotrs were available
        if r.authors is None:
            paper_PK += 1
            continue
        paper_scopus_id_list = [a.auid for a in r.authors]
        for i,scps_id in enumerate(paper_scopus_id_list):
            # if repetitive author, ignore:
            if scps_id in paper_scopus_id_list[:i]:
                continue
            
            if scps_id in curr_scopus_id_list:
                # extract existing author PK from scopus ID
                sql_cursor.execute('SELECT author_id \
                                    FROM author \
                                    WHERE author_scopus_ID = {}'.format(scps_id))
                this_author_PK = sql_cursor.fetchall()[0][0]
                sql_cursor.execute('INSERT INTO paper_author_mapping VALUES({}, {})'.format(
                    paper_PK, this_author_PK))
            else:
                # create a row for this new author
                sql_cursor.execute('INSERT INTO author \
                                    VALUES({}, "{}", "{}", "{}")'.format(
                                        author_PK,
                                        scps_id,
                                        r.authors[i].given_name,
                                        r.authors[i].surname)
                                   )
                sql_cursor.execute('INSERT INTO paper_author_mapping \
                                    VALUES({}, {})'.format(
                                        paper_PK, author_PK))
                # update the global authors scopus ID list
                curr_scopus_id_list += [scps_id]
                author_PK += 1
        paper_PK += 1

        sql_db.commit()

    return bad_dois

    
