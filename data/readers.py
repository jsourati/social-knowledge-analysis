import os
import sys
import pdb
import json
import pymysql
import numpy as np

path = '/home/jamshid/codes/social-knowledge-analysis/'
sys.path.insert(0, path)
from data import utils

class MatScienceDB(object):

    def __init__(self, config_path, DB_name):
        
        with open(config_path,'r') as f:
            self.configs = json.load(f)
        self.client_config = self.configs['client_config']
        self.db = pymysql.connect(**self.client_config)
        self.crsr = self.db.cursor()
        self.db_name = DB_name
        self.crsr.execute('USE {};'.format(self.db_name))

    def re_establish_connection(self):
        self.db.close()
        self.db = pymysql.connect(**self.client_config)
        self.crsr = self.db.cursor()
        self.crsr.execute('USE {};'.format(self.db_name))
        
    def count_table_rows(self, table_name):
        self.crsr.execute('SELECT COUNT(*) FROM {};'.format(table_name))
        return self.crsr.fetchall()[0][0]

    
    def get_authors_by_paper_id(self, paper_ids, cols):

        cols,pcols = self.prepare_column_headers(cols, 'A')

        if type(paper_ids) is int:
            # single paper ID
            scomm = 'SELECT {} \
                     FROM author A \
                     INNER JOIN paper_author_mapping P2A \
                     ON P2A.author_id=A.author_id \
                     WHERE P2A.paper_id={}'.format(pcols, paper_ids)
        else:
            # a list of paper IDs
            ID_list_str = '({})'.format(', '.join([str(a) for a in paper_ids]))
            scomm = 'SELECT {} \
                     FROM author A \
                     INNER JOIN paper_author_mapping P2A \
                     ON P2A.author_id=A.author_id \
                     WHERE P2A.paper_id IN {}'.format(pcols, ID_list_str)
            
        return self.execute_and_get_results(scomm, cols)


    def get_authors_by_paper_constraints(self, constraints, cols):

        # preparation columns of author table
        cols, pcols = self.prepare_column_headers(cols, 'A')

        # using paper columns to prepare the constraints' text
        self.crsr.execute('DESCRIBE paper;')
        paper_cols = [a[0] for a in self.crsr.fetchall()]
        for col in paper_cols:
            constraints = constraints.replace(col,'P.{}'.format(col))

        # nested INNER JOINs to to apply constraints on columns
        # of paper (P) that the mapping table (P2A) does not have
        scomm = 'SELECT {} \
                 From author A \
                 INNER JOIN paper_author_mapping P2A \
                 ON P2A.author_id=A.author_id \
                 INNER JOIN paper P \
                 ON P.paper_id=P2A.paper_id \
                 WHERE {};'.format(pcols, constraints)

        return self.execute_and_get_results(scomm, cols)

    
    def get_papers_by_author_id(self, author_ids, cols):

        cols, pcols = self.prepare_column_headers(cols, 'P')
        if type(author_ids) is int:
            scomm = 'SELECT {} \
                     FROM paper P \
                     INNER JOIN paper_author_mapping P2A \
                     ON P2A.paper_id=P.paper_id \
                     WHERE P2A.author_id={}'.format(pcols, author_ids)
        else:
            # a list of author IDs
            ID_list_str = '({})'.format(', '.join([str(a) for a in author_ids]))
            scomm = 'SELECT {} \
                     FROM paper P \
                     INNER JOIN paper_author_mapping P2A \
                     ON P2A.paper_id=P.paper_id \
                     WHERE P2A.author_id IN {}'.format(pcols, ID_list_str)

        return self.execute_and_get_results(scomm, cols)
    

    def get_affiliations_by_author_id(self, author_id, cols):
        
        cols, pcols = self.prepare_column_headers(cols, 'AFF')
        scomm = 'SELECT {} \
                 FROM affiliation AFF \
                 INNER JOIN author_affiliation_mapping A2AFF \
                 ON AFF.aff_id=A2AFF.aff_id \
                 WHERE A2AFF.author_id={}'.format(pcols, author_id)
        
        return self.execute_and_get_results(scomm, cols)


    def get_papers_by_chemicals(self, chemical_formula, cols=None):
        """Returning papers whose titles/abstracts have at least one of
        the given chemical formula
        """

        cols, pcols = self.prepare_column_headers(cols, 'P')
        chemical_formula = ['"{}"'.format(c) for c in chemical_formula]
        chems_array_str = '({})'.format(', '.join(chemical_formula))

        scomm = 'SELECT {} \
                 FROM paper P \
                 INNER JOIN chemical_paper_mapping C2P \
                 ON P.paper_id=C2P.paper_id \
                 INNER JOIN chemicals C \
                 ON C2P.chem_id=C.chem_id \
                 WHERE C.formula IN {}'.format(pcols, chems_array_str)

        return self.execute_and_get_results(scomm, cols)


    def get_authors_by_chemicals(self, chemical_formula, cols=None):
        """Getting the list of authors who (co)-authored papers that include
        one of the given chemicals in their titles/abstracts
        """

        cols, pcols = self.prepare_column_headers(cols, 'A')
        chemical_formula = ['"{}"'.format(c) for c in chemical_formula]
        chems_array_str = '({})'.format(', '.join(chemical_formula))

        scomm = 'SELECT {} \
                 FROM author A \
                 INNER JOIN paper_author_mapping P2A \
                 ON P2A.author_id=A.author_id \
                 INNER JOIN chemical_paper_mapping C2P \
                 ON C2P.paper_id=P2A.paper_id \
                 INNER JOIN chemicals C \
                 ON C.chem_id=C2P.chem_id \
                 WHERE C.formula IN {};'.format(pcols, chems_array_str)

        return self.execute_and_get_results(scomm, cols)
    
        
    def get_chemicals_by_paper_id(self, paper_ids):
        """Returning a list of chemicals that are present in the 
        titles/abstracts of the given papers
        """

        if type(paper_ids) is int:
            # single paper ID
            scomm = 'SELECT C.chem_id, C.formula \
                     FROM chemicals C \
                     INNER JOIN chemical_paper_mapping C2P \
                     ON C2P.chem_id=C.chem_id \
                     WHERE C2P.paper_id={}'.format(paper_ids)
        else:
            # a list of paper IDs
            ID_list_str = '({})'.format(', '.join([str(a) for a in paper_ids]))
            scomm = 'SELECT C.chem_id, C.formula \
                     FROM chemicals C \
                     INNER JOIN chemical_paper_mapping C2P \
                     ON C2P.chem_id=C.chem_id \
                     WHERE C2P.paper_id IN {}'.format(ID_list_str)
            
        return self.execute_and_get_results(scomm, ['chem_id', 'formula'])

    def get_papers_by_keywords(self, keywords, cols=None, logical_comb='OR'):
        """Returning papers that have the given list of keywords. 

        The logical combination (`logical_comb`) input specifies if
        the papers should have all the keywords at the same time (AND)
        or just having one of the keywords suffices (OR).
        """

        cols, pcols = self.prepare_column_headers(cols, 'P')
        constraints_str = ['P.abstract LIKE "%{}%"'.format(k) for k in keywords]
        constraints_str = ' {} '.format(logical_comb).join(constraints_str)

        scomm = 'SELECT {} \
                 FROM paper P \
                 WHERE {};'.format(pcols, constraints_str)
        
        return self.execute_and_get_results(scomm, cols)

    
    def get_authors_by_keywords(self, keywords, cols, logical_comb='OR'):
        """Returning authors who have paper with the given keywords mentioned
        in their titles/abstracts
        """

        cols, pcols = self.prepare_column_headers(cols, 'A')
        constraints_str = ['P.abstract LIKE "%{}%"'.format(k) for k in keywords]
        constraints_str = ' {} '.format(logical_comb).join(constraints_str)

        scomm = 'SELECT {} \
                 FROM author A \
                 INNER JOIN paper_author_mapping P2A \
                 ON P2A.author_id=A.author_id \
                 INNER JOIN paper P \
                 ON P.paper_id=P2A.paper_id \
                 WHERE {};'.format(pcols, constraints_str)

        return self.execute_and_get_results(scomm, cols)

    
    def prepare_column_headers(self, cols, prefix):
       
        if cols is None:
            cols = ['paper_id']
            pcols = '{}.paper_id'.format(prefix)
        else:
            pcols = ['{}.{}'.format(prefix, col) for col in cols]
            pcols = ', '.join(pcols)

        return cols, pcols

    def execute_and_get_results(self, scomm, cols):

        # output shape: [(a1,b1),(a2,b2)]
        #                 ------  ------
        #                  Row1    Row2
        self.crsr.execute(scomm)
        
        # [(a1,b1),(a2,b2)] --> [(a1,a2), (b1,b2)]
        # |---------------|      |--------------|
        #  row-wise listing      colunmwise listing 
        R = zip(*self.crsr.fetchall())
        # [A,B] + [(a1,a2), (b1,b2)] --> [(A,(a1,a2)), (B,(b1,b2))]
        # .. and then use column headers to make a dictionary
        return {x[0]:list(x[1]) for x in zip(cols,R)}


    def extract_titles_abstracts(self,
                                 before_year=None,
                                 em='RAM',
                                 save_path=None):
        """Returning titles and abstracts (merged together) as a list
        of lists (when `em=RAM`) or saving them into lines of a text file
        (when `em=HARD`). If the latter is specified, a path for saving the text
        file (`save_path`) should also be provided.
        """

        # MS text processor
        self.text_processor = utils.MatTextProcessor()

        if before_year:
            scomm = 'SELECT title, abstract FROM paper \
                     WHERE YEAR(date)<{};'.format(before_year)
        else:
            scomm = 'SELECT title, abstract FROM paper;'
        (_,titles), (_,abstracts) = self.execute_and_get_results(scomm, ['title','abstract']).items()
        
        if em=='HARD':
            # processing and saving
            assert save_path is not None, 'Specify a saving path.'

            with open(save_path, 'a') as f:
                for i in range(len(titles)):
                    A = titles[i] + '. ' + abstracts[i]
                    if 'Inf' in A:
                        A = A.replace('Inf', 'inf')
                    prA = ' '.join(sum(self.text_processor.mat_preprocess(A), []))
                    f.write(prA + '\n')

        elif em=='RAM':
            texts = []
            for i in range(len(titles)):
                A = titles[i] + '. ' + abstracts[i]
                prA = ' '.join(sum(self.text_processor.mat_preprocess(A), []))
                texts += [prA]
            return texts
        
