import os
import sys
import pdb
import json
import pickle
import pymysql
import numpy as np

path = '/home/jamshid/codes/social-knowledge-analysis/'
sys.path.insert(0, path)
from data import utils
from misc.helpers import set_up_logger, find_first_time_cocrs

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
                 INNER JOIN chemical C \
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
                 INNER JOIN chemical C \
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
                     FROM chemical C \
                     INNER JOIN chemical_paper_mapping C2P \
                     ON C2P.chem_id=C.chem_id \
                     WHERE C2P.paper_id={}'.format(paper_ids)
        else:
            # a list of paper IDs
            ID_list_str = '({})'.format(', '.join([str(a) for a in paper_ids]))
            scomm = 'SELECT C.chem_id, C.formula \
                     FROM chemical C \
                     INNER JOIN chemical_paper_mapping C2P \
                     ON C2P.chem_id=C.chem_id \
                     WHERE C2P.paper_id IN {}'.format(ID_list_str)
            
        return self.execute_and_get_results(scomm, ['chem_id', 'formula'])

    def get_papers_by_keywords(self, keywords,
                               cols=None,
                               logical_comb='OR',
                               case_sensitives=[]):
        """Returning papers that have the given list of keywords. 

        The logical combination (`logical_comb`) input specifies if
        the papers should have all the keywords at the same time (AND)
        or just having one of the keywords suffices (OR).

        The list `case_sensitives` also includes the keywords that needs to
        be searched in a case-sensitive fashion. 
        """

        cols, pcols = self.prepare_column_headers(cols, 'P')
        constraints_str = ['P.abstract LIKE "%{}%"'.format(k) for k in keywords]
        for k in case_sensitives:
            idx = np.where([x==k for x in keywords])
            idx = [] if len(idx)==0 else idx[0][0]
            constraints_str[idx] = 'P.abstract LIKE BINARY "%{}%"'.format(k)
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

    def get_yearwise_authors_by_keywords(self, terms,
                                         chemical=False,
                                         min_yr=None,
                                         max_yr=None,
                                         return_papers=True,
                                         case_sensitives=[]):


        # For chemicals, we could get X-authors directly, but since we want to
        # compute year-wise SDs, we will first get X-papers and their dates
        # and then get their authors
        if chemical:
            res_dict = self.get_papers_by_chemicals(terms, ['paper_id','date'])
        else:
            res_dict = self.get_papers_by_keywords(terms,
                                                   ['paper_id','date'],
                                                   'OR',
                                                   case_sensitives)


        if len(res_dict)>0:
            (_,papers), (_,dates) = res_dict.items()
        else:
            return {}

        dates_yrs = np.array([d.year for d in dates])
        if min_yr is None:    
            min_yr = np.min(dates_yrs)
        if max_yr is None:
            max_yr = np.max(dates_yrs)

        yr_authors = {yr: [] for yr in np.arange(min_yr,max_yr+1)}

        for yr in np.arange(min_yr, max_yr+1):
            yr_papers = [papers[i] for i in np.where(dates_yrs==yr)[0]]
            if len(yr_papers)>0:
                auths = self.get_authors_by_paper_id(yr_papers, ['author_id'])
                # taking care of author-less papers
                if len(auths)>0:
                    yr_authors[yr] = list(np.unique(auths['author_id']))

        if return_papers:
            papers_dates_dict = {}
            for yr in np.arange(min_yr, max_yr+1):
                papers_dates_dict[yr] = list(np.array(papers)[dates_yrs==yr])
            return yr_authors, papers_dates_dict
        else:
            return yr_authors

        
    def collect_authors_new_discoveries(self, full_chems,
                                        cocrs,
                                        yr_SDs,
                                        Y_terms,
                                        yrs,
                                        case_sensitives=[],
                                        logfile_path=None,
                                        savefile_path=None):
        """Collecting authors of papers with new co-occurrences (new discoveries)
        and extracting their previous papers on the topic of the property and/or
        the newly studied molecule
        """

        logger = set_up_logger(__name__,logfile_path,False)

        yr_Y_authors, yr_Y_papers = self.get_yearwise_authors_by_keywords(
            Y_terms, return_papers=True, case_sensitives=case_sensitives)

        # analyze years from 2001 to 2018 (note that: yrs[-1]=2019)
        start_yr = 2001
        disc_dict = {}
        for yr in np.arange(start_yr, yrs[-1]):
            yr_loc = np.where(yrs==yr)[0][0]
            thisyr_Y_papers = yr_Y_papers[yr]

            disc_dict[yr] = {}
            new_discs = find_first_time_cocrs(cocrs, yr_loc)
            logger.info('PROGRESS FOR {}: {} new discoveries found'.format(yr,len(new_discs)))
            for i,chm in enumerate(full_chems[new_discs]):
                yr_X_authors, yr_X_papers = self.get_yearwise_authors_by_keywords(
                    [chm], chemical=True, return_papers=True)
                thisyr_X_papers = yr_X_papers[yr]

                # papers with co-occurrences
                ov_papers = list(set(thisyr_Y_papers).intersection(set(thisyr_X_papers)))
                disc_dict[yr][chm] = {pid:{} for pid in ov_papers}
                for pid in ov_papers:
                    # authors of papers with co-occurrences
                    A = self.get_authors_by_paper_id([pid],['author_id'])
                    if len(A)>0: A=A['author_id']
                    disc_dict[yr][chm][pid]  = {a:[{},{}] for a in A}

                    for auth in A:
                        """ for the property """
                        # years that the current author has published a paper on Y so that ..
                        a_pubY_yrs = [y for y in yr_Y_authors if auth in yr_Y_authors[y] and y<yr]
                        if len(a_pubY_yrs)>0:
                            # .. we can consider only those years to query his/her papers
                            array_yrs = '({})'.format(','.join([str(y) for y in a_pubY_yrs]))
                            scomm = 'SELECT P.paper_id, YEAR(P.date) FROM paper P \
                                     INNER JOIN paper_author_mapping P2A ON P.paper_id=P2A.paper_id \
                                     WHERE P2A.author_id={} AND (YEAR(P.date) IN {})'.format(auth, array_yrs)
                            # Pa and Ya are the papers and years of those papers
                            (_,Pa),(_,Ya) = self.execute_and_get_results(scomm,['paper_id','year']).items()
                            uYa = np.unique(Ya)
                            disc_dict[yr][chm][pid][auth][0] = {yr: [Pa[i] for i in range(len(Pa))
                                                                     if Ya[i]==yr
                                                                     if Pa[i] in yr_Y_papers[yr]] for yr in uYa}
                            
                            
                        """ for the molecule """
                        a_pubX_yrs = [x for x in yr_X_authors if auth in yr_X_authors[x] and x<yr]
                        if len(a_pubX_yrs)>0:
                            array_yrs = '({})'.format(','.join([str(x) for x in a_pubX_yrs]))
                            scomm = 'SELECT P.paper_id, YEAR(P.date) FROM paper P \
                                     INNER JOIN paper_author_mapping P2A ON P.paper_id=P2A.paper_id \
                                     WHERE P2A.author_id={} AND (YEAR(P.date) IN {})'.format(auth, array_yrs)
                            (_,Pa),(_,Ya) = self.execute_and_get_results(scomm,['paper_id','year']).items()
                            uYa = np.unique(Ya)
                            disc_dict[yr][chm][pid][auth][1] = {yr: [Pa[i] for i in range(len(Pa))
                                                                     if Ya[i]==yr
                                                                     if Pa[i] in yr_X_papers[yr]] for yr in uYa}
                            
                if i>0 and not(i%100):
                    logger.info('\t{} materials have been analyzed'.format(i))

            if savefile_path is not None:
                with open(savefile_path, 'wb') as f:
                    pickle.dump(disc_dict, f)
                logger.info('The results have been saved in {}'.format(savefile_path))
                
        return disc_dict
