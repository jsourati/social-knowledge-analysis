import os
import re
import sys
import pdb
import json
import pickle
import pymysql
import numpy as np

path = '/home/jamshid/codes/social-knowledge-analysis/'
sys.path.insert(0, path)
from data import utils
from misc.helpers import set_up_logger #, find_first_time_cocrs

class DB(object):
    """Class of databases that we will be wokring for running/evaluating
    our predictions

    Compatibility Requirements:
    ---------------------------
    The database should have at least five tables:
    - paper: with at least four columns (id, date, title, abstract)
    - author: with at least one column (id)
    - entity: with at least two columns (id, name)--column "name" could
    be titled differently, but if that is the case it should be specified
    when constructing class object via input argument "entity_col"
    - paper_author_mapping: with columns (paper_id, author_id)
    - entity_paper_mapping: with columns (entity_id, paper_id)

    Table "entity" can have a different name. If that is the case, the table
    "entity_paper_mapping" should be renamed to the same title, and also the 
    input argument "entity_tab" should be set accordingly. For exampe, if
    the entity table is named as "chemical", then the mapping table should 
    be named as chemical_paper_mapping with columns (chemical_id, paper_id).

    NOTE: we intentionally did not use MySQL's `GROUP_CONCAT` when returning
    list of papers, authors or entities so that we won't be restricted by
    the variable `group_concat_max_len`.
    
    """

    def __init__(self, config_path, db_name, **kwargs):

        # name of entity table
        self.entity_tab = kwargs.get('entity_tab', 'entity')
        # column of names (symbols, etc) in entity table
        self.entity_col = kwargs.get('entity_col', 'name')
        
        with open(config_path,'r') as f:
            self.configs = json.load(f)
        self.client_config = self.configs['client_config']
        self.db = pymysql.connect(**self.client_config)
        self.crsr = self.db.cursor()
        self.crsr.execute('USE {};'.format(db_name))

        
    def re_establish_connection(self):
        self.db.close()
        self.db = pymysql.connect(**self.client_config)
        self.crsr = self.db.cursor()
        self.crsr.execute('USE {};'.format(self.db_name))


    def count_table_rows(self, table_name):
        self.crsr.execute('SELECT COUNT(*) FROM {};'.format(table_name))
        return self.crsr.fetchall()[0][0]

    
    def get_1d_query(self, scomm):
        self.crsr.execute(scomm)
        return np.array([x[0] for x in self.crsr.fetchall()])

    
    def get_LoA_by_PID(self, paper_ids, **kwargs):

        cols = kwargs.get('cols', ['id'])
        cols = ['A.{}'.format(col) for col in cols] + ['P2A.paper_id']
        pcols = ','.join(cols)
        
        if type(paper_ids) is int:
            paper_ids = [paper_ids]
        # a list of paper IDs
        ID_list_str = '({})'.format(', '.join([str(a) for a in paper_ids]))
        constraints_str = 'P2A.paper_id IN {}'.format(ID_list_str)

        scomm = 'SELECT {} FROM author A \
                 INNER JOIN paper_author_mapping P2A ON P2A.author_id=A.id \
                 WHERE {}'.format(pcols, constraints_str)
            
        R = self.execute_and_get_results(scomm, cols)
        if len(R)==0:
            return []
        
        A = R['P2A.paper_id']
        out = {}
        for a in np.unique(A):
            out[a] = {}
            for i,col in enumerate(cols[:-1]):
                name = col.split('.')[1]
                out[a][name] = np.array(R[col])[A==a]
        return out

    def get_NoA_by_PID(self, paper_ids):
        """Getting Number of Authors (NoA) of a set of paper IDs

        This can be basically done by counting the authors retuned by the previous method.
        However, we are adding a separate method for this purpose because it
        can be done using GROUP BY command of mySQL

        """

        if type(paper_ids) is int:
            paper_ids = [paper_ids]
        # a list of paper IDs
        ID_list_str = '({})'.format(', '.join([str(a) for a in paper_ids]))
        constraints_str = 'P2A.paper_id IN {}'.format(ID_list_str)

        scomm = 'SELECT P2A.paper_id, COUNT(*) FROM author A \
                 INNER JOIN paper_author_mapping P2A ON P2A.author_id=A.id \
                 WHERE {} GROUP BY P2A.paper_id;'.format(constraints_str)

        self.crsr.execute(scomm)
        res = self.crsr.fetchall()

        return dict(res)

    
    def get_LoA_by_ents(self, ent_names, **kwargs):
        """Getting the list of authors who (co)-authored papers that include
        one of the given chemicals in their titles/abstracts

        This is an indirect querying. That is, chemicals are not directly related
        to the authors, but via papers. Therefore, there will be an option of 
        returning papers too.
        """

        cols = kwargs.get('cols', ['id'])
        years = kwargs.get('years', [])
        return_papers = kwargs.get('return_papers', False)

        cols = [col if '.' in col else 'A.{}'.format(col) for col in cols] + \
            ['E.{}'.format(self.entity_col)]
        pcols = ','.join(cols)
        if return_papers:
            pcols += ',P.id AS paper_id'
            cols += ['.paper_id']
        

        ent_names = ['"{}"'.format(c) for c in ent_names]
        constraints_str = 'E.{} IN ({})'.format(self.entity_col, ','.join(ent_names))

        if len(years)>0:
            yrs_arr = ','.join([str(x) for x in years])
            constraints_str = '({}) AND YEAR(P.date) IN ({})'.format(
                constraints_str, yrs_arr)

        scomm = 'SELECT {} FROM author A \
                 INNER JOIN paper_author_mapping P2A ON P2A.author_id=A.id \
                 INNER JOIN {}_paper_mapping E2P ON E2P.paper_id=P2A.paper_id \
                 INNER JOIN paper P ON P.id=P2A.paper_id \
                 INNER JOIN {} E ON E.id=E2P.{}_id \
                 WHERE {};'.format(pcols, self.entity_tab, self.entity_tab,
                                   self.entity_tab, constraints_str)

        cols = [col.split('.')[1] for col in cols]
        R = self.execute_and_get_results(scomm, cols)
        if len(R)==0:
            return []

        A = np.array(R[self.entity_col])
        cols.remove(self.entity_col)
        out = {}
        for a in np.unique(A):
            out[a] = {}
            for i,col in enumerate(cols):
                out[a][col] = np.array(R[col])[A==a]
        return out

    
    def get_authors_by_keywords(self, keywords, **kwargs):
        """Returning authors who have paper with the given keywords mentioned
        in their titles/abstracts

        This is an indirect querying. That is, keywords are not related to authors
        directly but via papers. Thereforem there will be an option of returning 
        papers too.
        """

        cols = kwargs.get('cols', ['author_id'])
        # add A. if there is no prefix already in the column's name
        cols = [col if '.' in col else 'A.{}'.format(col) for col in cols]
        logical_comb = kwargs.get('logical_comb', 'OR')
        case_sensitives = kwargs.get('case_sensitives', [])
        years = kwargs.get('years', [])
        return_papers = kwargs.get('return_papers', False)
        
        constraints_str = ['P.abstract LIKE "%{}%"'.format(k) for k in keywords]
        for k in case_sensitives:
            idx = np.where([x==k for x in keywords])
            idx = [] if len(idx)==0 else idx[0][0]
            constraints_str[idx] = 'P.abstract LIKE BINARY "%{}%"'.format(k)
        constraints_str = ' {} '.format(logical_comb).join(constraints_str)

        if len(years)>0:
            yrs_arr = ','.join([str(x) for x in years])
            constraints_str = '({}) AND YEAR(P.date) IN ({})'.format(
                constraints_str, yrs_arr)

        if return_papers:
            cols += ['P.id']
        pcols = ','.join(cols)

        scomm = 'SELECT {} \
                 FROM author A \
                 INNER JOIN paper_author_mapping P2A ON P2A.author_id=A.id \
                 INNER JOIN paper P ON P.id=P2A.paper_id \
                 WHERE {};'.format(pcols, constraints_str)

        return self.execute_and_get_results(scomm, [col.split('.')[1] for col in cols])    
        
    
    def get_LoP_by_AID(self, author_ids, **kwargs):
        """Getting papers of a set of authors
        """

        years = kwargs.get('years',[])
        cols = kwargs.get('cols', ['id'])

        # taking care of the column headers
        cols = ['P.{}'.format(col) for col in cols] + ['P2A.author_id']
        pcols = ','.join(cols)
        
        # taking care of the conditions in mySQL querying
        if type(author_ids) is int:
            author_ids = [author_ids]
        ID_list_str = '({})'.format(', '.join([str(a) for a in author_ids]))
        constraints_str = 'P2A.author_id IN {}'.format(ID_list_str)

        if len(years)>0:
            years_arr = ','.join([str(x) for x in years])
            constraints_str = '{} AND YEAR(P.date) IN ({})'.format(constraints_str, years_arr)
            
        scomm = 'SELECT {} FROM paper P \
                 INNER JOIN paper_author_mapping P2A ON P2A.paper_id=P.id \
                 WHERE {}'.format(pcols, constraints_str)

        cols = [col.split('.')[1] for col in cols]
        R = self.execute_and_get_results(scomm, cols)
        if len(R)==0:
            return []
        
        A = np.array(R['id'])
        cols.remove('id')
        out = {}
        for a in np.unique(A):
            out[a] = {}
            for i,col in enumerate(cols):
                out[a][col] = np.array(R[col])[A==a]
        return out

        
    def get_NoP_by_AID(self, author_ids, **kwargs):
        """Getting Number of Papers (NoP) by author IDs

        This can be basically done by counting the papers retuned by the previous method.
        However, we are adding a separate method for this purpose because it
        can be done using GROUP BY command of mySQL
        """

        years = kwargs.get('years',[])
        # taking care of the conditions in mySQL querying
        if type(author_ids) is int:
            constraints_str = 'P2A.author_id={}'.format(author_ids)
        else:
            # a list of author IDs
            ID_list_str = '({})'.format(', '.join([str(a) for a in author_ids]))
            constraints_str = 'P2A.author_id IN {}'.format(ID_list_str)

        if len(years)>0:
            years_arr = ','.join([str(x) for x in years])
            constraints_str = '{} AND YEAR(P.date) IN ({})'.format(constraints_str, years_arr)

        scomm = "SELECT P2A.author_id, COUNT(*) FROM paper P \
                 INNER JOIN paper_author_mapping P2A ON P2A.paper_id=P.id \
                 WHERE {} GROUP BY P2A.author_id".format(constraints_str)

        self.crsr.execute(scomm)
        res = self.crsr.fetchall()

        return dict(res)

    
    def get_LoP_by_ents(self, names, **kwargs):
        """Returning papers whose titles/abstracts have at least one of
        the given chemical formula
        """

        years = kwargs.get('years',[])
        cols = kwargs.get('cols', ['id'])
        cols = [col if '.' in col else 'P.{}'.format(col) for col in cols] + \
            ['E.{}'.format(self.entity_col)]
        pcols = ','.join(cols)
        
        names = ['"{}"'.format(c) for c in names]
        constraints_str = 'C.formula IN ({})'.format(', '.join(names))

        if len(years)>0:
            years_arr = ','.join([str(x) for x in years])
            constraints_str = '{} AND YEAR(P.date) IN ({})'.format(constraints_str, years_arr)

        scomm = 'SELECT {} FROM paper P \
                 INNER JOIN {}_paper_mapping E2P ON P.id=E2P.paper_id \
                 INNER JOIN {} E ON E2P.{}_id=E.id \
                 WHERE {};'.format(pcols, self.entity_tab, self.entity_tab,
                                   self.entity_tab, constraints_str)

        cols = [col.split('.')[1] for col in cols]
        R = self.execute_and_get_results(scomm, cols)
        if len(R)==0:
            return []
        
        A = np.array(R[self.entity_col])
        cols.remove(self.entity_col)
        out = {}
        for a in np.unique(A):
            out[a] = {}
            for i,col in enumerate(cols):
                out[a][col] = np.array(R[col])[A==a]
        return out

    
    def get_papers_by_keywords(self, keywords, **kwargs):
        """Returning papers that have the given list of keywords. 

        The logical combination (`logical_comb`) input specifies if
        the papers should have all the keywords at the same time (AND)
        or just having one of the keywords suffices (OR).

        The list `case_sensitives` also includes the keywords that needs to
        be searched in a case-sensitive fashion. 
        """

        cols = kwargs.get('cols',['id'])
        cols = [col if '.' in col else 'P.{}'.format(col) for col in cols]
        pcols = ','.join(cols)
        years = kwargs.get('years', [])
        case_sensitives = kwargs.get('case_sensitives', [])
        logical_comb = kwargs.get('logical_comb', 'OR')

        constraints_str = ["(P.title LIKE '%{}%' OR P.abstract LIKE '%{}%')".format(k,k)
                           for k in keywords]
        for k in case_sensitives:
            idx = np.where([x==k for x in keywords])
            idx = [] if len(idx)==0 else idx[0][0]
            constraints_str[idx] = "(P.title LIKE BINARY '%{}%' OR P.abstract LIKE BINARY '%{}%')".format(k,k)
        constraints_str = " {} ".format(logical_comb).join(constraints_str)

        if len(years)>0:
            yrs_arr = ','.join([str(x) for x in years])
            constraints_str = "({}) AND YEAR(P.date) IN ({})".format(
                constraints_str, yrs_arr)

        scomm = "SELECT {} FROM paper P \
                 WHERE {};".format(pcols, constraints_str)
        
        return self.execute_and_get_results(scomm, [col.split('.')[1] for col in cols])

    
    def get_LoE_by_PID(self, paper_ids, **kwargs):
        """Returning a List of enities that are present in the 
        titles/abstracts of the given papers
        """

        cols = kwargs.get('cols',['id'])

        # taking care of the column headers
        cols = ['E.{}'.format(col) for col in cols] + ['E2P.paper_id']
        pcols = ','.join(cols)

        # taking care of the constraints
        if type(paper_ids) is int:
            paper_ids = [paper_ids]
        # a list of paper IDs
        ID_list_str = '({})'.format(', '.join([str(a) for a in paper_ids]))
        constraints_str = 'E2P.paper_id IN {}'.format(ID_list_str)
        
        scomm = 'SELECT {} FROM {} E \
                 INNER JOIN {}_paper_mapping E2P ON E2P.{}_id=E.id \
                 WHERE {}'.format(pcols, self.entity_tab, self.entity_tab,
                                  self.entity_tab, constraints_str)

        R = self.execute_and_get_results(scomm, cols)
        if len(R)==0:
            return []
        
        A = np.array(R['E2P.paper_id'])
        out = {}
        for a in np.unique(A):
            out[a] = {}
            for i,col in enumerate(cols[:-1]):
                name = col.split('.')[1]
                out[a][name] = np.array(R[col])[A==a]
        return out


    def get_NoE_by_PID(self, paper_ids):
        """Getting Number of Entities (NoE) of a set of paper IDs

        This can be basically done by counting the chemicals retuned by the previous method.
        However, we are adding a separate method for this purpose because it
        can be done using GROUP BY command of mySQL

        """

        if type(paper_ids) is int:
            paper_ids = [paper_ids]
        # a list of paper IDs
        ID_list_str = '({})'.format(', '.join([str(a) for a in paper_ids]))
        constraints_str = 'E2P.paper_id IN {}'.format(ID_list_str)

        scomm = 'SELECT E2P.paper_id, COUNT(*) FROM {} E \
                 INNER JOIN {}_paper_mapping E2P ON E2P.{}_id=E.id \
                 WHERE {} GROUP BY E2P.paper_id;'.format(self.entity_tab, self.entity_tab,
                                                         self.entity_tab, constraints_str)

        self.crsr.execute(scomm)
        res = self.crsr.fetchall()

        return dict(res)
    

    def get_affiliations_by_author_id(self, author_id, cols):
        
        cols, pcols = self.prepare_column_headers(cols, 'AFF')
        scomm = 'SELECT {} \
                 FROM affiliation AFF \
                 INNER JOIN author_affiliation_mapping A2AFF \
                 ON AFF.aff_id=A2AFF.aff_id \
                 WHERE A2AFF.author_id={}'.format(pcols, author_id)
        
        return self.execute_and_get_results(scomm, cols)
    

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
        return {x[0]:np.array(x[1]) for x in zip(cols,R)}


    def extract_titles_abstracts(self,
                                 before_year=None,
                                 em='RAM',
                                 save_path=None,
                                 logger=None):
        """Returning titles and abstracts (merged together) as a list
        of lists (when `em=RAM`) or saving them into lines of a text file
        (when `em=HARD`). If the latter is specified, a path for saving the text
        file (`save_path`) should also be provided.
        """

        # MS text processor
        self.text_processor = utils.MatTextProcessor()

        if before_year:
            scomm = 'SELECT paper_id, title, abstract FROM paper \
                     WHERE YEAR(date)<{};'.format(before_year)
        else:
            scomm = 'SELECT paper_id, title, abstract FROM paper;'
        (_,pids), (_,titles), (_,abstracts) = self.execute_and_get_results(scomm, ['paper_id', 'title','abstract']).items()

        if em=='HARD':
            # processing and saving
            assert save_path is not None, 'Specify a saving path.'

            with open(save_path, 'a') as f:
                for i,pid in enumerate(pids):
                    A = titles[i] + '. ' + abstracts[i]

                    A = A.replace('Inf', 'inf')
                    A = A.replace('All rights reserved.', '')

                    # MORE RULES
                    # removing [DOI: ...]
                    rule_1 = '\[DOI:(.*)\]'
                    rule_2 = '©(.*)\.'

                    se =  re.search(rule_1, A)
                    if se is not None:
                        removal = se.group(0)
                        if len(removal.split(' '))<20:
                            logger.info('{}: {} (to be removed)'.format(
                                pid, se.group(0)))
                            A = re.sub(rule_1, lambda x: '', A)
                    se = re.search(rule_2, A)
                    if se is not None:
                        removal = se.group(0)
                        if len(removal.split(' '))<20:
                            logger.info('{}: {} (to be removed)'.format(
                                pid, se.group(0)))
                            A = re.sub(rule_2, lambda x: '', A)

                               
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
                                        Y_terms,
                                        yrs,
                                        **kwargs):
        """Collecting authors of papers with new co-occurrences (new discoveries)
        and extracting their previous papers on the topic of the property and/or
        the newly studied molecule
        """

        case_sensitives = kwargs.get('case_sensitives', [])
        logfile_path = kwargs.get('logfile_path', None)
        savefile_path = kwargs.get('savefile_path', None)
        start_yr = kwargs.get('start_yr', 2001)
        yr_Y_authors = kwargs.get('yr_Y_authors', None)
        yr_Y_papers = kwargs.get('yr_Y_papers', None)
        
        logger = set_up_logger(__name__,logfile_path,False)

        if (yr_Y_authors is None) or (yr_Y_papers is None):
            yr_Y_authors, yr_Y_papers = self.get_yearwise_authors_by_keywords(
                Y_terms, return_papers=True, case_sensitives=case_sensitives)

        # analyze years from 2001 to 2018 (note that: yrs[-1]=2019)
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
