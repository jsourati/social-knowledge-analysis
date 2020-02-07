import os
import re
import sys
import pdb
import numpy as np
import pandas as pd
from itertools import groupby

# This module uses a text-processor class provided by mat2vec project
# specifically designed for materials science articles
mat2vec_path = '~/scratch-midway2/repos/mat2vec'
path = '/home/jamshid/codes/social-knowledge-analysis'
sys.path.insert(0, path)

from misc import helpers

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

    sql_cursor.execute('SELECT aff_id FROM affiliation;')
    all_aff_PKs = sql_cursor.fetchall()
    if len(all_aff_PKs)==0:
        aff_PK = 0
    else:
        aff_PK = max([a[0] for a in all_aff_PKs]) + 1



    # all previously entered paper DOIs to avoid repetition
    sql_cursor.execute('SELECT doi FROM paper;')
    all_dois = sql_cursor.fetchall()
    all_dois = [a[0] for a in all_dois]
    # ... same for authors
    sql_cursor.execute('SELECT author_scopus_ID FROM author;')
    curr_scopus_id_list = [a[0] for a in sql_cursor.fetchall()]
    sql_cursor.execute('SELECT aff_scopus_ID FROM affiliation;')
    # ... same for affiliations
    curr_aff_scopus_id_list = [a[0] for a in sql_cursor.fetchall()]
    # ... even same for (author, affiliation)'s, since they can be repeatitive
    sql_cursor.execute('SELECT * FROM author_affiliation_mapping;')
    curr_author_aff_pairs = list(sql_cursor.fetchall())
    
    bad_dois = []
    for i,doi in enumerate(dois):
        if doi in all_dois:
            print('{} has been already entered to the database'.format(doi))
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
        if r.title is not None:
            title = r.title.replace('\"','')
            title = title.replace('\\Vub\\', '|Vub|') # ad-hoc for a specific article
        else:
            title = 'NA'
        if r.description is not None:
            abst = r.description.replace('\"','')
            abst = abst.replace('\\Vub\\','|Vub|') # ad-hoc for a specific article
            abst = abst.replace('out.\\', 'out.')  # ad-hoc for a specific article
            # yet another ad-hoc
            if doi=='10.1140/epjb/e2012-30482-6':
                abst = re.sub(r'-duration(.*?), among others',
                              '-duration α, among others',abst)
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
                au_given_name = r.authors[i].given_name.replace('\"','') if \
                    r.authors[i].given_name is not None else r.authors[i].given_name
                au_surname = r.authors[i].surname.replace('\"','') if \
                    r.authors[i].surname is not None else r.authors[i].surname
                
                sql_cursor.execute('INSERT INTO author \
                                    VALUES({}, "{}", "{}", "{}")'.format(
                                        author_PK,
                                        scps_id,
                                        au_given_name,
                                        au_surname)
                )
                sql_cursor.execute('INSERT INTO paper_author_mapping \
                                    VALUES({}, {})'.format(
                                        paper_PK, author_PK))
                
                # update the global authors scopus ID list
                curr_scopus_id_list += [scps_id]
                this_author_PK = author_PK  #this will be used in affiliation table
                author_PK += 1
                
            # adding affiliations
            # ---------------------
            # handling None affiliations
            if r.authors[i].affiliation is not None:
                author_aff_scopus_id_list = np.unique(r.authors[i].affiliation)
            else:
                author_aff_scopus_id_list = []
            for aff_scps_id in author_aff_scopus_id_list:
                if aff_scps_id in curr_aff_scopus_id_list:
                    sql_cursor.execute('SELECT aff_id \
                    FROM affiliation \
                    WHERE aff_scopus_ID = {}'.format(aff_scps_id))
                    this_aff_PK = sql_cursor.fetchall()[0][0]

                    # add the pair only if the author/aff. have not already
                    # been added to the mapping table
                    if (this_author_PK, this_aff_PK) not in curr_author_aff_pairs:
                        sql_cursor.execute('INSERT INTO author_affiliation_mapping \
                                            VALUES({}, {})'.format(this_author_PK,
                                                                   this_aff_PK))
                        curr_author_aff_pairs += [(this_author_PK, this_aff_PK)]
                else:
                    lcn = np.where([x.id==aff_scps_id for x in r.affiliation])[0]
                    if len(lcn)>0:
                        lcn = lcn[0]
                        aff_name = r.affiliation[lcn].name.replace('"','\\"')
                        aff_city = r.affiliation[lcn].city
                        aff_country = r.affiliation[lcn].country
                    else:
                        aff_name = 'NA'
                        aff_city = 'NA'
                        aff_country = 'NA'

                    sql_cursor.execute('INSERT INTO affiliation \
                                        VALUES({},"{}","{}","{}","{}");'.format(
                                            aff_PK,
                                            aff_scps_id,
                                            aff_name,
                                            aff_city,
                                            aff_country)
                    )
                    sql_cursor.execute('INSERT INTO author_affiliation_mapping \
                                        VALUES({}, {})'.format(this_author_PK, aff_PK))
                    curr_author_aff_pairs += [(this_author_PK, aff_PK)]
                    # update the affliations list
                    curr_aff_scopus_id_list += [aff_scps_id]
                    aff_PK += 1

        paper_PK += 1

        sql_db.commit()

    return bad_dois

    
def complete_affiliations(paper_ids, sql_db, sql_cursor, logfile_path=None):

    logger = helpers.set_up_logger(__name__, logfile_path, False, file_mode='a')
    
    # initialize the affiliation primary key
    sql_cursor.execute('SELECT aff_id FROM affiliation;')
    all_aff_PKs = sql_cursor.fetchall()
    if len(all_aff_PKs)==0:
        aff_PK = 0
    else:
        aff_PK = max([a[0] for a in all_aff_PKs]) + 1
        
    sql_cursor.execute('SELECT aff_scopus_ID FROM affiliation;')
    curr_aff_scopus_id_list = [a[0] for a in sql_cursor.fetchall()]
    sql_cursor.execute('SELECT * FROM author_affiliation_mapping;')
    curr_author_aff_pairs = list(sql_cursor.fetchall())

    pids_array = ','.join([str(p) for p in paper_ids])
    sql_cursor.execute('SELECT doi, paper_id FROM paper WHERE paper_id IN {};'.format(pids_array))
    RES = sql_cursor.fetchall()
    dois = [a[0] for a in RES]
    paper_ids = [a[1] for a in RES]

    dois_with_nonexisting_authors = []
    for j,doi in enumerate(dois):
        
        try:
            r = AbstractRetrieval(doi)
        except Scopus429Error:
            print('Scopus resource exhausted. Check your quota.')
            return
        except:
            raise ValueError('Could not download doi {}'.format(doi))
        
        if r.authors is None:
            continue
        
        paper_scopus_id_list = [a.auid for a in r.authors]
        for i,scps_id in enumerate(paper_scopus_id_list):
            # if repetitive author, ignore:
            if scps_id in paper_scopus_id_list[:i]:
                continue

            sql_cursor.execute('SELECT author_id \
                                FROM author \
                                WHERE author_scopus_ID = {}'.format(scps_id))
            
            this_author_PK = sql_cursor.fetchall()
            if len(this_author_PK)==0:
                if doi not in dois_with_nonexisting_authors:
                    dois_with_nonexisting_authors += [doi]
                logger.info('(CASE NUMBER {}) PAPER_ID {}, DOI {}: author with scopus ID {} does not exist.'.format(306+len(dois_with_nonexisting_authors), paper_ids[j], doi, scps_id))
                continue
            else:
                this_author_PK = this_author_PK[0][0]
            
            # directly go to their affiliations
            if r.authors[i].affiliation is not None:
                author_aff_scopus_id_list = np.unique(r.authors[i].affiliation)
            else:
                author_aff_scopus_id_list = []
                
            for aff_scps_id in author_aff_scopus_id_list:
                if aff_scps_id in curr_aff_scopus_id_list:
                    sql_cursor.execute('SELECT aff_id \
                    FROM affiliation \
                    WHERE aff_scopus_ID = {}'.format(aff_scps_id))
                    this_aff_PK = sql_cursor.fetchall()[0][0]

                    # add the pair only if the author/aff. have not already
                    # been added to the mapping table
                    if (this_author_PK, this_aff_PK) not in curr_author_aff_pairs:
                        sql_cursor.execute('INSERT INTO author_affiliation_mapping \
                                            VALUES({}, {})'.format(this_author_PK,
                                                                   this_aff_PK))
                        curr_author_aff_pairs += [(this_author_PK, this_aff_PK)]
                        logger.info('{} have been added to A2A.'.format((r.authors[i].given_name,
                                                                         r.authors[i].surname,
                                                                         this_aff_PK)))
                else:
                    lcn = np.where([x.id==aff_scps_id for x in r.affiliation])[0]
                    if len(lcn)>0:
                        lcn = lcn[0]
                        aff_name = r.affiliation[lcn].name.replace('"','\\"')
                        aff_city = r.affiliation[lcn].city
                        aff_country = r.affiliation[lcn].country
                    else:
                        aff_name = 'NA'
                        aff_city = 'NA'
                        aff_country = 'NA'

                    sql_cursor.execute('INSERT INTO affiliation \
                                        VALUES({},"{}","{}","{}","{}");'.format(
                                            aff_PK,
                                            aff_scps_id,
                                            aff_name,
                                            aff_city,
                                            aff_country)
                    )
                    sql_cursor.execute('INSERT INTO author_affiliation_mapping \
                                        VALUES({}, {})'.format(this_author_PK, aff_PK))
                    curr_author_aff_pairs += [(this_author_PK, aff_PK)]
                    logger.info('{} have been added to A2A.'.format((r.authors[i].given_name,
                                                                     r.authors[i].surname,
                                                                     this_aff_PK)))

                    # update the affliations list
                    curr_aff_scopus_id_list += [aff_scps_id]
                    aff_PK += 1

        if not(j%1000):
            np.savetxt('/home/jamshid/codes/data/iter_inds.txt', [j])
        sql_db.commit()




def correct_mats_from_WOS(msdb,wos_D,wos_T,wos_A,yr_susp_dois):

    pr = MatTextProcessor()
    
    with open(yr_susp_dois,'r') as f:
        susp_dois = f.read().splitlines()

    missing_dois = []
    for i in range(1,len(susp_dois)):
        doi = susp_dois[i]
        idx = np.where(wos_D==doi)
        if len(idx)==0:
            missing_dois += [doi]
            continue
        else:
            idx = idx[0][0]

        true_tt = wos_T[idx]
        true_ab = wos_A[idx]

        msdb.crsr.execute("SELECT title,abstract FROM paper WHERE doi='{}'".format(doi))
        false_tt,false_ab = msdb.crsr.fetchone()

        # replce mistaken chemicals in false_tt and false_ab by
        # chemicals in true_tt and true_ab, respectively
        false_tt_ccs = [_[0] for _ in pr.process(false_tt)[1]]
        false_ab_ccs = [_[0] for _ in pr.process(false_ab)[1]]
        true_tt_ccs  = [_[0] for _ in pr.process(true_tt)[1]]
        true_ab_ccs  = [_[0] for _ in pr.process(true_ab)[1]]

        # title
        false_tt_tokens = sum(pr.tokenize(false_tt),[])
        for cc in np.unique(true_tt_ccs):
            if not(pr.is_simple_formula(cc)): continue

            # find the distorted forms of true chemical pattern in the tokens
            distorted_forms = []
            found_partial_matches = []
            for tok in false_tt_tokens:
                if not(pr.is_simple_formula(tok)): continue
                if tok in cc:
                    found_partial_matches += [tok]
                else:
                    # if the current token is not part of the pattern, but partial
                    # found matches is not empty, reset it to empty set
                    # consider scenario:
                    # pattern: 'Zn3P2'
                    # exp.:    'there are 3 properties for Zn and 2 for P'
                    if len(found_partial_matches)>0:
                        found_partial_matches = []

                # if the partial matches become complete, AND
                # the matches include broken parts, save the tokens
                # as one of the detected distorted forms
                if ''.join(found_partial_matches)==cc:
                    if len(found_partial_matches)>1:
                        distorted_forms += [' '.join(found_partial_matches)]

            # replace the distorted forms, if any
            for x in distorted_forms:
                false_tt_tokens.replace(x,cc)
                        
            
        # abstract
        false_ab_tokens = sum(pr.tokenize(false_ab),[])
        for cc in np.unique(true_ab_ccs):
            if not(pr.is_simple_formula(cc)): continue

            # find the distorted forms of true chemical pattern in the tokens
            distorted_forms = []
            found_partial_matches = []
            for tok in false_ab_tokens:
                if not(pr.is_simple_formula(tok)) and (tok not in cc): continue
                if tok in cc:
                    found_partial_matches += [tok]
                else:
                    # if the current token is not part of the pattern, but partial
                    # found matches is not empty, reset it to empty set
                    # consider scenario:
                    # pattern: 'Zn3P2'
                    # exp.:    'there are 3 properties for Zn and 2 for P'
                    if len(found_partial_matches)>0:
                        found_partial_matches = []

                # if the partial matches become complete, AND
                # the matches include broken parts, save the tokens
                # as one of the detected distorted forms
                if ''.join(found_partial_matches)==cc:
                    if len(found_partial_matches)>1:
                        distorted_forms += [' '.join(found_partial_matches)]
                    # resetting partial m
                    found_partial_matches = []

            pdb.set_trace()
            # replace the distorted forms, if any
            for x in np.unique(distorted_forms):
                false_ab.replace(x,cc)

        
    return missing_dois

            
        
