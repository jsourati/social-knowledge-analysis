import os
import gc
import sys
import pdb
import gzip
import json
import pymysql
import numpy as np
from lxml import etree
from zipfile import ZipFile

root_dir = '/srv/wos2019/wos_raw/xml/'

class WOSextractor(object):

    def __init__(self, path, sql_config_path=None,):
        self.path = path
        if sql_config_path is None:
            self.db = None
        else:
            with open(sql_config_path, 'r') as sqlc:
                sql_config = json.load(sqlc)
            self.client_config = sql_config['client_config']
            self.establish_connection(self.client_config)
            
            # creating database and table
            self.db_base_name = sql_config['db_name']
            self.table_name = sql_config['table_name']

    def establish_connection(self, client_config):
        self.db = pymysql.connect(**self.client_config)
        self.crsr = self.db.cursor()

    def create_db(self, year=None):
        self.year = year
        db_name = self.db_base_name + '_{}'.format(year)
        
        sql = 'CREATE DATABASE IF NOT EXISTS {};'.format(db_name)
        try:
            self.crsr.execute(sql)    
            sql = 'USE {};'.format(db_name)
            self.crsr.execute(sql)
            print('Database {} is created'.format(db_name))
        except:
            print('Something went wrong..')

    def create_table(self, table_name=None):
        if table_name is None:
            table_name = self.table_name
        sql = "CREATE TABLE IF NOT EXISTS {} (number INT, \
                                              type TEXT, \
                                              date DATE, \
                                              title TEXT, \
                                              abstract TEXT, \
                                              doi TEXT);".format(table_name)
        self.crsr.execute(sql)

        
    def store_WOS_docs_info(self):

        path = os.path.join(self.path, '{}_DSSHPSH.zip'.format(self.year))
        with ZipFile(path, 'r') as f:

            # reading all files in the namelist
            cnt = 0
            self.bad_rows = []
            for name in f.namelist():
                print('Only adding subfile {}'.format(name)) # temp
                if '.xml.gz' not in name:
                    continue

                with gzip.open(f.open(name), 'r') as z:
                    # get the XML root
                    root = etree.XML(z.read())

                    for i in range(len(root)):
                        doc = root[i]
                        (doctype,
                         docdate,
                         doctitle,
                         docabstract,
                         docdoi) = self.extract_info_from_one_doc(doc)

                        sql = """INSERT INTO {} VALUES ({}, "{}", "{}", "{}", "{}", "{}")""".format(
                            self.table_name, cnt, doctype, docdate, doctitle, docabstract, docdoi)

                        try:
                            self.crsr.execute(sql)
                        except:
                            self.bad_rows += [sql]

                        if not(cnt%1000):
                            self.db.commit()
                            #print(cnt, end=',')
                        cnt += 1
                        
                    del root
                    gc.collect()
                    
        self.db.commit()

    def extract_info_from_one_doc(self, doc):
        """Extracting various information from a single
        document
        """
        
        # type of the document
        doctype = doc[1][0][4][0].text

        # DOI of the document (if available)
        docdoi = 'NA'
        available_types = [a.get('type') for
                           a in doc[2][0][0].getchildren()]
        for child in doc[2][0][0].getchildren():
            if child.get('type')=='doi':
                docdoi = child.get('value')

        # date of the document
        docdate = None
        if 'sortdate' in doc[1][0][1].attrib:
            docdate = doc[1][0][1].get('sortdate')

        # title
        doctitle = 'NA'
        titles = doc[1][0][2]
        for t in titles.getchildren():
            if t.get('type')=='item':
                doctitle = t.text
                doctitle = doctitle.replace('"','\\"')
                break

        # abstract
        docabstract = 'NA'
        if np.any(['abstract' in c for c in
                   [a.tag for a in doc[1][1].getchildren()]]):
            docabstract = doc[1][1][-1][0][0][0].text
            docabstract = docabstract.replace('"','\\"')

        return (doctype,
                docdate,
                doctitle,
                docabstract,
                docdoi)

        
    def reenter_bad_rows(self):
        for r in self.bad_rows:
            r = r.replace('\\','')
            self.crsr.execute(r)

        self.db.commit()
