import string
import json
import os
import sqlite3
import pickle
from collections import defaultdict
import logging
import sys

from ratelimit import *
import numpy as np
import pandas as pd
from unidecode import unidecode
from cleanco import cleanco
from datetime import date
import distance
import requests as rq
import concurrent.futures as cf
import sqlalchemy.exc
import urllib.request

import analysis.dataPreparer as dp
from logManager import logged

input_path = "analysis/input/test.db"
pickle_names_path = "analysis/input/PatentsView/names.pkl"
pickle_patents_path = "analysis/input/PatentsView/patents.pkl"
output_path = "analysis/input/PatentsView/patents.db"
headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}
log = logging.getLogger(__name__)

def get_companies(path):
    with sqlite3.connect(path) as conn:
        query = "SELECT company_name, uuid FROM organizations WHERE primary_role = 'company';"
        companies = pd.read_sql_query(query, conn, index_col="uuid")
    return companies

def load_names(pickle_names_path, input_path):
    try: names = pd.read_pickle(pickle_names_path)
    except:
        names = get_companies(input_path)
        names["std_name"] = standardize_names(names["company_name"])
        pd.to_pickle(names, pickle_names_path)
    finally: return names

def load_patents(path):
    try:
        with sqlite3.connect(path) as conn:
            patents = pd.read_sql("SELECT * FROM patents;", conn, index_col="assignee_uuid")
    except: patents = pd.DataFrame()
    finally: return patents.index

@logged
def load_progress(pickle_names_path, input_path, output_path):
    names = load_names(pickle_names_path, input_path)
    patents = load_patents(output_path)
    names = names.drop(patents, errors="ignore")
    new_patents =  pd.DataFrame()
    tested_uuids = []
    return names, patents, new_patents, tested_uuids

def prepare_url(company_name):
    company_name = company_name.split(" ")[0]
    base = "http://www.patentsview.org/api/patents/query?"
    q = '{"_begins":{"assignee_organization":"%s"}}' % (company_name)
    f = '["assignee_organization","patent_num_combined_citations","patent_num_cited_by_us_patents","patent_type","patent_date"]'
    o = '{"per_page":10000}'
    path = "{}q={}&f={}&o={}".format(base, q, f, o)
    return path

@logged
def prepare_urls(names):
    return {uuid: prepare_url(company_name) for (uuid, company_name) in names["std_name"].iteritems()}

def load_url(session, url):
    return session.get(url, headers=headers)

def standardize_name(raw_name):
    std_name = unidecode(raw_name)
    std_name = std_name.lower()
    std_name = std_name.lstrip().strip()
    std_name = cleanco(std_name).clean_name()
    std_name = std_name.translate({ord(c):None for c in string.punctuation})
    return std_name

def standardize_names(raw_names):
    return raw_names.apply(standardize_name)

def names_are_similar(db_name, patents_name):
    if patents_name is None: return False
    patents_name = standardize_name(patents_name)
    dist1 = distance.nlevenshtein(db_name, patents_name, method=1)
    dist2 = distance.nlevenshtein(db_name, patents_name.split(" ")[0], method=1)
    dist3 = distance.nlevenshtein(db_name.split(" ")[0], patents_name, method=1)
    response = sum([dist1 < 0.2, dist2 < 0.2, dist3 < 0.2, (dist2 == 0)*2, (dist3 == 0)*2]) > 1
    if response: print("--Matched:", patents_name)
    return response

def parse_patents(company_name, response):
    temp = []
    similarity = {}
    for patent in response["patents"]:
        response_name = patent["assignees"][0]["assignee_organization"]
        if response_name not in similarity:
            similarity[response_name] = names_are_similar(company_name, response_name)
        if similarity[response_name]:
            del patent["assignees"]
            temp.append(patent)
    patents = pd.DataFrame(temp)
    return patents

def store_patents(patents, path):
    with sqlite3.connect(path) as conn:
        patents.to_sql("patents", conn, if_exists="append",index=False)

@logged
def save_progress(names, new_patents, pickle_names_path, output_path):
    pd.to_pickle(names, pickle_names_path)
    store_patents(new_patents, output_path)

def main():
    names, patents, new_patents, tested_uuids = load_progress(pickle_names_path, input_path, output_path)
    url_map = prepare_urls(names)
    ex = cf.ThreadPoolExecutor(max_workers=10)
    session = rq.Session()
    future_map = {ex.submit(load_url, session, url):uuid for (uuid, url) in url_map.items()}
    for tally, future in enumerate(cf.as_completed(future_map)):
        if tally % 5000 == 50:
            names = names.drop(tested_uuids)
            save_progress(names, new_patents, pickle_names_path, output_path)
            names, patents, new_patents, tested_uuids = load_progress(pickle_names_path, input_path, output_path)
        response = future.result()
        status = response.status_code
        uuid = future_map[future]
        company_name = names.loc[uuid]["std_name"]
        try:
            content = response.json()
            iterator = iter(content["patents"])
        except Exception as e:
            if type(e) == ValueError: msg = "JSON error"
            elif type(e) == TypeError: msg ="No patents"
            else: msg = "Unknown error"
            continue
        else:
            msg = "OK"
            temp = parse_patents(company_name, content)
            if not temp.empty:
                temp["assignee_uuid"] = uuid
                new_patents = pd.concat([new_patents, temp], ignore_index=True)
        finally:
            tested_uuids.append(uuid)
            log.info("{} | {} | {} | {}".format(tally, status, company_name, msg))

def loop():
    while True:
        try: main()
        except Exception as e:
            log.error(e,exc_info=True)
            save_progress(names, new_patents, pickle_names_path, output_path)

if __name__ == "__main__":
    loop()

