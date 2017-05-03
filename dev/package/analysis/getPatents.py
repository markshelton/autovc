
import string
import json
import os
import sqlite3
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from unidecode import unidecode
from cleanco import cleanco
from datetime import date
import distance
import requests

import analysis.dataPreparer as dp
from logManager import logged


input_path = "analysis/input/test.db"
pickle_names_path = "analysis/input/PatentsView/names.pkl"
pickle_patents_path = "analysis/input/PatentsView/patents.pkl"
output_path = "analysis/input/PatentsView/patents.db"


@logged
def get_companies(path):
    with sqlite3.connect(path) as conn:
        query = "SELECT company_name, uuid FROM organizations WHERE primary_role = 'company';"
        companies = pd.read_sql_query(query, conn, index_col="uuid")
    return companies

def standardize_name(raw_name):
    try:
        std_name = unidecode(raw_name)
        std_name = std_name.lower()
        std_name = std_name.lstrip().strip()
        std_name = cleanco(std_name).clean_name()
        std_name.translate({ord(c):None for c in string.punctuation})
    except: std_name = raw_name
    finally: return std_name

@logged
def standardize_names(raw_names):
    return raw_names.apply(standardize_name)

@logged
def store_patents(patents, path):
    with sqlite3.connect(path) as conn:
        patents.to_sql("patents", conn, if_exists="replace",index_label ="assignee_uuid")

def names_are_similar(db_name, patents_name):
    try:
        patents_name = standardize_name(patents_name)
        dist1 = distance.nlevenshtein(db_name, patents_name, method=1)
        dist2 = distance.nlevenshtein(db_name, patents_name.split(" ")[0], method=1)
        dist3 = distance.nlevenshtein(db_name.split(" ")[0], patents_name, method=1)
        response = sum([dist1 < 0.2, dist2 < 0.2, dist3 < 0.2, (dist2 == 0)*2, (dist3==0)*2]) > 1
        if response: print("--Matched:", patents_name)
    except: response = False
    finally: return response

def request_patents(company_name):
    company_name = company_name.split(" ")[0]
    base = "http://www.patentsview.org/api/patents/query?"
    q = '{"_begins":{"assignee_organization":"%s"}}' % (company_name)
    f = '["assignee_organization", \
        "patent_num_combined_citations", \
        "patent_num_cited_by_us_patents", \
        "patent_type", \
        "patent_date"]'
    o = '{"per_page":10000}'
    path = "{}q={}&f={}&o={}".format(base, q, f, o)
    try:
        response = requests.get(path).json()
        if response["total_patent_count"] == 0: response = None
    except: response = None
    finally: return response

def parse_patents(company_name, response):
    try:
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
    except: patents = pd.DataFrame()
    finally: return patents

def get_patents(index, company_name):
    response = request_patents(company_name)
    patents = parse_patents(company_name, response)
    patents["index"] = index
    patents.set_index("index", inplace=True)
    return patents

@logged
def go_patents():
    try: names = pd.read_pickle(pickle_names_path)
    except:
        names = get_companies(input_path)
        names["std_name"] = standardize_names(names["company_name"])
        pd.to_pickle(names, pickle_names_path)
    try: patents = pd.read_pickle(pickle_patents_path)
    except: patents = pd.DataFrame()
    print(names.shape)
    for i, (uuid, std_name) in enumerate(names["std_name"].iteritems()):
        print("Request:", i, std_name)
        if uuid not in patents.index.tolist():
            temp = get_patents(uuid, std_name)
            if temp.empty: names = names.drop(uuid)
            patents = pd.concat([patents, temp],axis=0)
            pd.to_pickle(patents, pickle_patents_path)
        else: print("--Already Matched")
        if i % 1000 == 0:
            pd.to_pickle(names, pickle_names_path)
            store_patents(patents, output_path)

def main():
    go_patents()

if __name__ == "__main__":
    main()