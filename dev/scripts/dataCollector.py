#!/Anaconda3/env/honours python

"""dataCollector.py"""

#standard modules
import logging
import sqlite3
import os
import csv
import time
import sys
import json

#third-party modules
from ratelimit import *
import requests as rq
import concurrent.futures as cf
import dpath.util as dp

#local modules
import logManager
import dbLoader as db
import configManager

#constants

TABLE_PK_LOOKUP = {
    "acquisitions":"uuid",
    "funding_rounds":"uuid",
    "funds":"uuid",
    "ipos":"uuid",
    "organizations":"permalink",
    "people":"permalink",
    "products":"permalink"
}

API_TABLE_LOOKUP = {
    "acquisitions":"acquisitions",
    "funding_rounds":"funding-rounds",
    "funds":"funds",
    "ipos":"ipos",
    "organizations":"organizations",
    "people":"people",
    "products":"products"
}

#configmanager
cm = db.load_config()

#logger
log = logging.getLogger(__name__)

#helper functions

def format_url(node_type, node_id):
    url = "%s/%s/%s/%s?user_key=%s" % (cm.base_url, cm.version, node_type, node_id, cm.cb_key)
    return url

def parse_json(json):
    record = {}
    record["uuid"] = json["data"]["uuid"]
    fields = json["data"]["properties"].keys()
    for field in fields:
        record[field] = json["data"]["properties"][field]
    return record

def get_table(json):
    table = json["properties"]["api_path"].split("/")[0]
    table_lookup = {v: k for k, v in API_TABLE_LOOKUP.items()}
    table = table_lookup[table]
    return table

def get_value(dictionary, glob):
    try: return dp.get(dictionary, glob)
    except KeyError: return None

def unpack_properties(keychain, reference, response):
    res_keys = get_value(response, keychain).keys() #keys=[created_at]
    table = get_value(reference, keychain).split(".")[0] #funds
    dp.delete(reference, keychain)
    for res_key in res_keys:
        keychain.append(res_key) #kc=[properties, created_at]
        value = "{0}.{1}".format(table, res_key) #funds.created_at
        dp.new(reference, keychain, value)
        keychain.pop()

def store(keychain, reference, response, records):
    #kc = [properties, created_at]
    ref = get_value(reference, keychain).split(".") # [funds, created_at]
    table, attribute = tuple(ref) #table = funds, attribute = created_at
    res = get_value(response, keychain) #1474603967
    if get_value(records, table):
        length = len(get_value(records, table))
    else: length = 1
    ref = [table, length - 1, attribute]
    existing = get_value(records, ref)
    if existing: ref = [table, length, attribute]
    dp.new(records, ref, res)
    return records

def split_key(keychain, reference):
    # kc = [relationships, venture_firm.item.uuid]
    print("BEFORE:", keychain)
    key = ''.join(keychain[-1:]) # key = venture_firm.item.uuid
    value = get_value(reference, keychain) # value = funds.venture_firm
    dp.delete(reference, keychain)
    keychain.pop() # kc=[relationships]
    keychain.extend(key.split(".",maxsplit=1))
    # kc=[relationships, venture_firm, item.uuid]
    print("AFTER:", keychain)
    dp.new(reference, keychain, value)

def parse(keychain, visited, reference, response, records):
    key = ''.join(keychain[-1:]) #kc=[relationships, images.items]
    value = get_value(reference, keychain)
    print(key, keychain, value)
    nb = input("yo")
    if "." in key: #images.items
        print("A")
        split_key(keychain, reference)
        if keychain not in visited:
            parse(keychain, visited, reference, response, records)
            visited.append(list(keychain))
        keychain.pop()
    elif type(value) is list:
        print("B")
        #TODO
    elif type(value) is dict or value is None: #kc=[relationships]
        print("C")
        if type(value) is dict: ref_keys = value.keys()
        elif value is None: ref_keys = reference.keys()
        for ref_key in ref_keys:
            keychain.append(ref_key) #kc=[relationships, investors.items]
            if keychain not in visited:
                parse(keychain, visited, reference, response, records)
                visited.append(list(keychain))
            keychain.pop()
    elif key == "items":
        print("D")
        res_items = get_value(response, keychain)
        print("ITEMS:", res_items)
        for i, res_item in enumerate(res_items):
            parse(keychain, visited, reference, response, records)
    elif key == "properties": #kc=[properties]
        print("E")
        if keychain not in visited:
            unpack_properties(keychain, reference, response)
            parse(keychain, visited, reference, response, records)
            visited.append(list(keychain))
    else:
        print("F")
        records = store(keychain, reference, response, records)
        print("RECORDS:", records)
        nb = input("hi")

def play(reference, response):
    records = {}
    keychain = []
    visited = []
    parse(keychain, visited, reference, response, records)
    exit()
    return records

def parse_json_new(reference, response):
    record = {}
    response = response["data"]
    table = get_table(response)
    reference = configManager.load_yaml(reference)[table]
    record = play(reference, response)
    print(reference)
    return record

    #dump = json.dumps(output, indent=4)

def store_record(record, store, columns):
    write_header = True
    if os.path.isfile(store):
        write_header = False
    elif not os.path.exists(os.path.dirname(store)):
        os.makedirs(os.path.dirname(store))
    with open(store, 'a+',newline='',encoding="utf-8") as f:
        writer = csv.writer(f,delimiter=",")
        if columns:
            if write_header: writer.writerow(columns)
            values = list(record.get(column) for column in columns)
        else:
            if write_header: writer.writerow(record.keys())
            values = record.values()
        writer.writerow(values)

def get_tables(database):
    connnection = sqlite3.connect(database)
    query = "SELECT name FROM sqlite_master WHERE type=\'table\'"
    return connnection.execute(query)

def get_nodelist():
    request = format_url("node_keys","node_keys.tar.gz")
    response = rq.get(request)
    with open(cm.nl_archive_file, 'wb') as f:
        f.write(response.content)
    return cm.nl_archive_file

def track_time(start_time, current_record, total_records, table, status):
    elapsed_time = time.time() - start_time
    percent_complete = current_record / float(total_records) + 0.00001
    time_remaining = (elapsed_time / percent_complete - elapsed_time)
    log.info("%s | %s | %s | %.2f | %.2f | %.2f | %s" % (table, current_record, total_records, percent_complete * 100, elapsed_time / 60, time_remaining / 60, status))

@rate_limited(float(cm.request_space))
def load_url(session, url):
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }
    return session.get(url, headers=headers)

def get_columns(database, table):
    c_database = sqlite3.connect(database)
    cursor = c_database.execute("SELECT * FROM %s" % (table))
    columns = [desc[0] for desc in cursor.description]
    return columns

#Done
def store_response(response, database, table):
    if response.status_code == 200:
        store = "%s%s.csv" % (cm.crawl_extract_dir, table)
        record = parse_json(response.json())
        try: columns = get_columns(database, table)
        except: columns = None
        store_record(record, store, columns)
        status = "Successful"
    else: status = "Failed"
    return status

#core functions

#Done
def setup():
    nodelist = load_nodelist()
    database = cm.database_file
    tables = get_tables(nodelist)
    ex = cf.ThreadPoolExecutor(max_workers=10)
    return nodelist, database, tables, ex

#Done
def load_nodelist():
    if not(os.path.isfile(cm.nl_database_file)):
        if not (os.path.isfile(cm.nl_archive_file)):
            nodelist = get_nodelist()
        db.load_database(cm.nl_archive_file,cm.nl_extract_dir, cm.nl_database_file)
    nodelist = cm.nl_database_file
    return nodelist

#Done
def select_records(nodelist, database, table):
    c_database = sqlite3.connect(database)
    c_nodelist = sqlite3.connect(nodelist)
    query = "SELECT name FROM sqlite_master WHERE type= \'table\' AND name=\'%s\';" % (table)
    table_exists = (len(c_database.execute(query).fetchall()) == 1)
    if table_exists:
        query = "ATTACH DATABASE \'{0}\' AS db2;".format(nodelist)
        c_database.execute(query)
        query = "SELECT A.{0} FROM db2.{1} AS A LEFT OUTER JOIN {1} AS B ON REPLACE(A.{0},\'-\',\'\') = B.{0} WHERE datetime(A.updated_at) > datetime(B.updated_at,\'unixepoch\') OR B.updated_at IS NULL;".format(TABLE_PK_LOOKUP[table],table)
        records = c_database.execute(query).fetchall()
    else:
        query = "SELECT %s FROM %s LIMIT 600" % (TABLE_PK_LOOKUP[table],table)
        records = c_nodelist.execute(query).fetchall()
    return records

#Done
def prepare_urls(records, table):
    node_type = API_TABLE_LOOKUP[table]
    if TABLE_PK_LOOKUP[table] == "uuid":
        urls = [format_url(node_type, record) for record, in records]
    else: urls = [format_url(node_type, record.split("/")[2]) for record, in records]
    return urls

#Done
def make_requests(ex, urls, database, table):
    start_time = time.time()
    session = rq.Session()
    futures = [ex.submit(load_url, session, url) for url in urls]
    for tally, future in enumerate(cf.as_completed(futures)):
        response = future.result()
        status = store_response(response, database, table)
        track_time(start_time, tally, len(urls), table, status)
        if tally % 500 == 50: load_responses(database, table)

#Done
def load_responses(database, table):
    store = "%s%s.csv" % (cm.crawl_extract_dir, table)
    db.load_file(store, database)
    db.clear_files(store)

def main():
    db.clear_files(cm.crawl_extract_dir)
    nodelist, database, tables, ex = setup()
    for table, in tables:
        records = select_records(nodelist, database, table)
        urls = prepare_urls(records, table)
        make_requests(ex, urls, database, table)
        load_responses(database, table)

def clean_exit():
    nodelist, database, tables, ex = setup()
    for table, in tables:
        try: load_responses(database, table)
        except: pass
    ex.shutdown(wait=False)
    log.info("Program completed.")

def loop():
    db.clear_files(cm.crawl_extract_dir, cm.database_file, cm.nl_database_file)
    while True:
        try: main()
        except Exception as e:
            log.error("Error: %s" % e)
            clean_exit()

if __name__ == "__main__":
    #loop()
    ref_path = "../config/crawler.yaml"
    json_path = "../../sources/api_examples/funds.json"
    json = json.loads(open(json_path).read())
    parse_json_new(ref_path, json)

