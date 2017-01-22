#!/Anaconda3/env/honours python

"""dataCollector.py"""

#standard modules
import logging
import sqlite3
import os
import csv
import time
import sys

#third-party modules
from ratelimit import *
import requests as rq
import concurrent.futures as cf
import dpath.util as dp

#local modules
import dbLoader as db
import responseParser as rp

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


csv.field_size_limit(2147483647)

#helper functions

"""
New Response:
1. check - Response.keys == Store.keys
    yes - Load Response -> Store [DONE]
    no -
        2a. Read Store --> Dict
        2b. Merge Response & Store --> Combined
        2c. Delete current Store file
        2d. Load Combined -> Store [DONE]

Hits Milestone (e.g. 100 Responses):
3. check - Store.keys == SQL.keys
    yes - Load Store -> SQL [DONE]
    no -
        4a. Read SQL --> Dict
        4b. Merge Store & SQL --> Combined
        4c. Delete current SQL table
        4d. Load Combined -> SQL [DONE]
"""

#record = {uuid: blah, xx: blah}
#store_dict[{uuid: blah, xx: blah}, {uuid:blah, xx: blah}]

def find_keys(temp, database, table):
    with sqlite3.connect(database) as c_database:
        try: cursor = c_database.execute("SELECT * FROM %s" % (table))
        except: columns = []
        else: columns = [desc[0] for desc in cursor.description]
    if len(columns) > len(temp.keys()): keys = columns
    else: keys = temp.keys()
    #print("KEYS:", keys)
    return keys

def read_store(store):
    store_keys, store_dict = [], []
    with open(store, 'r+',newline='',encoding="utf-8") as f:
        reader = csv.DictReader(f,delimiter=",")
        store_keys = reader.fieldnames
        for line in reader:
            store_dict.append(line)
    return store_keys, store_dict

def update_store(store_dict, record):
    records = []
    records.append(record)
    for store_record in store_dict:
        new_store_record = {}
        for key in record.keys():
            if key in store_record.keys():
                new_store_record[key] = store_record[key]
            else: new_store_record[key] = None
        records.append(new_store_record)
    return records

def fill_na(store_keys, record):
    new_record = {}
    for key in store_keys:
        if key in record.keys(): new_record[key] = None
        else: new_record[key] = record[key]
    return new_record

def store_records(records, store):
    if type(records) is dict: records = [records]
    write_header = True
    if os.path.isfile(store):
        write_header = False
    if not os.path.exists(os.path.dirname(store)):
        os.makedirs(os.path.dirname(store))
    with open(store, 'a+',newline='',encoding="utf-8") as f:
        writer = csv.writer(f,delimiter=",")
        if write_header:
            writer.writerow(records[0].keys())
        for record in records:
            writer.writerow(record.values())

def format_url(node_type, node_id):
    url = "%s/%s/%s/%s?user_key=%s" % (cm.base_url, cm.version, node_type, node_id, cm.cb_key)
    return url

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

def store_response(response, database):
    if response.status_code == 200:
        records = rp.parse(cm.crawler_ref, response.json())
        for table in records:
            store = "%s%s.csv" % (cm.crawl_extract_dir, table)
            for record_number in records[table]:
                record = records[table][record_number]
                if os.path.isfile(store):
                    store_keys, store_dict = read_store(store)
                    if list(record.keys()) > store_keys:
                        #print("MORE", table)
                        record = update_store(store_dict, record)
                        db.clear_files(store)
                    elif list(record.keys()) < store_keys:
                        #print("LESS", table)
                        record = fill_na(store_keys, record)
                    #else: print("EQUAL", table)
                store_records(record, store)
        status = "Pass"
    else: status = "Fail"
    return status

#Done
def make_requests(ex, urls, database, table):
    start_time = time.time()
    session = rq.Session()
    futures = [ex.submit(load_url, session, url) for url in urls]
    for tally, future in enumerate(cf.as_completed(futures)):
        response = future.result()
        status = store_response(response, database)
        track_time(start_time, tally, len(urls), table, status)
        #input("yo")
        if tally % 400 == 100: load_responses(database)


def load_responses(database):
    db.load_files(cm.crawl_extract_dir, database)
    db.clear_files(cm.crawl_extract_dir)

def main():
    db.clear_files(cm.crawl_extract_dir)
    nodelist, database, tables, ex = setup()
    for table, in tables:
        records = select_records(nodelist, database, table)
        urls = prepare_urls(records, table)
        make_requests(ex, urls, database, table)
        load_responses(database)

def clean_exit():
    nodelist, database, tables, ex = setup()
    for table, in tables:
        try: load_responses(database, table)
        except: pass
    ex.shutdown(wait=False)
    log.info("Program completed.")

def loop():
    db.clear_files(cm.crawl_extract_dir, cm.database_file)
    while True:
        try: main()
        except Exception as e:
            log.error(e)
            clean_exit()

if __name__ == "__main__":
    #loop()
    db.clear_files(cm.crawl_extract_dir, cm.database_file)
    main()
