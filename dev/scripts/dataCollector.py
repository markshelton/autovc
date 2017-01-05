#!/Anaconda3/env/honours python

"""dataCollector.py"""

#standard modules
import logging
import sqlite3
import os
import csv
import time
import sys
import signal

#third-party modules
from ratelimit import *
import requests as rq
import concurrent.futures as cf

#local modules
import logManager
import dbLoader as db
from configManager import configManager

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

def store_record(record, store):
    write_header = True
    if os.path.isfile(store):
        write_header = False
    elif not os.path.exists(os.path.dirname(store)):
        os.makedirs(os.path.dirname(store))
    with open(store, 'a+',newline='',encoding="utf-8") as f:
        writer = csv.writer(f,delimiter=",")
        if write_header: writer.writerow(record.keys())
        writer.writerow(record.values())

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
    return session.get(url)

#Done
def store_response(response, table):
    if response.status_code == 200:
        store = "%s%s.csv" % (cm.crawl_extract_dir, table)
        record = parse_json(response.json())
        store_record(record, store)
        status = "Successful"
    else: status = "Failed"
    return status

#core functions

#Done
def setup():
    nodelist = load_nodelist()
    database = cm.database_file
    tables = get_tables(nodelist)
    ex = cf.ThreadPoolExecutor()
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
        query = "SELECT %s FROM %s LIMIT 20" % (TABLE_PK_LOOKUP[table],table)
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
def make_requests(ex, urls, table):
    start_time = time.time()
    session = rq.Session()
    futures = [ex.submit(load_url, session, url) for url in urls]
    for tally, future in enumerate(cf.as_completed(futures)):
        response = future.result()
        status = store_response(response, table)
        track_time(start_time, tally, len(urls), table, status)
        if tally == 50: raise ValueError

#Done
def load_responses(database, table):
    store = "%s%s.csv" % (cm.crawl_extract_dir, table)
    db.load_file(store, database)

def main():
    #db.clear_files(cm.crawl_extract_dir, cm.database_file, cm.nl_database_file)
    nodelist, database, tables, ex = setup()
    for table, in tables:
        records = select_records(nodelist, database, table)
        urls = prepare_urls(records, table)
        make_requests(ex, urls, table)
        load_responses(database, table)

def exit_gracefully():
    nodelist, database, tables, ex = setup()
    for table, in tables:
        try: load_responses(database, table)
        except: pass
    ex.shutdown(wait=False)
    sys.exit(0)

if __name__ == "__main__":
    try: main()
    except: exit_gracefully()

#Graveyard
