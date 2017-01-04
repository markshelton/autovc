#!/Anaconda3/env/honours python

"""dataCollector.py"""

#standard modules
import logging
import sqlite3
import json
import os
import csv
import time
import atexit

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
    for field in json["data"]["properties"].keys():
        record[field] = json["data"]["properties"][field]
    return record

def store_record(record, store):
    write_header = True
    if os.path.isfile(store):
        write_header = False
    elif not os.path.exists(os.path.dirname(store)):
        os.makedirs(os.path.dirname(store))
    with open(store, 'a+',newline='') as f:
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

def track_time(start_time, current_record, total_records, table):
    elapsed_time = time.time() - start_time
    percent_complete = current_record / float(total_records) + 0.001
    time_remaining = (elapsed_time / percent_complete - elapsed_time)
    log.info("%s | %s | %s | %.2f | %.2f | %.2f" % (table, current_record, total_records, percent_complete * 100, elapsed_time / 60, time_remaining / 60))

@rate_limited(float(cm.request_space))
def load_url(session, url):
    return session.get(url)

#Done
def store_response(response, table):
    store = "%s%s.csv" % (cm.crawl_extract_dir, table)
    record = parse_json(response.json())
    store_record(record, store)

#core functions

#Done
def setup():
    db.clear_files(cm.crawl_extract_dir, cm.database_file, cm.nl_database_file)
    nodelist = load_nodelist()
    database = cm.database_file
    tables = get_tables(nodelist)
    return nodelist, database, tables

#Done
def load_nodelist():
    if not(os.path.isfile(cm.nl_database_file)):
        nodelist = get_nodelist()
        db.load_database(nodelist,cm.nl_extract_dir, cm.nl_database_file)
    nodelist = cm.nl_database_file
    return nodelist

#TODO more complicated diff
def select_records(nodelist, database, table):
    connection = sqlite3.connect(nodelist)
    query = "SELECT %s FROM %s LIMIT 10" % (TABLE_PK_LOOKUP[table],table)
    records = connection.execute(query).fetchall()
    return records

#Done
def prepare_urls(records, table):
    urls = [format_url(table, record) for record, in records]
    return urls

#Done
def make_requests(urls, table):
    start_time = time.time()
    session = rq.Session()
    with cf.ThreadPoolExecutor() as ex:
        futures = [ex.submit(load_url, session, url) for url in urls]
        for tally, future in enumerate(cf.as_completed(futures)):
            response = future.result()
            if response.status_code == 200:
                store_response(response, table)
            track_time(start_time, tally, len(urls), table)

#Done
def load_responses(database, table):
    store = "%s%s.csv" % (cm.crawl_extract_dir, table)
    db.load_file(store, database)

def main():
    nodelist, database, tables = setup()
    for table, in tables:
        records = select_records(nodelist, database, table)
        urls = prepare_urls(records, table)
        make_requests(urls, table)
        load_responses(database, table)

if __name__ == "__main__":
    main()

#Graveyard
