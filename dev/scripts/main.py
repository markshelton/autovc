#!/Anaconda3/env/honours python

"""main"""

#standard modules
import logging
import sqlite3
import json
import os
import csv
import time
from functools import wraps
import atexit

#third-party modules
import concurrent.futures as cf
from requests_futures.sessions import FuturesSession
from ratelimit import *

#local modules
import logManager as lm
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

#requests
session = FuturesSession(max_workers=int(cm.max_workers))

#functions

def request_api(base_url, version, node_type, node_id, cb_key):
    url = ("%s/%s/%s/%s?user_key=%s"
           % (base_url, version, node_type, node_id, cb_key))
    future = session.get(url)
    return future

def load_node_list(archive_file, extract_dir, database_file, cb_key):
    response = request_api("node_keys","node_keys.tar.gz",cb_key).result()
    with open(archive_file, 'wb') as f:
        f.write(response.content)
    db.load_database(archive_file, extract_dir, database_file)

def parse_json(json):
    record = {}
    record["uuid"] = json["data"]["uuid"]
    for field in json["data"]["properties"].keys():
        record[field] = json["data"]["properties"][field]
    return record

def store_temp(record, temp_file):
    write_header = True
    if os.path.isfile(temp_file):
        write_header = False
    elif not os.path.exists(os.path.dirname(temp_file)):
        os.makedirs(os.path.dirname(temp_file))
    with open(temp_file, 'a+',newline='') as f:
        writer = csv.writer(f,delimiter=",")
        if write_header: writer.writerow(record.keys())
        writer.writerow(record.values())

def track_time(start_time, current_record, total_records):
        elapsed_time = time.time() - start_time
        percent_complete = current_record / float(total_records)
        time_remaining = (elapsed_time / percent_complete - elapsed_time)
        log.info("%s | %s | %s | %.2f | %.2f | %.2f" % (table, current_record, total_records, percent_complete * 100, elapsed_time / 60, time_remaining / 60))

def select_records(table, conn_db, conn_nl):
    query = "SELECT %s FROM %s" % (TABLE_PK_LOOKUP[table],table)
    #TODO more complicated diff
    records = connection.execute(query).fetchall()
    return records

def make_request(table, record, cb_key):
    future = request_api(table,record,cb_key)
    return future

def store_response(response, temp_file):
    data = parse_json(response.json())
    store_temp(data, temp_file)

def load_temp(temp_file, database_file):
    db.load_file(temp_file, database_file)

def load_table(table, conn_db, conn_nl, extract_dir, cb_key):

    records = select_records(connection, table)
    current_record, total_records = 1, len(records)
    start_time = time.time()
    futures = {}
    for record, in records:
        future = make_request(API_TABLE_LOOKUP[table],record,cb_key)
        futures[future] = record
    temp_file = "%s%s.csv" % (extract_dir, table)
    atexit.register(load_temp, temp_file, database_file)
    for future in cf.as_completed(futures.keys(), timeout=cm.api_timeout):
        try:
            response = future.result()
            store_response(response, temp_file)
            track_time(start_time, current_record, total_records)
        except Exception as e:
            log.error("Error %s on %s/%s" % (e, table, futures[future]))
        finally:
            current_record += 1
    load_temp(temp_file, database_file)

def get_tables(database_file):
    connnection = sqlite3.connect(database_file)
    query = "SELECT name FROM sqlite_master WHERE type=\'table\'"
    return connnection.execute(query)

def load_data(nl_database_file, extract_dir, database_file, cb_key):
    for table, in get_tables(nl_database_file):
        load_table(table,conn_db,conn_nl,extract_dir,cb_key)
    conn_nl.close()

def main():
    #cm = db.load_config()
    #load_node_list(cm.nl_archive_file, cm.nl_extract_dir,cm.nl_database_file, cm.cb_key)
    load_data(cm.nl_database_file, cm.crawl_extract_dir, cm.database_file, cm.cb_key)

if __name__ == "__main__":
    main()

