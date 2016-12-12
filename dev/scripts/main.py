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

#third-party modules
import concurrent.futures as cf
from requests_futures.sessions import FuturesSession

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

#logger
log = logging.getLogger(__name__)

#requests
session = FuturesSession()
futures = {}

#timing
def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time.time()
        result = f(*args, **kwds)
        elapsed = time.time() - start
        log.info("%s took %0.2fs" % (f.__name__, elapsed))
        return result
    return wrapper

#functions

@timed
def request_api(node_type, node_id, cb_key):
    url = ("https://api.crunchbase.com/v/3/%s/%s?user_key=%s"
           % (node_type, node_id, cb_key))
    future = session.get(url)
    response = future.result()
    return response

def load_node_list(archive_file, extract_dir, database_file, cb_key):
    response = request_api("node_keys","node_keys.tar.gz",cb_key)
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

def load_data(nl_database_file, extract_dir, database_file, cb_key):

    def load_table(table, connection, extract_dir, cb_key):

        def track_time(start_time, current_record, total_records):
            elapsed_time = time.time() - start_time
            percent_complete = current_record / float(total_records)
            time_remaining = (elapsed_time / percent_complete - elapsed_time)
            log.info("%s | %s | %s | %.2f | %.2f | %.2f" % (table, current_record, total_records, percent_complete * 100, elapsed_time / 60, time_remaining / 60))

        def download_record(record, table, temp_file, cb_key):
            response = request_api(API_TABLE_LOOKUP[table],record,cb_key)
            data = parse_json(response.json())
            store_temp(data, temp_file)

        temp_file = "%s%s.csv" % (extract_dir, table)
        query = "SELECT COUNT(*) FROM %s" % (table)
        total_records = connection.execute(query).fetchone()[0]
        query = "SELECT %s FROM %s" % (TABLE_PK_LOOKUP[table],table)
        records = connection.execute(query)
        current_record = 0
        start_time = time.time()
        for record, in records:
            current_record += 1
            download_record(record, table, temp_file, cb_key)
            track_time(start_time, current_record, total_records)
        db.load_file(temp_file, database_file)

    conn_nl = sqlite3.connect(nl_database_file)
    query = "SELECT name FROM sqlite_master WHERE type=\'table\'"
    tables = conn_nl.execute(query)
    for table, in tables:
        load_table(table,conn_nl,extract_dir,cb_key)
    conn_nl.close()

def main():
    cm = db.load_config()
    #load_node_list(cm.nl_archive_file, cm.nl_extract_dir,cm.nl_database_file, cm.cb_key)
    load_data(cm.nl_database_file, cm.crawl_extract_dir, cm.database_file, cm.cb_key)

if __name__ == "__main__":
    main()

