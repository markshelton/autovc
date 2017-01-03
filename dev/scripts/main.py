#!/Anaconda3/env/honours python

"""main"""

#standard modules
import logging
import sqlite3
import json
import os
import csv
import time
import atexit

#third-party modules
from simple_requests import *

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

#requests
#session = FuturesSession(max_workers=int(cm.max_workers))


#helper functions

def format_request(node_type, node_id):
    request = "%s/%s/%s/%s?user_key=%s" % (cm.base_url, cm.version, node_type, node_id, cm.cb_key)
    return request

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
    request = format_request("node_keys","node_keys.tar.gz")
    session = Requests()
    print(request)
    response = session.one(request)
    with open(cm.nl_archive_file, 'wb') as f:
        f.write(response.content)
    return cm.nl_archive_file

def track_time(start_time, current_record, total_records, table):
    elapsed_time = time.time() - start_time
    percent_complete = current_record / float(total_records) + 0.001
    time_remaining = (elapsed_time / percent_complete - elapsed_time)
    log.info("%s | %s | %s | %.2f | %.2f | %.2f" % (table, current_record, total_records, percent_complete * 100, elapsed_time / 60, time_remaining / 60))

def store_response(response):
    record = parse_json(response.json())
    table = record["api_path"].split("/")[0]
    store = "%s%s.csv" % (cm.crawl_extract_dir, table)
    store_record(record, store)

class preprocessor(ResponsePreprocessor):
    def success(self, bundle):
        store_response(bundle.response)
        return bundle.ret()

    def error(self, bundle):
        #raise type(bundle.exception)(bundle.exception).with_traceback(bundle.traceback)
        return None

#core functions

#Done
def setup():
    nodelist = load_nodelist()
    database = cm.database_file
    tables = get_tables(nodelist)
    return nodelist, database, tables

#Done
def load_nodelist():
    if not(os.path.isfile(cm.nl_database_file)):
        nodelist = get_nodelist()
        db.load_file(nodelist,cm.nl_database_file)
    nodelist = cm.nl_database_file
    return nodelist

#TODO more complicated diff
def select_records(nodelist, database, table):
    connection = sqlite3.connect(nodelist)
    query = "SELECT %s FROM %s" % (TABLE_PK_LOOKUP[table],table)
    records = connection.execute(query).fetchall()
    return records

#Done
def prepare_requests(records, table):
    requests = [format_request(table, record) for record, in records]
    return requests

#Done
def make_requests(requests):
    session = Requests(concurrent=int(cm.max_workers),
                   minSecondsBetweenRequests=60/float(cm.requests_per_min),
                   defaultTimeout=int(cm.default_timeout),
                   responsePreprocessor=preprocessor(),
                   retryStrategy=Backoff())
    session.swarm(requests)
    """
    while True:
        print(len(session._requestQueue.queue))
        if len(session._requestQueue.queue) == 0:
            print("HELLO")
            break
    """

#Done
def load_responses(database, table):
    store = "%s%s.csv" % (cm.crawl_extract_dir, table)
    db.load_file(store, database)

def main():
    nodelist, database, tables = setup()
    for table, in tables:
        records = select_records(nodelist, database, table)
        requests = prepare_requests(records, table)
        make_requests(requests)
        #store_responses(responses, len(records), table)
        #load_responses(database, table)

if __name__ == "__main__":
    main()

#Graveyard

#Done
def store_responses(responses, total, table):
    start_time = time.time()
    store = "%s%s.csv" % (cm.crawl_extract_dir, table)
    for tally, response in enumerate(responses):
        record = parse_json(response.json())
        store_record(record, store)
        track_time(start_time, tally, total, table)
