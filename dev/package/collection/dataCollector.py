#!/Anaconda3/env/honours python

"""dataCollector.py"""

#standard modules
import logging
import sqlite3
import os
import csv
import time
import json
import yaml
import codecs

#third-party modules
from ratelimit import *
import requests as rq
import concurrent.futures as cf
import sqlalchemy.exc

#local modules
import dbLoader as db
import sqlManager as sm
import collection.responseParser as rp
from logManager import logged

#constants

#configmanager
cm = db.load_config()

#logger
log = logging.getLogger(__name__)

def load_yaml(path):
    if os.path.exists(path):
        with open(path, 'rt') as f:
            output = yaml.safe_load(f.read())
            return output

def find_keys(temp, database, table):
    with sqlite3.connect(database) as c_database:
        try: cursor = c_database.execute("SELECT * FROM %s" % (table))
        except: columns = []
        else: columns = [desc[0] for desc in cursor.description]
    if len(columns) > len(temp.keys()): keys = columns
    else: keys = temp.keys()
    return keys

def format_url(node_type, node_id):
    url = "%s/%s/%s/%s?user_key=%s" % (cm.base_url, cm.version, node_type, node_id, cm.cb_key)
    return url

def track_time(start_time, current_record, total_records, table="", status=""):
    elapsed_time = time.time() - start_time
    percent_complete = current_record / float(total_records) + 0.00001
    time_remaining = (elapsed_time / percent_complete - elapsed_time)
    log.info("%s | %s | %s | %.2f | %.2f | %.2f | %s" % (table, current_record, total_records, percent_complete * 100, elapsed_time / 60, time_remaining / 60, status))

@rate_limited(float(cm.request_space))
def load_url(session, url):
    return session.get(url, headers=cm.headers)

#core functions

def setup():
    nodelist = load_nodelist()
    database = cm.database_file
    tables = cm.selected_tables
    ex = cf.ThreadPoolExecutor(max_workers=int(cm.max_workers))
    return nodelist, database, tables, ex

def download_nodelist(archive_file):
    request = format_url("node_keys","node_keys.tar.gz")
    response = rq.get(request)
    os.makedirs(os.path.dirname(archive_file), exist_ok=True)
    with open(archive_file, 'wb') as f:
        f.write(response.content)

def store_requests(requests, requests_dir):
    os.makedirs(requests_dir, exist_ok=True)
    for table in requests:
        with codecs.open("{0}{1}.txt".format(requests_dir, table), "w+",encoding="utf-8") as file:
            for request in requests[table]:
                file.write(request+"\n")

def select_records(nodelist, database):
    nl_tables = set(sm.get_tables(nodelist))
    db_tables = set(sm.get_tables(database))
    records = {}
    with sqlite3.connect(database) as c_database:
        for table in nl_tables:
            if table in db_tables:
                query = "ATTACH DATABASE \'{0}\' AS db2;".format(nodelist)
                c_database.execute(query)
                query = "SELECT A.{0} FROM db2.{1} AS A LEFT OUTER JOIN {1} AS B ON REPLACE(A.{0},\'-\',\'\') = B.{0} WHERE datetime(A.updated_at) > datetime(B.updated_at,\'unixepoch\') OR B.updated_at IS NULL;".format(cm.table_pk_lookup[table],table)
                records[table] = c_database.execute(query).fetchall()
            else:
                with sqlite3.connect(nodelist) as c_nodelist:
                    query = "SELECT %s FROM %s" % (cm.table_pk_lookup[table],table)
                    records[table] = c_nodelist.execute(query).fetchall()
    return records

@logged
def retrieve_requests(directory):
    requests = {}
    for file in db.get_files(directory):
        table = os.path.basename(file).split(".")[0]
        with codecs.open(file, encoding="utf-8") as f:
            requests[table] = [line.strip() for line in f]
    return requests

def prepare_requests(records):
    requests = {}
    for table in records:
        node_type = cm.api_table_lookup[table]
        if cm.table_pk_lookup[table] == "uuid":
            requests[table] = [format_url(node_type, record) for record, in records[table]]
        else:
            requests[table] = [format_url(node_type, record.split("/")[2]) for record, in records[table]]
    return requests

def read_store(store):
    store_keys, store_list = [], []
    with open(store, 'r+',newline='',encoding="utf-8") as f:
        reader = csv.DictReader(f,delimiter=",")
        store_keys = reader.fieldnames
        for line in reader:
            store_list.append(line)
    return store_keys, store_list

def update_store(store_list, record, record_keys):
    records = []
    records.extend(record)
    for store_record in store_list:
        new_store_record = {}
        for key in record_keys:
            if key in store_record.keys():
                new_store_record[key] = store_record[key]
            else: new_store_record[key] = None
        records.append(new_store_record)
    return records

def fill_na(new_keys, partial_records):
    if type(partial_records) is dict:
        partial_records = [partial_records]
    full_records = []
    for partial_record in partial_records:
        full_record = {}
        for key in new_keys:
            if key in partial_record.keys():
                full_record[key] = partial_record[key]
            else: full_record[key] = None
        full_records.append(full_record)
    return full_records

def check_type(records):
    return (type(records) is list and type(records[0]) is dict)

def store_records(records, keys, store):
    store_name = os.path.basename(store)
    if type(records) is dict: records = list(records)
    if check_type(records):
        write_header = True
        if os.path.isfile(store):
            write_header = False
        if not os.path.exists(os.path.dirname(store)):
            os.makedirs(os.path.dirname(store))
        with open(store, 'a+',newline='',encoding="utf-8") as f:
            writer = csv.writer(f,delimiter=",")
            if write_header:
                writer.writerow(keys)
            for record in records:
                writer.writerow([record[key] for key in keys])
    else: log.error("{0} | Record format error".format(store_name))

def check_record_structure(record, store):
    store_name = os.path.basename(store)
    record, record_keys = [record], list(record.keys())
    record_keys = sorted(record_keys)
    new_store, new_keys = record, record_keys
    try: store_keys, store_list = read_store(store)
    except FileNotFoundError: pass
    else:
        store_keys = sorted(store_keys)
        if record_keys != store_keys:
            if len(record_keys) <= len(store_keys):
                log.debug("{0} | Irregular record - missing key".format(store_name))
                new_store = fill_na(store_keys, record)
                new_keys = store_keys
            else:
                log.info("{0} | Irregular record - new key".format(store_name))
                new_store = update_store(store_list, record, record_keys)
                new_keys = record_keys
                db.clear_files(store)
    return new_store, new_keys

def store_response(response, reference, database, extract_dir):
    records = rp.parse(reference, response)
    for table in records:
        store = "%s%s.csv" % (extract_dir, table)
        for record_number in records[table]:
            record = records[table][record_number]
            log.debug("Record: {0}".format(json.dumps(record, indent=1)))
            new_store, new_keys = check_record_structure(record, store)
            store_records(new_store, new_keys, store)

@logged
def make_requests(requests, database,max_workers=15):
    session = rq.Session()
    ex = cf.ThreadPoolExecutor(max_workers=max_workers)
    futures = [ex.submit(load_url, session, request) for request in requests]
    return futures, ex

def make_requests_old(ex, urls, database, table):
    start_time = time.time()
    session = rq.Session()
    futures = [ex.submit(load_url, session, url) for url in urls]
    for tally, future in enumerate(cf.as_completed(futures)):
        response = future.result()
        if response.status_code == 200:
            reference = load_yaml(cm.crawler_ref)[table]
            extract_dir = cm.crawl_extract_dir
            store_response(response.json(), reference, database, extract_dir)
            status = True
        else:
            status = False
            log.debug("Request failed: {0}".format(response.request.url))
        track_time(start_time, tally, len(urls), table, status)
        if tally % cm.load_rate == 0:
            load_responses(cm.crawl_extract_dir, database)
            db.export_files(database, cm.crawl_export_dir)

def read_db(database, table):
    db_keys, db_list = [], []
    with sqlite3.connect(database) as c_database:
        try:
            cursor = c_database.execute('SELECT * FROM {0}'.format(table))
            db_list = list(cursor.fetchall())
            db_keys = list([description[0] for description in cursor.description])
        except: pass
    return db_keys, db_list

def update_db(db_dict, new_store, new_keys):
    records = list(new_store)
    for db_record in db_dict:
        new_db_record = {}
        for key in new_keys:
            if key in db_record.keys(): new_db_record[key] = db_record[key]
            else: new_db_record[key] = None
        records.append(new_db_record)
    return records

def check_store_structure(store, database):
    drop = False
    store_name = os.path.basename(store)
    table = os.path.splitext(store_name)[0]
    store_keys, store_list = read_store(store)
    store_keys = sorted(store_keys)
    new_store, new_keys = store_list, store_keys
    try: db_keys, db_list = read_db(database, table)
    except sqlalchemy.exc.DatabaseError:
        log.error("{0} | Read Database failed", store_name,exc_info=1)
    else:
        db_keys = sorted(db_keys)
        if len(db_keys) > 0 and store_keys != db_keys:
            if len(store_keys) <= len(db_keys):
                log.debug("{0} | Irregular store - missing key".format(store_name))
                new_store = fill_na(db_keys, store_list)
                new_keys = db_keys
            else:
                log.info("{0} | Irregular store - new key".format(store_name))
                db_dict = [dict(zip(db_keys,db_entry)) for db_entry in db_list]
                new_store = update_db(db_dict, new_store, store_keys)
                new_keys = store_keys
                drop = True
    return new_store, new_keys, drop

def load_responses(extract_dir, database):
    try: stores = db.get_files(extract_dir, full=True)
    except: pass
    else:
        for store in stores:
            store_name = os.path.basename(store)
            new_store, new_keys, drop = check_store_structure(store, database)
            db.clear_file(store)
            store_records(new_store, new_keys, store)
            try: db.load_file(store, database, drop)
            except sqlalchemy.exc.DatabaseError:
                log.error("{0} | Loading error".format(store_name), exc_info=1)
            finally: db.clear_file(store)

def main():
    db.clear_files(cm.crawl_extract_dir)
    nodelist, database, tables, ex = setup()
    for table in tables:
        records = select_records(nodelist, database, table)
        urls = prepare_urls(records, table)
        make_requests(ex, urls, database, table)
        load_responses(cm.crawl_extract_dir, database)

def clean_exit():
    cm = configManager()
    nodelist, database, tables, ex = setup()
    for table in tables:
        try: load_responses(cm.crawl_extract_dir, database, table)
        except: pass
    ex.shutdown(wait=False)
    log.info("Program completed.")

def loop():
    db.clear_files(cm.crawl_extract_dir, cm.database_file)
    while True:
        try: main()
        except Exception as e:
            log.error(e,exc_info=True)
            clean_exit()

if __name__ == "__main__":
    loop()
    #db.clear_files(cm.crawl_extract_dir, cm.database_file)
    #main()
