#!/Anaconda3/env/honours python

"""__main__.py"""

#standard modules
import logging
import os
import json
import time

#third-party modules
import concurrent.futures as cf
from functools import wraps

#local modules
import dbLoader as db
import collection.dataCollector as dc
from logManager import logged

build = 0
base = "collection/crawl/"
base_build = "{0}build/{1}/".format(base, build)

#constants
nl_database_file = "{0}nodes/nodes.db".format(base_build)
nl_archive_file = "{0}nodes/nodes.tar.gz".format(base_build)
nl_extract_dir = "{0}nodes/extract/".format(base_build)
requests_dir = "{0}requests/".format(base_build)
database_file ="{0}crawl.db".format(base_build)
parse_dir = "{0}crawl/parse/".format(base_build)
temp_file = "{0}crawl/temp.txt".format(base_build)
record_file = "{0}crawl/record.txt".format(base_build)
reference_file = "{0}config/_reference.yaml".format(base)

#logger
log = logging.getLogger(__name__)

def extract_filter(src, dst):
    base = os.path.basename(src)
    if base.endswith(".csv"):
        log.info("%s | Extraction", base)
        return (nl_extract_dir+base)
    return None

@logged
def get_incomplete(all_requests, record_file):
    try:
        with open(record_file, "r") as record:
            done_requests = set(file.strip() for file in record.readlines())
    except FileNotFoundError: done_requests = set()
    flat_requests = []
    for request in all_requests.values():
        flat_requests.extend(request)
    incomplete_requests = set(flat_requests) - done_requests
    return list(incomplete_requests)

@logged
def mark_done(file, temp_file):
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    with open(temp_file, "a") as temp:
        temp.write(file+"\n")

@logged
def save_record(temp_file, record_file):
    with open(temp_file, "r") as temp:
        os.makedirs(os.path.dirname(record_file), exist_ok=True)
        with open(record_file, "a") as record:
            for line in temp:
                record.write(line)

@logged
def parse_file(start_time, i, request, json_content, total_records):
    try:
        reference = dc.load_yaml(reference_file)
        dc.store_response(json_content, reference, database_file, parse_dir)
    except: log.error("{0} Parser Failed".format(i),exc_info=True)
    else: dc.track_time(start_time, i, total_records)
    finally: mark_done(request, temp_file)

@logged
def save_progress():
    try: dc.load_responses(parse_dir, database_file)
    except:
        db.clear_files(temp_file)
        log.error("Save Progress Failed", exc_info=True)
        return False
    else:
        save_record(temp_file, record_file)
        db.clear_files(temp_file)
        log.info("Save Progress Successful")
        return True

@logged
def parse(start_time):
    all_requests = dc.retrieve_requests(requests_dir)
    requests = get_incomplete(all_requests, record_file)
    futures, ex = dc.make_requests(requests, database_file)
    start_time = time.time()
    total_records = len(futures)
    for i, future in enumerate(cf.as_completed(futures)):
        response = future.result()
        json_content = response.json()
        request = response.request.url
        parse_file(start_time, i, request, json_content, total_records)
        if i % 500 == 0:
            if not save_progress():
                ex.shutdown()
                return False
    ex.shutdown()
    return True

@logged
def extract():
    db.clear_files(nl_archive_file, nl_extract_dir, nl_database_file)
    dc.download_nodelist(nl_archive_file)
    db.extract_archive(nl_archive_file, nl_extract_dir, extract_filter)
    db.load_files(nl_extract_dir, nl_database_file)

@logged
def prepare():
    db.clear_files(database_file, requests_dir)
    records = dc.select_records(nl_database_file, database_file)
    requests = dc.prepare_requests(records)
    dc.store_requests(requests, requests_dir)

@logged
def load():
    db.clear_files(parse_dir, temp_file)
    start_time = time.time()
    while(True):
        if parse(start_time): break

@logged
def export():
    db.clear_files(export_dir)
    db.export_files(database_file, export_dir)

def main():
    #cm = db.load_config(config_dir)
    #extract()       #Done
    #prepare()       #Done
    load()          #Pending
    #export()        #Pending

if __name__ == "__main__":
    main()