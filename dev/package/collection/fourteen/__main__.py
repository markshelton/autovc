#!/Anaconda3/env/honours python

"""__main__.py"""

#standard modules
import logging
import os
import json
import time

#third-party modules

#local modules
import dbLoader as db
import collection.dataCollector as dc

#constants
input_file = "collection/fourteen/input/2014-May-xx_json.zip"
extract_dir = "collection/fourteen/output/extract/"
parse_dir = "collection/fourteen/output/parse/"
export_dir = "collection/fourteen/output/export/"
temp_file = "collection/fourteen/output/temp.txt"
record_file = "collection/fourteen/output/record.txt"
database_file = "collection/fourteen/output/2014-May.db"
config_dir = "collection/fourteen/config/"
reference_file = "collection/fourteen/config/_reference.yaml"

#logger
log = logging.getLogger(__name__)

def extract_filter(src, dst):
    base = os.path.basename(src)
    if base.endswith(".zip"):
        log.info("%s | Extraction", base)
        return (extract_dir+base)
    elif base.endswith(".json"):
        return (extract_dir+base)
    return None

def extract():
    db.clear_files(extract_dir)
    db.extract_archive(input_file, extract_dir, extract_filter)
    for file in db.get_files(extract_dir):
        db.extract_archive(file, extract_dir, extract_filter)
        db.clear_files(file)

def get_incomplete(extract_dir):
    all_files = set(db.get_files(extract_dir))
    try:
        with open(record_file, "r") as record:
            done_files = set(file.strip() for file in record.readlines())
    except FileNotFoundError: done_files = set()
    json_files = all_files - done_files
    return list(json_files)

def mark_done(file, temp_file):
    with open(temp_file, "a") as temp:
        temp.write(file+"\n")

def save_record(temp_file, record_file):
    with open(temp_file, "r") as temp:
        with open(record_file, "a") as record:
            for line in temp:
                record.write(line)

def parse_file(start_time, i, file, total_records):
    try:
        json_content = json.loads(open(file, encoding="utf8").read())
        reference = dc.load_yaml(reference_file)
        dc.store_response(json_content, reference, database_file, parse_dir)
    except: log.error("{0} | File Parser Failed".format(file),exc_info=True)
    else: dc.track_time(start_time, i, total_records)
    finally: mark_done(file, temp_file)

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

def parse(start_time):
    json_files = get_incomplete(extract_dir)
    total_records = len(json_files)
    for i, file in enumerate(json_files):
        parse_file(start_time, i, file, total_records)
        if i % 500 == 0:
            if not save_progress(): return False
    return True

def load():
    #db.clear_files(parse_dir, database_file, record_file, temp_file)
    db.clear_files(parse_dir, temp_file)
    start_time = time.time()
    while(True):
        if parse(start_time): break

def export():
    db.clear_files(export_dir)
    db.export_files(database_file, export_dir)

def main():
    #cm = db.load_config(config_dir)
    #extract()       #Done
    load()          #Pending
    export()        #Pending

if __name__ == "__main__":
    main()
