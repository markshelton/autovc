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

def load():
    db.clear_files(parse_dir, database_file)
    start_time = time.time()
    json_files = db.get_files(extract_dir)
    total_records = len(json_files)
    for i, file in enumerate(json_files):
        json_content = json.loads(open(file, encoding="utf8").read())
        reference = dc.load_yaml(reference_file)
        dc.store_response(json_content, reference, database_file, parse_dir)
        dc.track_time(start_time, i, total_records)
        if i % 500 == 0: dc.load_responses(parse_dir, database_file)

def export():
    db.clear_files(export_dir)
    db.export_files(database_file, export_dir)

def main():
    #cm = db.load_config(config_dir)
    extract()       #Done
    load()          #Pending
    export()        #Pending

if __name__ == "__main__":
    main()
