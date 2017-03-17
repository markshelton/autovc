#!/Anaconda3/env/honours python

"""__main__.py"""

#standard modules
import logging
import os
import sqlite3

#third-party modules

#local modules
import dbLoader as db
from logManager import logged

#constants
input_file = "collection/sixteen/input/2016-Sep-09_csv.tar.gz"
extract_dir = "collection/sixteen/output/extract/"
export_dir = "collection/sixteen/output/export/"
dict_dir = "collection/sixteen/output/export/dictionary/"
dict_file = "collection/sixteen/output/dict.csv"
database_file = "collection/sixteen/output/2016-Sep.db"
config_dir = "collection/sixteen/config/"
flat_file = "collection/sixteen/output/flatten/flat.csv"
flatten_config = "collection/sixteen/config/flatten.sql"

#logger
log = logging.getLogger(__name__)

def extract_filter(src, dst):
    base = os.path.basename(src)
    if base.endswith(".csv"):
        log.info("%s | Extraction", base)
        return (extract_dir+base)
    return None

def extract():
    db.clear_files(extract_dir)
    db.extract_archive(input_file, extract_dir, extract_filter)

def load():
    db.clear_files(database_file)
    db.load_files(extract_dir, database_file)

def export():
    db.clear_files(export_dir)
    db.export_files(database_file, export_dir)

@logged
def flatten_database(database_file, flat_file, flatten_config, flatten_table="flat"):
    with sqlite3.connect(database_file) as conn:
        with open(flatten_config) as file:
            script = file.read()
        conn.executescript(script)
    db.export_file(database_file, os.path.dirname(flat_file)+"/", flatten_table)

def flatten():
    db.clear_files(flat_file)
    flatten_database(database_file, flat_file, flatten_config)

def main():
    #cm = db.load_config(config_dir)
    #extract()       #Done
    #load()          #Done
    #export()        #Done
    flatten()

if __name__ == "__main__":
    main()
