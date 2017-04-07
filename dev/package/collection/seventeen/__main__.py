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
input_file = "collection/seventeen/input/2017-Apr-04_csv.tar.gz"
extract_dir = "collection/seventeen/output/extract/"
export_dir = "collection/seventeen/output/export/"
dict_dir = "collection/seventeen/output/export/dictionary/"
dict_file = "collection/seventeen/output/dict.csv"
database_file = "collection/seventeen/output/2017-Apr.db"
config_dir = "collection/seventeen/config/"
flat_file = "collection/seventeen/output/flatten/flat.csv"
flatten_config = "collection/seventeen/config/flatten.sql"

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
    extract()       #Done
    load()          #Done
    export()        #Done
    #flatten()

if __name__ == "__main__":
    main()
