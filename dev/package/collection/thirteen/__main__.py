#!/Anaconda3/env/honours python

"""__main__.py"""

#standard modules
import logging
import os
import codecs

#third-party modules
import psycopg2
import sqlalchemy
import sqlalchemy.exc
import sqlite3

#local modules
import dbLoader as db
import collection.sqlConverter as sc
from logManager import logged

#constants
input_file = "collection/thirteen/input/2013-Dec-xx_mysql.tar.gz"
extract_dir = "collection/thirteen/output/extract/"
export_dir = "collection/thirteen/output/export/"
dict_file = "collection/thirteen/output/dict.csv"
parse_dir = "collection/thirteen/output/parse/"
database_file = "collection/thirteen/output/2013-Dec.db"
config_dir = "collection/thirteen/config/"
flat_file = "collection/thirteen/output/flatten/flat.csv"
flatten_config = "collection/thirteen/config/flatten.sql"

#logger
log = logging.getLogger(__name__)

def extract_filter(src, dst):
    base = os.path.basename(src)
    if base.endswith(".sql"):
        if not base.startswith("._"):
            log.info("%s | Extraction", base)
            return (extract_dir+base)
    return None

def extract():
    db.clear_files(extract_dir)
    db.extract_archive(input_file, extract_dir, extract_filter)

def parse():
    db.clear_files(parse_dir)
    sc.convert_db(extract_dir, parse_dir)

def load():
    db.clear_files(database_file)
    sm.create_db(parse_dir, database_file)

def export():
    db.clear_files(export_dir)
    db.export_files(database_file, export_dir)

def flatten():
    db.clear_files(flat_file)
    flatten_database(database_file, flat_file, flatten_config)

def explore():
    db.clear_files(dict_file)
    db.summarise_files(flatten_dir, flatten_dir, dict_file)


def main():
    #cm = db.load_config(config_dir)
    #extract()      #Done
    #parse()          #Done
    #load()         #Done
    #export()        #Done
    flatten()
    #explore()       #Done

if __name__ == "__main__":
    main()
