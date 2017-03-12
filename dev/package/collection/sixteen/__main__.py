#!/Anaconda3/env/honours python

"""__main__.py"""

#standard modules
import logging
import os

#third-party modules

#local modules
import dbLoader as db

#constants
input_file = "collection/sixteen/input/2016-Sep-09_csv.tar.gz"
extract_dir = "collection/sixteen/output/extract/"
export_dir = "collection/sixteen/output/export/"
dict_dir = "collection/sixteen/output/export/dictionary/"
dict_file = "collection/sixteen/output/dict.csv"
database_file = "collection/sixteen/output/2016-Sep.db"
config_dir = "collection/sixteen/config/"

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

def explore():
    db.clear_files(dict_dir, dict_file)
    db.summarise_files(export_dir, dict_dir, dict_file)

def main():
    #cm = db.load_config(config_dir)
    #extract()       #Done
    #load()          #Done
    #export()        #Done
    explore()

if __name__ == "__main__":
    main()
