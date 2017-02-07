#!/Anaconda3/env/honours python

"""__main__.py"""

#standard modules
import logging
import os
import pymysql
import codecs

#third-party modules

#local modules
import dbLoader as db

#constants
input_file = "collection/thirteen/input/2013-Dec-xx_mysql.tar.gz"
extract_dir = "collection/thirteen/output/extract/"
export_dir = "collection/thirteen/output/export/"
temp_dir = "collection/thirteen/output/temp/"
database_file = "collection/thirteen/output/2013-Dec.db"
temp_database = "collection/thirteen/output/2013-Dec-temp.db"
config_dir = "collection/thirteen/config/"

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

def build_temp_db(extract_dir, database_file):
    conn =  pymysql.connect()
    for file_name in db.get_files(extract_dir):
        with  codecs.open(file_name, encoding='latin-1') as file:
            script = file.read()
            conn.executescript(script)

def export_temp_files(temp_database, temp_dir):
    conn = pymysql.connect()
    pass

def parse():
    db.clear_files(temp_database, temp_dir)
    build_temp_db(extract_dir, temp_database)
    export_temp_files(temp_database, temp_dir)

def load():
    db.clear_files(database_file)
    db.load_files(temp_dir, database_file)

def export():
    db.clear_files(export_dir)
    db.export_files(database_file, export_dir)

def main():
    #cm = db.load_config(config_dir)
    #extract()      #Done
    load()          #Pending
    parse()         #Pending
    export()        #Pending

if __name__ == "__main__":
    main()
