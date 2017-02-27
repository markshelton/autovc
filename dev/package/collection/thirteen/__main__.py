#!/Anaconda3/env/honours python

"""__main__.py"""

#standard modules
import logging
import os
import pymysql
import codecs
import sqlite3

#third-party modules

#local modules
import dbLoader as db
import collection.sqlConverter as sc

#constants
input_file = "collection/thirteen/input/2013-Dec-xx_mysql.tar.gz"
extract_dir = "collection/thirteen/output/extract/"
export_dir = "collection/thirteen/output/export/"
dict_dir = "collection/thirteen/output/export/dictionary/"
dict_file = "collection/thirteen/output/dict.csv"
parse_dir = "collection/thirteen/output/parse/"
database_file = "collection/thirteen/output/2013-Dec.db"
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

def convert_db(mysql_dir,sqlite_dir):
    os.makedirs(sqlite_dir, exist_ok=True)
    for mysql_file in db.get_files(mysql_dir):
        mysql_short = os.path.basename(mysql_file)
        sqlite_file = sqlite_dir+mysql_short
        log.info("{0} | SQL Conversion Started".format(mysql_short))
        try: sc.mysql_to_sqlite(mysql_file,sqlite_file)
        except: log.error("{0} | SQL Conversion Failed".format(mysql_short))
        else: log.info("{0} | SQL Conversion Successful".format(mysql_short))

def create_db(sqlite_dir, database):
    log.info("Started Database Build process")
    with sqlite3.connect(database) as conn:
        for sqlite_file in db.get_files(sqlite_dir):
            with codecs.open(sqlite_file,encoding="latin-1") as file:
                script = file.read()
            sqlite_short = os.path.basename(sqlite_file)
            log.info("{0} | Table Build Started".format(sqlite_short))
            try: conn.executescript(script)
            except:
                conn.execute("ROLLBACK")
                log.error("{0} | Table Build Failed".format(sqlite_short),exc_info=True)
            else: log.info("{0} | Table Build Successful".format(sqlite_short))
    log.info("Completed Database Build process")

def extract():
    db.clear_files(extract_dir)
    db.extract_archive(input_file, extract_dir, extract_filter)

def parse():
    db.clear_files(parse_dir)
    convert_db(extract_dir, parse_dir)

def load():
    db.clear_files(database_file)
    create_db(parse_dir, database_file)

def export():
    db.clear_files(export_dir)
    db.export_files(database_file, export_dir)

def explore():
    db.clear_files(dict_dir, dict_file)
    db.summarise_files(export_dir, dict_dir, dict_file)

def main():
    #cm = db.load_config(config_dir)
    #extract()      #Done
    #parse()          #Done
    #load()         #Done
    #export()        #Done
    #explore()       #Done

if __name__ == "__main__":
    main()
