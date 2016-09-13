#load_csv_to_sql.py

import tarfile
import os
import csv
import sqlite3

import log

TARGET_TYPE = ".csv"

ARCHIVE_FILE = "../data/raw/2016-Sep-09_csv.tar.gz"
CSV_ENCODING = "latin1"
INDEX_KEY = "uuid"

EXTRACT_DIR = ARCHIVE_FILE.split(".tar")[0]
DATABASE_FILE = EXTRACT_DIR.split("_")[0] + "_sqlite.db"

log = log.setup_logging('load_csv_to_sql')

def extract_archive(achive_file, extract_dir):
    log.info("%s | Started extraction process", achive_file)
    with tarfile.open(achive_file) as archive:
        for name in archive.getnames():
            if name.endswith(TARGET_TYPE):
                if os.path.exists(os.path.join(extract_dir, name)):
                    log.debug("%s | Already extracted", name)
                else:
                    log.debug("%s | Extraction started", name)
                    try:
                        archive.extract(name, path=extract_dir)
                        log.debug("%s | Extraction successful", name)
                    except:
                        log.error("%s | Extraction failed", name, exc_info=1)
    log.info("%s | Completed extraction process", achive_file)

def load_file(csv_file, csv_encoding, database_file, index_key):
    with open(csv_file, encoding=csv_encoding) as f:
        con = sqlite3.connect(database_file)
        con.text_factory = str
        c = con.cursor()
        reader = csv.reader(f)
        header = True
        for row in reader:
            if header:
                header = False
                table_name = csv_file.split("/")[-1].split(".")[0]
                sql = "DROP TABLE IF EXISTS %s" % table_name
                c.execute(sql)
                sql = "CREATE TABLE %s (%s)" % (table_name,
                          ", ".join([ "%s text" % column for column in row ]))
                c.execute(sql)
                for column in row:
                    if index_key in column.lower():
                        index = "%s__%s" % ( table_name, column )
                        sql = "CREATE INDEX %s on %s (%s)" % (index,table_name,column)
                        c.execute(sql)
                insertsql = "INSERT INTO %s VALUES (%s)" % (table_name,
                            ", ".join([ "?" for column in row ]))
                rowlen = len(row)
            else:
                if len(row) == rowlen:
                    c.execute(insertsql, row)
        con.commit()

def load_files(extract_dir, database_file, csv_encoding, index_key):
    log.info("%s | Started import process", database_file)
    for file in os.listdir(extract_dir):
        csv_file = extract_dir + "/" + file
        try:
            log.debug("%s | Import started", file)
            load_file(csv_file, csv_encoding, database_file, index_key)
            log.debug("%s | Import successful", file)
        except:
            log.error("%s | Import failed", file,exc_info=1)
    log.info("%s | Completed import process", database_file)

extract_archive(ARCHIVE_FILE, EXTRACT_DIR)
load_files(EXTRACT_DIR, DATABASE_FILE, CSV_ENCODING, INDEX_KEY)
