#!/Anaconda3/env/honours python

"""dbLoader"""

import csv
import os
import sqlite3
import tarfile

from odo import odo

from log import Log

LOG = Log(__file__).logger

#file operations
ARCHIVE_FILE = "../data/raw/2016-Sep-09_csv.tar.gz"
EXTRACT_DIR = "../data/processed/2016-Sep-09_csv/"
DATABASE_FILE = "../data/2016-Sep-09_sqlite2.db"

#reading files
TARGET_TYPE = ".csv"
CSV_ENCODING = "latin1"

#table operations
INDEX_KEY = "uuid"

#metaprogramming
LOG_LEVEL = "info"

class dbLoader(object):

    def __init__(self, archive_file, extract_dir, database_file, csv_encoding, index_key, log):
        self.archive_file = archive_file
        self.extract_dir = extract_dir
        self.database_file = database_file
        self.csv_encoding = csv_encoding
        self.index_key = index_key
        self.log = log
        self.extract_archive(self.archive_file, self.extract_dir, self.log)
        self.load_files(self.extract_dir, self.database_file, self.csv_encoding, self.index_key, self.log)

    def load_config(self, config_file):
        with open(config_file) as config:
            pass

    def extract_archive(self, achive_file, extract_dir, log):
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
                            log.error("%s | Extraction failed", name)
        log.info("%s | Completed extraction process", achive_file)

    def load_files(self, extract_dir, database_file, csv_encoding, index_key, log):
        log.info("%s | Started import process", database_file)
        for file in os.listdir(extract_dir):
            try:
                log.debug("%s | Import started", file)
                database_uri = ("sqlite:///"+database_file
                            +"::"+file.split(".")[0])
                csv_file = extract_dir + file
                odo(csv_file, database_uri)
                log.debug("%s | Import successful", file)
            except:
                log.error("%s | Import failed", file)
        log.info("%s | Completed import process", database_file)

def main():
    db = dbLoader(ARCHIVE_FILE,EXTRACT_DIR, DATABASE_FILE, CSV_ENCODING, INDEX_KEY, LOG)


if __name__ == '__main__':
    main()
