#!/Anaconda3/env/honours python

"""dbLoader"""

#standard modules
import os
import tarfile
import logging
import shutil
import sqlite3

#third-party modules
import odo
import datashape as ds
import sqlalchemy.exc
import pandas as pd

#local modules
import logManager
from configManager import configManager

#constants
TARGET_TYPE = ".csv"

#logger
log = logging.getLogger(__name__)

#functions
def load_config(config_dir=None):
    return configManager(config_dir)

def clear_file(path):
    if os.path.isfile(path):
        os.remove(path)
        log.info("%s | File deleted", os.path.basename(path))
    elif os.path.isdir(path):
        shutil.rmtree(path)
        log.info("%s | Directory deleted", path)
    else:
        log.info("%s | Already deleted", path)

def clear_files(*paths):
    log.info("Started clear process")
    for path in paths:
        clear_file(path)
    log.info("Completed clear process")

def get_files(directory, endswith=None, full=True):
    files = os.listdir(directory)
    files = [file for file in files if os.path.splitext(file)[1] == endswith]
    if full is True: files=["{0}{1}".format(directory,file) for file in files]
    return files

def extract_file(archive, archive_file, extract_dir):
    if os.path.exists(os.path.join(extract_dir, archive_file)):
        log.info("%s | Already extracted", archive_file)
    else:
        try:
            log.info("%s | Extraction started", archive_file)
            archive.extract(archive_file, path=extract_dir)
            log.info("%s | Extraction successful", archive_file)
        except:
            log.error("%s | Extraction failed", archive_file,exc_info=True)

def extract_archive(achive_dir, extract_dir):
    log.info("%s | Started extraction process", achive_dir)
    with tarfile.open(achive_dir) as archive:
        for archive_file in archive.getnames():
            if archive_file.endswith(TARGET_TYPE):
                extract_file(archive, archive_file, extract_dir)
    log.info("%s | Completed extraction process", achive_dir)

def fix_datashape(dshape):
    dshape = ''.join(str(dshape).split("*")[1].split()).replace(" ", "").replace(":", "\":\"").replace(",","\",\"").replace("{", "{\"").replace("}", "\"}")
    dictshape = eval(dshape)
    dkeys = [x.split(":")[0].replace("\"","") for x in dshape.replace("{","").replace("}","").split(",")]
    dictList = []
    for key in dkeys:
        value = dictshape[key]
        value = value.strip().replace("?","")
        if value == "bool": value = "int32"
        value = ds.Option(value)
        dictList.append([key, value])
    dshape = ds.var * ds.Record(dictList)
    return dshape

def get_datashape(odo_resource):
    dshape = odo.discover(odo_resource, engine="python", has_header=True, encoding="utf-8", errors="ignore")
    dshape = fix_datashape(dshape)
    return dshape

def load_file(abs_source_file, database_file, drop=False):
    source_file = os.path.basename(abs_source_file)
    source_name = source_file.split(".")[0]
    log.info("%s | Import started", source_file)
    db_uri = ("sqlite:///"+ database_file +"::"+ source_name)
    if drop: odo.drop(db_uri)
    odo_resource = odo.resource(abs_source_file, engine="python", has_header=True,  encoding="utf-8", errors="ignore")
    try: dshape = get_datashape(odo_resource)
    except ValueError:
        log.error("%s | Datashape failed",source_file,exc_info=1)
    else:
        log.debug("%s | Datashape successful", source_file)
        try: odo.odo(odo_resource, db_uri, dshape=dshape)
        except sqlalchemy.exc.DatabaseError:
            log.error("%s | Import failed", source_file, exc_info=1)
            log.debug("%s | Datashape: %s", source_file, dshape)
        else: log.info("%s | Import successful", source_file)

def load_files(extract_dir, database_file):
    log.info("%s | Started import process", database_file)
    for base_name in os.listdir(extract_dir):
        source_file = extract_dir + base_name
        load_file(source_file, database_file)
    log.info("%s | Completed import process", database_file)

def load_database(archive_file, extract_dir, database_file):
    clear_files(database_file, extract_dir)
    extract_archive(archive_file, extract_dir)
    load_files(extract_dir, database_file)

def main():
    cm = load_config()
    clear_files(cm.database_file, cm.extract_dir)
    extract_archive(cm.archive_file, cm.extract_dir)
    load_files(cm.extract_dir, cm.database_file)

if __name__ == "__main__":
    main()
