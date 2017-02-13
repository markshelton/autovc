#!/Anaconda3/env/honours python

"""dbLoader"""

#standard modules
import os
import logging
import shutil
import csv

#third-party modules
import odo
import datashape as ds
import sqlalchemy.exc
import setuptools.archive_util

#local modules
import logManager
from configManager import configManager
import sqlManager as sm

#constants

#configManager
def load_config(config_dir=None):
    return configManager(config_dir)

#logger
log = logging.getLogger(__name__)

#functions

def export_file(database_file, export_dir, table, drop=False):
    table_name = "{0}.csv".format(table)
    table_uri = "sqlite:///{0}::{1}".format(database_file, table)
    export_file = "{0}{1}".format(export_dir, table_name)
    log.info("{0} | Export started".format(table_name))
    os.makedirs(os.path.dirname(export_dir), exist_ok=True)
    if drop: odo.drop(table_uri)
    try: odo.odo(table_uri, export_file)
    except sqlalchemy.exc.DatabaseError:
        log.error("{0} | Export failed".format(table_name), exc_info=1)
    else: log.info("{0} | Export successful".format(table_name))

def export_files(database_file, export_dir):
    log.info("{0} Started export process".format(database_file))
    tables = sm.get_tables(database_file)
    for table, in tables:
        export_file(database_file, export_dir, table)
    log.info("{0} Completed export process".format(database_file))

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
    if endswith: files = [file for file in files if os.path.splitext(file)[1] == endswith]
    if full: files=["{0}{1}".format(directory,file) for file in files]
    return files

def extract_archive(archive_dir, extract_dir, extract_filter = None):
    log.info("%s | Started extraction process", archive_dir)
    os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
    if extract_filter: setuptools.archive_util.unpack_archive(archive_dir, extract_dir, progress_filter=extract_filter)
    else: setuptools.archive_util.unpack_archive(archive_dir, extract_dir)
    log.info("%s | Completed extraction process", archive_dir)

def get_datashape(odo_resource):
    dshape = odo.discover(odo_resource, engine="c", has_header=True, encoding="utf-8", errors="ignore", lineterminator="\n",quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True)
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

def load_file(abs_source_file, database_file, drop=False):
    source_file = os.path.basename(abs_source_file)
    source_name = source_file.split(".")[0]
    log.info("%s | Import started", source_file)
    db_uri = ("sqlite:///"+ database_file +"::"+ source_name)
    if drop: odo.drop(db_uri)
    odo_resource = odo.resource(abs_source_file, engine="c", has_header=True,encoding="utf-8", errors="ignore", lineterminator="\n",quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True)
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
    clear_files(cm.database_file, cm.csv_extract_dir)
    extract_archive(cm.csv_archive_file, cm.csv_extract_dir)
    load_files(cm.csv_extract_dir, cm.database_file)

if __name__ == "__main__":
    main()
