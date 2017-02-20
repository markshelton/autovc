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
import numpy as np
import pandas as pd
import scipy.stats

#local modules
import logManager
from logManager import logged
from configManager import configManager
import sqlManager as sm

#constants

#configManager
def load_config(config_dir=None):
    return configManager(config_dir)

#logger
log = logging.getLogger(__name__)

odo_args = dict(engine="c", has_header=True,encoding="utf-8", errors="ignore", lineterminator="\n",quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True)

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

@logged
def clear_files(*paths):
    for path in paths:
        clear_file(path)

def get_files(directory, endswith=None, full=True):
    files = os.listdir(directory)
    if endswith: files = [file for file in files if os.path.splitext(file)[1] == endswith]
    if full: files=["{0}{1}".format(directory,file) for file in files]
    return files

def normality_test(df):
    numerical = df.select_dtypes(include=[np.number]).columns
    norm_chi, norm_pval = {}, {}
    for column in df[numerical]:
        try:
            stat = scipy.stats.normaltest(df[column],nan_policy='omit')
            norm_chi[column], norm_pval[column] = stat
        except: pass
    norm_chi = pd.Series(norm_chi)
    norm_pval = pd.Series(norm_pval)
    return norm_chi, norm_pval

@logged
def explore(df):
    stats_num = pd.DataFrame(df.describe())
    categorical = df.dtypes[df.dtypes == "object"].index
    stats_cat = pd.DataFrame(df[categorical].describe())
    stats = pd.concat([stats_num, stats_cat],axis=1)
    missing = [len(df.index) - df[column].count() for column in stats]
    datatypes = stats.dtypes
    stats = stats.transpose()
    norm_chi, norm_pval = normality_test(df)
    norm_chi = norm_chi.reindex(stats.index)
    norm_pval = norm_pval.reindex(stats.index)
    skew = df.skew(axis=0).reindex(stats.index)
    kurtosis = df.kurt(axis=0).reindex(stats.index)
    stats["norm_k2"] = norm_chi
    stats["norm_pval"] = norm_pval
    stats["missing"] = missing
    stats["skew"] = skew
    stats["kurtosis"] = kurtosis
    stats["attribute"] = stats.index
    stats["datatype"] = datatypes
    return stats

@logged
def summarise_files(export_dir, dictionary_dir=None, dict_file=None):
    if not dictionary_dir: dictionary_dir = export_dir+"dictionary/"
    if not dict_file: dict_file = dictionary_dir + "dict.csv"
    os.makedirs(dictionary_dir)
    stats = pd.DataFrame()
    for file in get_files(export_dir,endswith=".csv",full=True):
        short = os.path.basename(file)
        try: df = odo.odo(file, pd.DataFrame)
        except: df = odo.odo(file, pd.DataFrame, **odo_args)
        stat = explore(df)
        try: odo.odo(stat, dictionary_dir+short)
        except: odo.odo(stat, dictionary_dir+short, **odo_args)
        stat["table"] = short.split(".")[0]
        stats = pd.concat([stat, stats])
    odo.odo(stats, dict_file)

@logged
def extract_archive(archive_dir, extract_dir, extract_filter = None):
    os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
    if extract_filter: setuptools.archive_util.unpack_archive(archive_dir, extract_dir, progress_filter=extract_filter)
    else: setuptools.archive_util.unpack_archive(archive_dir, extract_dir)

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
