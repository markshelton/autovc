#!/Anaconda3/env/honours python

"""data_preparer"""

#standard modules
from collections import Counter, defaultdict
import datetime
import sqlite3
import logging
import string
import re
import itertools
import os
import urllib

#third party modules
from unidecode import unidecode
from stop_words import get_stop_words
import gender_guesser.detector as gender
import numpy as np
import pandas as pd
import sqlalchemy
import requests

#local modules
from logManager import logged
import dbLoader as db
import sqlManager as sm

#constants
files = {}
files["thirteen"] = {}
files["thirteen"]["database_file"] = "collection/thirteen/output/2013-Dec.db"
files["thirteen"]["flat_raw_file"] = "analysis/output/flatten/raw/thirteen.csv"
files["thirteen"]["flat_clean_file"] = "analysis/output/flatten/clean/thirteen.csv"
files["thirteen"]["flatten_config"] = "analysis/config/flatten/thirteen.sql"
files["sixteen"] = {}
files["sixteen"]["database_file"] = "collection/sixteen/output/2016-Sep.db"
files["sixteen"]["flat_raw_file"] = "analysis/output/flatten/raw/sixteen.csv"
files["sixteen"]["flat_clean_file"] = "analysis/output/flatten/clean/sixteen.csv"
files["sixteen"]["flatten_config"] = "analysis/config/flatten/sixteen.sql"

output_table = "combo"
merge_config = "analysis/config/flatten/merge.sql"
database_file = "analysis/output/combo.db"
date_data_file = "analysis/output/temp/date_data.csv"

codes = ["a","b","c","d","e","f","g","h"]
types = ["angel","convertible_note","debt_financing","equity_crowdfunding","grant",
    "non_equity_assistance","post_ipo_debt","post_ipo_equity","private_equity",
    "product_crowdfunding","secondary_market","seed","undisclosed","venture"]

#logger
log = logging.getLogger(__name__)

@logged
def read(file,nrows=None):
    if nrows is None: df = pd.read_csv(file,low_memory=False)
    else: df = pd.read_csv(file,low_memory=False,nrows=nrows)
    return df

@logged
def clean(df):

    def handle_column(df, column, date_data=None):

        def go_filter(series, patterns, mode="included"):
            series = series.dropna()
            total = pd.Series()
            for x in patterns:
                pattern = "^{0}$".format(x)
                if mode == "excluded":
                    pattern = r"^(" + re.escape(x) + r"$).*"
                temp = series[series.str.match(pattern, as_indexer=True)]
                total = pd.concat([total, temp])
            return total

        def make_dummies(series, topn=None, sep=None, text=False):
            series_name = series.name
            series = series.str.lower()
            if sep: series = series.str.split(sep,expand=True).stack()
            if topn:
                counts = series.value_counts()
                included = counts.nlargest(topn).index
                series = go_filter(series, included)
            if sep:
                series.index = pd.MultiIndex.from_tuples(series.index)
                df = series.unstack()
                df = df.apply(lambda x: sep.join(x.dropna()),axis=1)
                series = df.squeeze()
                df = series.str.get_dummies(sep=sep)
            else: df = pd.get_dummies(series)
            df = df.rename(columns=str.lower)
            if "type" in series_name: sublist = types
            elif "code" in series_name: sublist = codes
            else: sublist = None
            if sublist is not None:
                for x in sublist:
                    if x not in list(df): df[x] = np.nan
                df = df[sublist]
            df = df.add_prefix(series_name+"_")
            return df

        def dropna(x):
            return pd.Series([str(y) for y in x if type(y) is dict])

        def suffix_keys(x):
            c = defaultdict(list)
            for d in x:
                c.update({k: int(v) for k,v in eval(d).items()})
            b = {}
            for k, v in c.items():
                if type(v) is list:
                    for i, x in enumerate(sorted(v)):
                        b["{}_{}".format(k, i)] = x
                else: b[k] = v
            return b

        def sum_dicts(x):
            c = Counter()
            for d in x:
                try: c.update({k: float(v) for k,v in eval(d).items()})
                except: pass
            c = dict(c)
            return c

        def combine_pairs(series, sep):
            series_name = series.name
            suffix = series_name.split("_")[-1]
            series = series.str.split(sep,expand=True).stack()
            series = series.apply(lambda x: dict([tuple(x.split(" "))]))
            df = series.unstack()
            series = df.apply(lambda x: ",".join(dropna(x)).split(","),axis=1)
            if series_name.endswith("date_pair"): series = series.apply(lambda x: suffix_keys(x))
            else: series = series.apply(lambda x: sum_dicts(x))
            df = pd.DataFrame.from_records(series.tolist(), index=series.index)
            df = df.rename(columns=str.lower)
            if "type" in series_name: sublist = types
            elif "code" in series_name: sublist = codes
            else: sublist = None
            if sublist is not None:
                for x in sublist:
                    if x not in list(df): df[x] = np.nan
                df = df[sublist]
            df = df.add_prefix(series_name+"_")
            df = df.add_suffix("_"+suffix)
            return df

        column = column.split(":")[0]
        if column.endswith("bool"): temp = df[column]
        elif column.endswith("date"): temp = df[column]
        elif column.endswith("duration"): temp = df[column]
        elif column.endswith("pair"): temp = combine_pairs(df[column],sep=";")
        elif column.endswith("types_list"): temp = make_dummies(df[column],sep=";")
        elif column.endswith("codes_list"): temp = make_dummies(df[column],sep=";")
        elif column.endswith("list"): temp = make_dummies(df[column],topn=10,sep=";")
        #elif column.endswith("dummy"): temp = make_dummies(df[column],topn=10)
        #elif column.endswith("text"): temp = make_dummies(df[column],topn=10,sep=" ",text=True)
        elif column.endswith("number"): temp = pd.to_numeric(df[column], errors="coerce")
        elif column.startswith("keys"): temp = df[column]
        else: temp = pd.DataFrame()
        return temp

    def create_durations(df):
        df = df.replace(to_replace = 0, value = np.nan)
        combos = list(itertools.combinations(list(df), 2))
        durations = []
        temp = pd.DataFrame()
        for x, y in combos:
            values = df[x] - df[y]
            label = "{}_to_{}_duration".format(y,x)
            new = pd.Series(values, name=label)
            temp = pd.concat([temp, new],axis=1)
        return temp

    df_new = pd.DataFrame()
    for column in df:
        try: temp = handle_column(df, column)
        except:
            log.error("Error with Column: {0}".format(column),exc_info=1)
            temp = df[column]
        if temp is not None and not temp.empty:
            df_new = pd.concat([df_new, temp],axis=1)
    df_new.columns = [unidecode(x).strip().replace(" ","-") for x in list(df_new)]
    return df_new

@logged
def flatten(database_file, script_file):
    sm.execute_script(database_file, script_file)

@logged
def flatten_file(database_file, config_file, export_file, file_name):
    flatten(database_file, config_file)
    db.export_file(database_file, export_file, file_name)

@logged
def clean_file(raw_file, clean_file,nrows=None):
    df = read(raw_file, nrows=nrows)
    df = clean(df)
    df.to_csv(clean_file, mode="w+", index=False)

@logged
def load_file(database_file, clean_file, file_name, index=False):
    if index: df = pd.read_csv(clean_file, encoding="latin1", index_col=0)
    else: df = pd.read_csv(clean_file, encoding="latin1")
    with sqlite3.connect(database_file) as conn:
        df.to_sql(file_name, conn, if_exists='append', index=index)

@logged
def merge(database_file, script_file):
    sm.execute_script(database_file, script_file)

@logged
def export_dataframe(database_file, table, index=False):
    uri = db.build_uri(database_file, table=None, db_type="sqlite")
    engine = sqlalchemy.create_engine(uri)
    with engine.connect() as conn:
        if index: df = pd.read_sql_table(table, conn, index_col ="index")
        else:
            df_list = pd.read_sql_table(table, conn, chunksize=10000)
            df = pd.concat([chunk for chunk in df_list],ignore_index=True)
    return df

def main():
    nrows = None
    db.clear_files(database_file)
    #del files['sixteen']
    for file_name, file in files.items():
        flatten_file(file["database_file"], file["flatten_config"], file["flat_raw_file"], file_name)
        clean_file(file["flat_raw_file"], file["flat_clean_file"],nrows=nrows)
        load_file(database_file, file["flat_clean_file"], file_name)
    merge(database_file, merge_config)
    #df = export_dataframe(database_file, output_table)
    #print(df)

def test():
    path = 'C:/Users/mark/Documents/GitHub/honours/dev/package/'
    input_path = path+"analysis/input/master.db"
    flatten_config = path+"analysis/config/master_feature.sql"
    raw_flat_file = path+"analysis/output/temp/raw.csv"
    clean_flat_file = path+"analysis/output/temp/clean.csv"
    output_path = path+"analysis/output/temp/output.db"

    #flatten_file(input_path, flatten_config, raw_flat_file, "feature")
    clean_file(raw_flat_file, clean_flat_file)
    load_file(output_path, clean_flat_file, "feature")
    df = export_dataframe(output_path, "feature")

if __name__ == "__main__":
    test()
