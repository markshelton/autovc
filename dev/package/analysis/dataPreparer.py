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

#logger
log = logging.getLogger(__name__)

@logged
def read(file,nrows=None):
    if nrows is None: df = pd.read_csv(file,low_memory=False)
    else: df = pd.read_csv(file,low_memory=False,nrows=nrows)
    return df

@logged
def clean(df):

    @logged
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

        def get_excluded():
            stop_words = get_stop_words('english')
            punctuation = list(string.punctuation)
            excluded = stop_words + punctuation
            return excluded

        def prepare_text(series, topn, sep=None):
            for punc in list(string.punctuation):
                series = series.str.replace(punc, "")
            series = series.replace(r'\s+', np.nan,regex=True)
            series = series.dropna()
            series = series.apply(lambda x: unidecode(str(x)))
            excluded = go_filter(series, get_excluded(), mode="excluded")
            series = series[series.index.difference(excluded.index)]
            return series

        def make_dummies(series, topn=None, sep=None, text=False):
            series_name = series.name
            series = series.str.lower()
            if sep: series = series.str.split(sep,expand=True).stack()
            if text: series = prepare_text(series, topn, sep)
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
            df = df.add_prefix(series_name+"_")
            return df

        def dropna(x):
            return pd.Series([str(y) for y in x if type(y) is dict])

        def suffix_keys(x):
            c = defaultdict(list)
            for d in x:
                d = eval(d) # e.g. {'series-a: 2}
                d = {k: int(v) for k,v in d.items()}
                for k,v in d.items():
                    c.update(d)
            b = {}
            for k, v in c.items():
                if type(v) is list:
                    input(v)
                    v = sorted(v)
                    for i, x in enumerate(v):
                        b["{}_{}".format(k, i)] = x
                else: b[k] = v
            return b

        def sum_dicts(x):
            c = Counter()
            for d in x:
                d = eval(d)
                d = {k: float(v) for k,v in d.items()}
                c.update(d)
            c = dict(c)
            return c

        #todo - [a 5; b 4; a 3] // index: dummy: value
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
            df = df.add_prefix(series_name+"_")
            df = df.add_suffix("_"+suffix)
            return df

        def go_dates(series, date_data):

            def match_date_data(date, date_data):
                try:
                    date = datetime.datetime.fromtimestamp(date).strftime("%Y-%m-%d")
                    result = date_data["Close"].ix[date]
                except: result = 0
                return result

            new = series.apply(lambda x: match_date_data(x, date_data)) # BROKEN
            new.name = "confidence_context_broader_" + series.name+"_"+"SP500"+"_"+"number"
            #series.replace(np.nan, 0, inplace=True)
            df = pd.concat([series, new],axis=1)
            return df

        @logged
        def go_gender(series, sep):

            def get_gender(names, sep):
                #input(names)
                try:
                    name_list = names.split(sep)
                    c = Counter()
                    for name in name_list:
                        d = gender.Detector()
                        sex = d.get_gender(name)
                        if sex == "mostly_male": sex = "male"
                        elif sex == "mostly_female": sex = "female"
                        elif sex == "andy": sex = "unknown"
                        c.update([sex])
                    names = ["{} {}".format(k,v) for (k,v) in c.items()]
                    names = ";".join(names)
                except: pass
                return names

            series = series.apply(lambda x: get_gender(x, sep))
            series.name = series.name + "_"+"number"
            series = combine_pairs(series,sep=sep)
            return series

        column = column.split(":")[0]
        print(column)
        if column.endswith("bool"): temp = df[column]
        elif column.endswith("date"): temp = go_dates(df[column], date_data=date_data)
        elif column.endswith("duration"): temp = df[column]
        elif column.endswith("dummy"): temp = make_dummies(df[column],topn=10)
        elif column.endswith("types_list"): temp = make_dummies(df[column],sep=";")
        elif column.endswith("codes_list"): temp = make_dummies(df[column],sep=";")
        elif column.endswith("list"): temp = make_dummies(df[column],topn=10,sep=";")
        elif column.endswith("text"): temp = make_dummies(df[column],topn=10,sep=" ",text=True)
        elif column.endswith("number"): temp = pd.to_numeric(df[column], errors="ignore").fillna(0)
        elif column.endswith("pair"): temp = combine_pairs(df[column],sep=";")
        elif column.startswith("keys"): temp = df[column]
        else: temp = pd.DataFrame()
        return temp

    def get_date_data(ticker, start="19700101", end="20170301"):
        path = "https://stooq.com/q/d/l/?s={}&i=d&d1={}&d2={}".format(ticker, start, end)
        try:
            response = requests.get(path)
            os.makedirs(os.path.dirname(date_data_file), exist_ok=True)
            with open(date_data_file, 'wb') as f:
                f.write(response.content)
        except: pass
        df = pd.read_csv(date_data_file, index_col="Date")
        return df

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

    def create_null_dummies(df):
        pass

    df_new = pd.DataFrame()
    date_data = get_date_data("^spx")
    for column in df:
        temp = handle_column(df, column, date_data=date_data)
        if temp is not None and not temp.empty:
            df_new = pd.concat([df_new, temp],axis=1)
    df_new.columns = [unidecode(x).strip().replace(" ","-") for x in list(df_new)]
    dates = [col for col in list(df) if col.endswith("date")]
    temp = create_durations(df_new[dates])
    df_new = pd.concat([df_new, temp], axis=1)
    temp = create_null_dummies(df_new)
    df_new = pd.concat([df_new, temp], axis=1)
    #df_new.replace(np.nan, 0, inplace=True)
    #input(df_new.head())
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
        else: df = pd.read_sql_table(table, conn)
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

if __name__ == "__main__":
    main()
