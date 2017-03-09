#!/Anaconda3/env/honours python

"""__main__.py"""

#standard modules
import logging
import os
import sqlite3

#third-party modules
import yaml
import yamlordereddictloader

#local modules
import dbLoader as db
from logManager import logged

#constants
input_file = "collection/fifteen/input/2015-Dec-04_csv.zip"
extract_dir = "collection/fifteen/output/extract/"
export_dir = "collection/fifteen/output/export/"
database_file = "collection/fifteen/output/2015-Dec.db"
config_dir = "collection/fifteen/config/"
dict_dir = "collection/fifteen/output/export/dict/"
dict_file = "collection/fifteen/output/dict.csv"
flat_file = "collection/fifteen/output/2015-Dec.csv"
sql_file = "collection/fifteen/flatten.sql"
flat_reference = "collection/fifteen/config/_flatten.yaml"

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

def load_yaml(path):
    if os.path.exists(path):
        with open(path, 'rt') as f:
            output = yaml.load(f.read(),Loader=yamlordereddictloader.Loader)
            return output

def main():
    #cm = db.load_config(config_dir)
    extract()       #Done
    load()          #Done
    export()         #Done
    explore()
    #flatten()         #Pending

if __name__ == "__main__":
    main()

#graveyard


def get_round_type():
    pass

def get_category():
    pass

@logged
def add_attribute(conn, table, attribute_name, attribute_formula):
    conn.execute("DROP TABLE IF EXISTS temp_one")
    conn.execute("DROP TABLE IF EXISTS temp_two")
    conn.execute("CREATE TABLE temp_one AS SELECT * FROM flat")
    conn.execute("CREATE TABLE temp_two AS SELECT permalink, {0} AS {1} FROM {2} NATURAL JOIN flat".format(attribute_formula,attribute_name,table))
    conn.execute("DROP TABLE IF EXISTS flat")
    conn.execute("CREATE TABLE flat AS SELECT * FROM temp_one NATURAL JOIN temp_two")

@logged
def build_sql_old(ref):
    with sqlite3.connect(database_file) as conn:
        for table in ref:
            conn.execute("DROP TABLE IF EXISTS flat")
            conn.execute("CREATE TABLE flat AS SELECT permalink FROM {0}".format(table))
            for attribute in ref[table]:
                value = ref[table][attribute]
                if value.startswith("//"):
                    value = value.replace("//","")
                    value = eval(value)
                else: add_attribute(conn, table, attribute, value)
            conn.execute("DROP TABLE IF EXISTS temp_one")
            conn.execute("DROP TABLE IF EXISTS temp_two")

def build_sql(sql_file):
    with sqlite3.connect(database_file) as conn:
        with open(sql_file) as file:
            script = file.read()
        conn.executescript(script)

@logged
def flatten():
    db.clear_files(flat_file)
    #ref = load_yaml(flat_reference)
    #build_sql(ref)
    build_sql(sql_file)
    #TODO: Flatten