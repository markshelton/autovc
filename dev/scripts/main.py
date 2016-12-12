#!/Anaconda3/env/honours python

"""main"""

#standard modules
import logging
import sqlite3
import json

#third-party modules
import requests

#local modules
import logManager as lm
import dbLoader as db
from configManager import configManager

#constants

table_pk_lookup = {
    "acquisitions":"uuid",
    "funding_rounds":"uuid",
    "funds":"uuid",
    "ipos":"uuid",
    "organizations":"permalink",
    "people":"permalink",
    "products":"permalink"
}

api_table_lookup = {
    "acquisitions":"acquisitions",
    "funding_rounds":"funding-rounds",
    "funds":"funds",
    "ipos":"ipos",
    "organizations":"organizations",
    "people":"people",
    "products":"products"
}

#logger
log = logging.getLogger(__name__)

#functions

def request_api(node_type, node_id, cb_key):
    url = ("https://api.crunchbase.com/v/3/%s/%s?user_key=%s"
        % (node_type, node_id, cb_key))
    response = requests.get(url)
    return response

def load_node_list(archive_file, extract_dir, database_file, cb_key):
    response = request_api("node_keys","node_keys.tar.gz",cb_key)
    with open(archive_file, 'wb') as f:
        f.write(response.content)
    db.load_database(archive_file, extract_dir, database_file)

def parse_json(json_record):
    record = {}
    record["uuid"] = json_record["data"]["uuid"]
    for field in json_record["data"]["properties"].keys():
        record[field] = json_record["data"]["properties"][field]
    return record

def store_data(record, table, database_file):
    conn_db = sqlite3.connect(database_file)
    #TODO
    conn_db.close()

def load_data(nl_database_file, database_file, cb_key):
    conn_nl = sqlite3.connect(nl_database_file)
    query = "SELECT name FROM sqlite_master WHERE type=\'table\'"
    tables = conn_nl.execute(query)
    for table, in tables:
        query = "SELECT %s FROM %s" % (table_pk_lookup[table],table)
        records = conn_nl.execute(query)
        for record, in records:
            response = request_api(api_table_lookup[table],record,cb_key)
            data = parse_json(response.json())
            store_data(data, table, database_file)
    conn_nl.close()

def main():
    cm = db.load_config()
    load_node_list(cm.nl_archive_file, cm.nl_extract_dir,cm.nl_database_file, cm.cb_key)
    load_data(cm.nl_database_file, cm.database_file, cm.cb_key)

if __name__ == "__main__":
    main()

