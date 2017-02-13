#!/Anaconda3/env/honours python

"""sqlManager.py"""

#standard modules
import logging
import sqlite3

#third-party modules

#local modules
import dbLoader as db

#constants


#logger
log = logging.getLogger(__name__)

#helper functions

def get_uuids(tables, database):
    uuids = {}
    for table, in tables:
        query = "SELECT uuid FROM {0}".format(table)
        with sqlite3.connect(database) as conn:
            try: uuid = conn.execute(query)
            except: pass
            else: uuids[table] = uuid
    return uuids

def get_tables(database):
    with sqlite3.connect(database) as connection:
        query = "SELECT name FROM sqlite_master WHERE type=\'table\'"
        try:
            tables = connection.execute(query)
            tables = [table[0] for table in tables]
        except: tables = list()
    return tables

#core functions

def main():
    cm = db.load_config()
    database = cm.database_file
    tables = get_tables(database)
    uuids = get_uuids(tables, database)
    db.export_files(database, cm.export_dir)


if __name__ == "__main__":
    main()


#graveyard

"""
features = {
    "has_investment_any":,
    "has_investment_seed":,
    "has_investment_a":,
    "has_investment_b":,
    "has_investment_c":,
    "has_investment_d":,
    "has_investment_e":,
    "has_investment_pe":,
    "has_ipo":,
    "num_current_employees":,
    "num_members":,
    "num_founders":,
    "num_board":,
    "num_degrees_founders":,
    "num_experience_founders":,
    "founder_has_mba":,
    "founder_has_phd":,
    "num_prev_startups_founders":,
}
"""
