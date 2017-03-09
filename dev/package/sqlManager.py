#!/Anaconda3/env/honours python

"""sqlManager.py"""

#standard modules
import logging
import sqlite3
import os

#third-party modules
import sqlalchemy
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

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
        uri = db.build_uri(database, table, db_type="postgresql")
        engine = sqlalchemy.create_engine(uri)
        with engine.connect() as conn:
            try: uuid = conn.execute(query)
            except: pass
            else: uuids[table] = uuid
    return uuids

def get_tables(database):
    uri = db.build_uri(database, db_type="postgresql")
    m = sqlalchemy.MetaData()
    engine = sqlalchemy.create_engine(uri)
    m.reflect(engine)
    tables = [table.name for table in m.tables.values()]
    return tables

def drop_database(database_file):
    db_name = os.path.basename(database_file).split(".")[0]
    uri = db.build_uri("postgres", db_type="postgresql", create=False)
    engine = sqlalchemy.create_engine(uri,isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        try: conn.execute("DROP DATABASE \"{0}\";".format(db_name))
        except: log.info("{0} | Database already dropped".format(db_name))

#core functions

def main():
    cm = db.load_config()
    database = cm.database_file
    tables = get_tables(database)
    uuids = get_uuids(tables, database)
    db.export_files(database, cm.export_dir)

if __name__ == "__main__":
    main()