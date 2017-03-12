#!/Anaconda3/env/honours python

"""sqlManager.py"""

#standard modules
import logging
import sqlite3
import os

#third-party modules
import sqlalchemy
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import odo

#local modules
import dbLoader as db

#constants


#logger
log = logging.getLogger(__name__)

#helper functions

def get_uuids(tables, database, db_type="sqlite"):
    uuids = {}
    for table, in tables:
        query = "SELECT uuid FROM {0}".format(table)
        uri = db.build_uri(database, table, db_type=db_type)
        engine = sqlalchemy.create_engine(uri)
        with engine.connect() as conn:
            try: uuid = conn.execute(query)
            except: pass
            else: uuids[table] = uuid
    return uuids

def get_permalinks(database, table, db_type="sqlite"):
    permalinks = []
    query = "SELECT permalink FROM {0};".format(table)
    try:
        uri = db.build_uri(database, table, db_type=db_type)
        engine = sqlalchemy.create_engine(uri)
        with engine.connect() as conn:
            permalinks = conn.execute(query)
    except: pass
    return permalinks

def get_tables(database, db_type="sqlite"):
    uri = db.build_uri(database, db_type=db_type)
    m = sqlalchemy.MetaData()
    engine = sqlalchemy.create_engine(uri)
    m.reflect(engine)
    tables = [table.name for table in m.tables.values()]
    return tables

def drop_database(database_file, db_type="sqlite"):
    db_name = os.path.basename(database_file).split(".")[0]
    if db_type == "sqlite":
        try: odo.drop(database_file)
        except: log.info("{0} | Database already dropped".format(db_name))
    else:
        db_name = os.path.basename(database_file).split(".")[0]
        uri = db.build_uri("postgres", db_type="postgresql", create=False)
        engine = sqlalchemy.create_engine(uri,isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            try: conn.execute("DROP DATABASE \"{0}\";".format(db_name))
            except: log.info("{0} | Database already dropped".format(db_name))

def create_db(scripts_dir, database, db_type="sqlite"):
    if db_type == "postgresql": pass
    else:
        log.info("Started Database Build process")
        with sqlite3.connect(database) as conn:
            for sql_file in db.get_files(scripts_dir):
                with codecs.open(sql_file,encoding="latin-1") as file:
                    file_content = file.read()
                sql_short = os.path.basename(sql_file)
                log.info("{0} | Table Build Started".format(sql_file))
                try: conn.executescript(file_content)
                except:
                    conn.execute("ROLLBACK")
                    log.error("{0} | Table Build Failed".format(sql_short),exc_info=True)
                else: log.info("{0} | Table Build Successful".format(sql_short))
        log.info("Completed Database Build process")

#core functions

def main():
    cm = db.load_config()
    database = cm.database_file
    tables = get_tables(database)
    uuids = get_uuids(tables, database)
    db.export_files(database, cm.export_dir)

if __name__ == "__main__":
    main()