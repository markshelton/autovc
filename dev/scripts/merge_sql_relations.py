#!/Anaconda3/env/honours python

"""merge_sql_relations"""

import sqlite3

from log import Log

DATABASE_FILE = "../data/raw/2016-Sep-09_sqlite.db"

def load_database(database_file):
    con = sqlite3.connect(DATABASE_FILE)
    con.text_factory = str
    return con

def create_view(db_connection):
    c = db_connection.cursor()
    sql = "DROP VIEW IF EXISTS test"
    c.execute(sql)
    sql = """
        CREATE VIEW test AS
            SELECT *,
                (acquiree_uuid is not Null) as acquired
            FROM organizations o
            LEFT OUTER JOIN
                (SELECT acquiree_uuid
                FROM acquisitions)
            ON o.uuid = acquiree_uuid
            WHERE primary_role = 'company'"""
    try:
        c.execute(sql)
        log.info("%s | SUCCESSFUL", sql)
    except:
        log.error("%s | FAILED", sql, exc_info=1)

def main():
    log = Log(__file__).logger
    db_connection = load_database(DATABASE_FILE)
    create_view(db_connection)

if __name__ == "__main__":
    main()

