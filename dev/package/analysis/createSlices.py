import dbLoader as db
import sqlManager as sm
import os
import sqlite3
from datetime import date
from dateutil.rrule import rrule, YEARLY

def make_slice(input_path, output_path, date):
    tables = sm.get_tables(input_path)
    input_abs_path = os.path.abspath(input_path)
    with sqlite3.connect(output_path) as conn:
        c = conn.cursor()
        c.execute("ATTACH DATABASE '{0}' AS src;".format(input_path))
        for table in tables:
            c.execute("SELECT sql FROM src.sqlite_master WHERE type='table' AND name='{0}'".format(table))
            c.execute(c.fetchone()[0])
            try: c.execute("INSERT INTO main.{0} SELECT * FROM src.{0} WHERE created_at < '{1}';".format(table, date))
            except:
                try: c.execute("INSERT INTO main.{0} SELECT * FROM src.{0} WHERE started_on < '{1}';".format(table, date))
                except: c.execute("INSERT INTO main.{0} SELECT * FROM src.{0};".format(table, date))
            finally: print("Table import complete: {0}".format(table))

def main():
    input_path = "analysis/output/thirteen/input.db"
    for year in range(2013, 2010, -1):
        new_date = date(year, 6, 1).strftime("%Y-%m-%d")
        output_path = "analysis/output/thirteen/{0}.db".format(new_date)
        make_slice(input_path, output_path, new_date)

if __name__ == "__main__":
    main()