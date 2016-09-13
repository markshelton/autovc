#merge_sql_relations.py

import sqlite3

DATABASE_FILE = "../data/raw/2016-Sep-09_sqlite.db"

con = sqlite3.connect(DATABASE_FILE)
con.text_factory = str
c = con.cursor()

table_list = []
sql = "SELECT name FROM sqlite_master WHERE type = 'table'"
for table_name in c.execute(sql):
    table_list.append(table_name[0])
print (table_list)

sql = "CREATE VIEW test AS SELECT *, (a.acquiree_uuid is not Null) as acquired FROM organizations o LEFT OUTER JOIN acquisitions a ON o.uuid = a.acquiree_uuid"

print (sql)
