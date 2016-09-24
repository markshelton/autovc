import odo
import sqlite3

with sqlite3.connect('example.db') as conn:
    conn.text_factory = str
    c = conn.cursor()



odo.odo("acquisitions.csv", 'sqlite:///example.db::acquisitions')
