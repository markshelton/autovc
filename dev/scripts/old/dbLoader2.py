import os
import csv


def create_database(database_file):
    db_connection = sqlite3.connect(database_file)
    db_connection.text_factory = str
    return db_connection

def

def create_table(csv_file, db_connection):
    with csv_reader(csv_file) as read:
        for attribute in read.headers:


def create_tables(extract_dir, db_connection):
    for file in os.listdir(extract_dir):
        csv_file = extract_dir + file
        table = create_table(file, db, )


def load_files()
