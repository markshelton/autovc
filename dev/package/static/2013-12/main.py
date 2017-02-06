#!/Anaconda3/env/honours python

"""sqlManager.py"""

#standard modules
import logging

#third-party modules

#local modules
from . import dbLoader

#constants
input_file = "2013-Dec-xx_mysql.tar.gz"
extract_dir = "extract/"

def main():
    cm = db.load_config()
    db.clear_files()
    db.extract_archive(input_file, extract_dir)
    #db.load_files(cm.csv_extract_dir, cm.database_file)

if __name__ == "__main__":
    main()
