#!/Anaconda3/env/honours python

"""dbLoader"""

#standard modules
import os
import tarfile
import logging

#third-party modules
from odo import odo

#local modules
import logManager
from configManager import configManager

#constants

#logger
log = logging.getLogger(__name__)

#program
class dbLoader(configManager):

    def __init__(self):
        self.load_config()
        self.extract_archive(
            self.archive_file,
            self.extract_dir,
            self.target_type
        )
        self.load_files(
            self.extract_dir,
            self.database_file
        )

    def load_config(self):
        super(dbLoader, self).__init__()

    def extract_archive(self, achive_file, extract_dir, target_type):
        log.info("%s | Started extraction process", achive_file)
        with tarfile.open(achive_file) as archive:
            for name in archive.getnames():
                if name.endswith(target_type):
                    if os.path.exists(os.path.join(extract_dir, name)):
                        log.debug("%s | Already extracted", name)
                    else:
                        log.debug("%s | Extraction started", name)
                        try:
                            archive.extract(name, path=extract_dir)
                            log.debug("%s | Extraction successful",name)
                        except:
                            log.error("%s | Extraction failed", name)
        log.info("%s | Completed extraction process", achive_file)

    def load_files(self, extract_dir, database_file):
        log.info("%s | Started import process", database_file)
        for file in os.listdir(extract_dir):
            try:
                log.debug("%s | Import started", file)
                db_uri = ("sqlite:///"+database_file+"::"+file.split(".")[0])
                csv_file = extract_dir + file
                odo(csv_file, db_uri)
                log.debug("%s | Import successful", file)
            except:
                log.error("%s | Import failed", file)
        log.info("%s | Completed import process", database_file)

#testing
def main():
    db = dbLoader()

if __name__ == '__main__':
    main()
