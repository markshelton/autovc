#!/Anaconda3/env/honours python

"""dbLoader"""

#standard modules
import os
import tarfile
import logging
import shutil

#third-party modules
from odo import odo

#local modules
import logManager
from configManager import configManager

#constants

#logger
log = logging.getLogger(__name__)

#functions
def load_config(config_dir=None):
    return configManager(config_dir)

def clear_files(*paths):

    def clear_file(path):
        if os.path.isfile(path):
            os.remove(path)
            log.info("%s | File deleted", path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
            log.info("%s | Directory deleted", path)
        else:
            log.info("%s | Already deleted", path)

    log.info("Started clear process")
    for path in paths:
        clear_file(path)
    log.info("Completed clear process")

def extract_archive(achive_dir, extract_dir, target_type):

    def extract_file(archive, archive_file, extract_dir):
        if os.path.exists(os.path.join(extract_dir, archive_file)):
            log.info("%s | Already extracted", archive_file)
        else:
            log.info("%s | Extraction started", archive_file)
            try:
                archive.extract(archive_file, path=extract_dir)
                log.info("%s | Extraction successful", archive_file)
            except:
                log.error("%s | Extraction failed", archive_file,exc_info=True)

    log.info("%s | Started extraction process", achive_dir)
    with tarfile.open(achive_dir) as archive:
        for archive_file in archive.getnames():
            if archive_file.endswith(target_type):
                extract_file(archive, archive_file, extract_dir)
    log.info("%s | Completed extraction process", achive_dir)

def load_files(extract_dir, database_file):

    def load_file(source_file, database_file):
        source_file = os.path.basename(source_file)
        log.info("%s | Import started", source_file)
        try:
            db_uri = ("sqlite:///"+ database_file
                    +"::"+ source_file.split(".")[0])
            odo(abs_source_file, db_uri)
            log.info("%s | Import successful", source_file)
        except:
            log.error("%s | Import failed", source_file, exc_info=True)

    log.info("%s | Started import process", database_file)
    for base_name in os.listdir(extract_dir):
        source_file = extract_dir + base_name
        load_file(source_file, database_file)
    log.info("%s | Completed import process", database_file)

def main():
    cm = load_config()
    clear_files(cm.extract_dir, cm.database_file)
    extract_archive(cm.archive_file, cm.extract_dir, cm.target_type)
    load_files(cm.extract_dir, cm.database_file)

if __name__ == "__main__":
    main()
