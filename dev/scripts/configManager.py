#!/Anaconda3/env/honours python

"""configManager"""

#standard modules
import configparser
import logging
import os

#third-party modules

#local modules
import logManager

#constants
CONFIG_DIR = "../config/"

#logger
log = logging.getLogger(__name__)

#program
class configManager(object):

    def __init__(self, config_dir=CONFIG_DIR):
        log.info("Started config loading process")
        file_names = self.find_config_files(config_dir)
        self.read_config_files(file_names)
        log.info("Completed config loading process")
        self.log_config()

    def find_config_files(self, config_dir):

        def search_directory(config_dir):
            file_names = []
            for file in os.listdir(config_dir):
                file_names.append(config_dir+file)
            return file_names

        try:
            file_names = search_directory(config_dir)
            log.info("%s | config directory loaded", config_dir)
        except:
            log.warn("%s | config directory failed", config_dir)
            file_names = search_directory(CONFIG_DIR)
            log.info("Default config directory loaded")
        return file_names

    def read_config(self, file_name, attributes=None):
        if attributes is None: attributes = []
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(file_name)
        for section in parser.sections():
            config_data = parser.items(section)
            self.__dict__.update(config_data)
            if section == "config":
                for (key, value) in config_data:
                    attributes.append(key)
                    self.read_config(value, attributes)

    def read_config_files(self, file_names):
        for file_name in file_names:
            try:
                self.read_config(file_name)
                log.info(" %s | config file loaded", file_name)
            except: continue

    def log_config(self):
        for entry in self.__dict__.items():
            log.info(entry)

#testing
def main():
    cm = configManager()

if __name__ == "__main__":
    main()
