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
        file_names = self.search_directory(config_dir)
        for file_name in file_names:
            try: self.read_config(file_name)
            except: continue
        self.log_config()

    def search_directory(self, config_dir):
        file_names = []
        for file in os.listdir(config_dir):
            file_names.append(config_dir+file)
        return file_names

    def read_config(self, file_name, attributes=[]):
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(file_name)
        for section in parser.sections():
            self.__dict__.update(parser.items(section))
            if section == "config":
                for (key, value) in parser.items(section):
                    attributes.append(key)
                    self.read_config(value, attributes)

    def log_config(self):
        log.info("Configuration loaded successfully")
        for entry in self.__dict__.items():
            log.debug(entry)

#testing
def main():
    cm = configManager()

if __name__ == "__main__":
    main()
