#!/Anaconda3/env/honours python

"""configManager"""

#standard modules
import logging
import os
import datetime

#third-party modules
import yaml

#local modules
import logManager

#constants
CONFIG_DIR = "config/"
PRIV_FILE = "_"

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
                if not file.startswith(PRIV_FILE):
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

    def load_yaml(self, path):
        if os.path.exists(path):
            with open(path, 'rt') as f:
                output = yaml.safe_load(f.read())
                return output

    def read_config_file(self, file_name):

        def construct_paths(file_content):
            for attr in file_content["paths"]:
                new = [str(x) for x in file_content["paths"][attr]]
                file_content[attr] = "".join(new)
            del file_content["paths"]
            return file_content

        def update_timestamp(file_content, file_name):
            prev_date = file_content.get("date", None)
            prev_version = file_content.get("compile_version", 0)
            today = datetime.date.today()
            current_date = "{0000}-{00}-{00}".format(today.year, today.month, today.day)
            if prev_date == current_date:
                file_content["date"] = current_date
                file_content["compile_version"] = prev_version + 1
            with open(file_name, "w") as f:
                yaml.dump(file_content, f)
            return file_content

        file_content = self.load_yaml(file_name)
        try: file_content = construct_paths(file_content)
        except: pass
        try: file_content = update_timestamp(file_content, file_name)
        except: pass
        self.__dict__.update(file_content)

    def read_config_files(self, file_names):
        for file_name in file_names:
            try:
                self.read_config_file(file_name)
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
