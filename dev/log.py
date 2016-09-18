#!/Anaconda3/env/honours python

"""log"""

import logging

class Log:

    def __init__(self, script_file):
        file_name = script_file.split(".")[0] + ".log"
        logging.basicConfig(
            filemode='w+',
            filename=file_name,
            level=logging.DEBUG,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%m-%d %H:%M:%S')

        console = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s | %(message)s")
        console.setFormatter(formatter)

        self.logger = logging.getLogger(file_name)
        console.setLevel(logging.INFO)
        self.logger.addHandler(console)
