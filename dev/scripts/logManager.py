#!/Anaconda3/env/honours python

"""logManager"""

#standard modules
import logging
import logging.config
import os

#third-party modules
import yaml

#local modules

#constants
LOG_CONFIG = "../config/logger.yaml"

#program
warn = False
if os.path.exists(LOG_CONFIG):
    with open(LOG_CONFIG, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)
    warn = True
log = logging.getLogger(__name__)
if warn: log.warn("Default log config failed")
else: log.info("Default log config loaded")
log.info("Logger created")

