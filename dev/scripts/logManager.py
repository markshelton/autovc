#!/Anaconda3/env/honours python

"""logManager"""

#standard modules
import logging
import logging.config
import os

#third-party modules

#local modules
import configManager

#constants
LOG_CONFIG = "../config/logger.yaml"

#program
try: config = configManager.load_yaml(LOG_CONFIG)
except:
    logging.basicConfig(level=logging.INFO)
    log.warn("Default log config failed")
else:
    logging.config.dictConfig(config)
    log = logging.getLogger(__name__)
    log.info("Default log config loaded")
    log.info("Logger created")

