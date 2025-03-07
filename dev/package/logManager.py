#!/Anaconda3/env/honours python

"""logManager"""

#standard modules
import logging
import logging.config
import inspect
import time
from functools import wraps
import os
import sys

#third-party modules
import yaml

#local modules

#constants
LOG_CONFIG = "C:/Users/mark/Documents/GitHub/honours/dev/package/config/_logger.yaml"

def load_yaml(path):
    if os.path.exists(path):
        with open(path, 'rt') as f:
            output = yaml.safe_load(f.read())
            if output is None: raise(IOError)
            else: return output

try: config = load_yaml(LOG_CONFIG)
except:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.warn("Default log config failed")
else:
    logging.config.dictConfig(config)
    log = logging.getLogger(__name__)
    log.debug("Default log config loaded")
    log.debug("Logger created")

def traced(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        result = f(*args, **kwds)
        trace = inspect.currentframe()
        frames = inspect.getouterframes(inspect.currentframe())
        trace.extend([getattr(frame,"function") for frame in frames if getattr(frame,"function") is not "wrapper"])
        log.debug("Trace: %s" % (trace))
        return result
    return wrapper

def logged(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        log.info("{0} | Started".format(f.__name__))
        start_time = time.time()
        try: result = f(*args, **kwargs)
        except:
            elapsed_time = time.time() - start_time
            log.error("{0} | Failed | {1:.2f}".format(f.__name__,elapsed_time), exc_info=True)
        else:
            elapsed_time = time.time() - start_time
            log.info("{0} | Passed | {1:.2f}".format(f.__name__,elapsed_time))
            return result
    return wrapper