
#initialise package
__all__ = ["logManager", "configManager", "dbLoader"]

#local modules
import logManager
import configManager
import dbLoader

#program
db = dbLoader.dbLoader()
