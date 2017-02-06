import sys
sys.path.insert(0, "/dev/scripts/package")
__all__ = ['configManager', 'dbLoader', 'logManager', 'sqlManager']

from source_sql.main import main; main()
