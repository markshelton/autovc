from sqlalchemy import create_engine
import sqlalchemy_utils
from sqlalchemy_utils import database_exists, create_database
from odo import odo

INPUT_FILES = ["./data/acquisitions.csv"]
DATABASE = "postgresql://postgres@localhost/test"

engine = create_engine(DATABASE)

if not database_exists(engine.url):
    create_database(engine.url)

print(database_exists(engine.url))

