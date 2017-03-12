
import re
import sys
import codecs

def mysql_to_sqlite(mysql_db, sqlite_db):
    with codecs.open(mysql_db,encoding="latin-1") as mysql_file:
        content = mysql_file.read()

    def insertvals_replacer(match):
        insert, values = match.groups()
        replacement = []
        for vals in INSERTVALS_SPLIT_RE.split(values):
            replacement.append( '%s (%s);' % (insert, vals) )
        return '\n'.join( replacement )

    COMMAND_RE = re.compile(r'^(SET).*?;\n$', re.I | re.M | re.S)
    content = COMMAND_RE.sub('', content)

    TCONS_RE = re.compile(r'\)(\s*(CHARSET|DEFAULT|ENGINE)(=.*?)?\s*)+;', re.I | re.M | re.S)
    content = TCONS_RE.sub(');', content)
    content = content.replace( r"\'", "''" )

    INSERTVALS_RE = re.compile(r'^(INSERT INTO.*?VALUES)\s*\((.*?)\);$', re.M | re.S)
    INSERTVALS_SPLIT_RE = re.compile(r'\)\s*,\s*\(', re.I | re.M | re.S)
    content = INSERTVALS_RE.sub(insertvals_replacer, content)

    REMOVE_LOCK_RE = re.compile(r'^(LOCK).+', re.M)
    content = REMOVE_LOCK_RE.sub('BEGIN TRANSACTION',content)

    REMOVE_UNLOCK_RE = re.compile(r'^(UNLOCK).+', re.M)
    content = REMOVE_UNLOCK_RE.sub('COMMIT',content)

    REMOVE_KEYS_RE = re.compile(r'^\s*(PRIMARY KEY|FOREIGN KEY|UNIQUE KEY|KEY).+', re.M)
    content = REMOVE_KEYS_RE.sub('', content)

    FIX_COMMAS_RE = re.compile(r',\s*\)', re.M)
    content = FIX_COMMAS_RE.sub(')',content)

    FIX_CB_PREFIX_RE = re.compile(r'\"cb_', re.M)
    content = FIX_CB_PREFIX_RE.sub("\"", content)

    FIX_ESCAPES_RE = re.compile(r"(?<!\\)\\''", re.M)
    content = FIX_ESCAPES_RE.sub("'", content)

    with open(sqlite_db,"w+",encoding="latin-1") as sqlite_file:
        sqlite_file.write(content)

def convert_db(source_dir,destination_dir, source_type="mysql", destination_type="postgresql"):
    os.makedirs(destination_dir, exist_ok=True)
    for source_file in db.get_files(source_dir):
        source_short = os.path.basename(source_file)
        destination_file = destination_dir+source_short
        log.info("{0} | SQL Conversion Started".format(source_short))
        try: mysql_to_sqlite(source_file,destination_file)
        except: log.error("{0} | SQL Conversion Failed".format(source_short))
        else: log.info("{0} | SQL Conversion Successful".format(source_short))