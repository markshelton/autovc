
import re
import sys
import codecs

def mysql_to_postgresql(mysql_db, postgresql_db):
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

    with open(postgresql_db,"w+",encoding="latin-1") as postgresql_file:
        postgresql_file.write(content)