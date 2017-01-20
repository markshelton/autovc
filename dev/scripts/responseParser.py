from json import dumps, loads
import logging


import dpath.util as dp
import dpath.exceptions

import logManager
import configManager


API_TABLE_LOOKUP = {
    "acquisitions":"acquisitions",
    "funding_rounds":"funding-rounds",
    "funds":"funds",
    "ipos":"ipos",
    "organizations":"organizations",
    "people":"people",
    "products":"products"
}


log = logging.getLogger(__name__)


def get_table(json):
    table = json["properties"]["api_path"].split("/")[0]
    table_lookup = {v: k for k, v in API_TABLE_LOOKUP.items()}
    table = table_lookup[table]
    return table

def safe(glob):
    glob = [x for x in glob if not isinstance(x, int)]
    return glob

def get_value(dictionary, glob):
    try: return dp.get(dictionary, glob)
    except KeyError: return None

#GOOD
def split_key(keychain, reference):
    #kc = [relationships, news.items]
    key = ''.join(keychain[-1:])
    #key = news.items
    ref_value = get_value(reference, safe(keychain))
    #ref_value = {uuid: [blah], properties: blah}
    dp.delete(reference, safe(keychain))
    keychain.pop() # kc=[relationships]
    keychain.extend(key.split(".",maxsplit=1))
    #kc=[relationships, news, items]
    dp.new(reference, keychain, ref_value)

#GOOD
def parse_split(keychain, visited, reference, response, records):
    print("SPLIT")
    #kc = [relationships, news.items]
    split_key(keychain, reference)
    visited.append(list(keychain[:-1]))
    #kc=[relationships, news, items]
    if keychain not in visited:
        parse(keychain, visited, reference, response, records)
        visited.append(list(keychain))
    keychain.pop()
    #kc = [relationships, news]

#GOOD
def parse_items(keychain, visited, reference, response, records):
    print("ITEMS")
    #kc = [relationships, news, items]
    res_value = get_value(response, keychain)
    #res_value = [{type: News, uuid: blah ...},{},{}]
    if res_value: #True
        #print("ITEMS:", res_value)
        for i, res_item in enumerate(res_value):
            #i = 1, res_item = {type: News, uuid: blah}
            keychain.append(i)
            #kc=[relationships, news, items, 1]
            parse(keychain, visited, reference, response, records)
            keychain.pop()

#GOOD
def parse_dict(keychain, visited, reference, response, records):
    print("DICT")
    #kc=[relationships, news, items, 1, properties]
    ref_value = get_value(reference, safe(keychain))
    #ref_value = {title: blah, author: Iris Dorbian, ...}
    if type(ref_value) is dict: #True
        ref_keys = ref_value.keys()
        #ref_keys = [title, author, posted_on etc.]
    else: ref_keys = reference.keys()
    for ref_key in ref_keys:
        #ref_key = author
        keychain.append(ref_key)
        #kc = [relationships, news, items, 1, properties, author]
        if keychain not in visited: #False
            parse(keychain, visited, reference, response, records)
            visited.append(list(keychain))
        keychain.pop()
        #kc = [relationships, news, items, 1, properties]

#GOOD
def unpack(keychain, reference, response):
    #kc=[relationships, news, items, 1, properties]
    res_keys = get_value(response, keychain).keys()
    #res_keys = [title, author, posted_on, ...]
    table = get_value(reference, safe(keychain)).split(".")[0]
    #table = news
    dp.delete(reference, safe(keychain))
    for res_key in res_keys:
        #res_key = author
        keychain.append(res_key)
        #kc = [relationships, news, items, 1, properties, author]
        value = "{0}.{1}".format(table, res_key)
        #value = news.author
        dp.new(reference, safe(keychain), value)
        keychain.pop()
        #kc = [relationships, news, items, 1, properties]

#GOOD
def parse_properties(keychain, visited, reference, response, records):
    print("PROPERTIES")
    if keychain not in visited:
        unpack(keychain, reference, response)
        parse(keychain, visited, reference, response, records)
        visited.append(list(keychain))

#GOOD
def get_ref(keychain, reference, records):
    #kc = [relationships, news, items, 1, properties, author]
    ref_inits = get_value(reference, safe(keychain))
    #print("REF:", ref_inits, type(ref_inits))
    if type(ref_inits) is str: ref_inits = [ref_inits]
    ref_values = []
    for ref in ref_inits:
        #ref = [news, author]
        ref = ref.split(".")
        table, attribute = tuple(ref) #table = news, attribute = author
        if get_value(records, table): #True
            length = len(get_value(records, table)) -1 #0
        else: length = 0
        ref = [table, length, attribute] #ref = [news, 1, author]
        if get_value(records, ref): #False
            ref = [table, length+1, attribute]
        ref_values.append(ref)
    return ref_values

#GOOD
def store(keychain, reference, response, records):
    #kc = [relationships, news, items, 1, properties, author]
    ref_values = get_ref(keychain, reference, records)
    for ref in ref_values:
        #ref = [news, 1, author]
        res = get_value(response, keychain)
        if res is None: res = "Null"
        # res = "Iris Dorbian"
        dp.new(records, ref, res)
    return records

def parse(keychain, visited, reference, response, records):
    #kc=[relationships, news, items, 0]
    try: key = str(keychain[-1])
    except: key = ""
    ref_value = get_value(reference, safe(keychain))
    #res_value = get_value(response, keychain)
    #print(key, keychain, type(ref_value))
    #print(key, keychain, ref_value, res_value, sep=" | ")
    if "." in key:
        parse_split(keychain, visited, reference, response, records)
    elif key == "items":
        parse_items(keychain, visited, reference, response, records)
    elif key == "properties" and type(ref_value) is not dict: #kc=[properties]
        parse_properties(keychain, visited, reference, response, records)
    elif type(ref_value) is dict or ref_value is None: #kc=[relationships]
        parse_dict(keychain, visited, reference, response, records)
    else:
        records = store(keychain, reference, response, records)

def play(reference, response):
    records = {}
    keychain = []
    visited = []
    parse(keychain, visited, reference, response, records)
    return records

def parse_json(reference, response):
    response = response["data"]
    table = get_table(response)
    reference = configManager.load_yaml(reference)[table]
    records = play(reference, response)
    print(dumps(records, indent=1))
    return records

if __name__ == "__main__":
    ref_path = "../config/crawler.yaml"
    json_path = "../../sources/api_examples/products.json"
    json_content = loads(open(json_path).read())
    parse_json(ref_path, json_content)
