#!/Anaconda3/env/honours python

"""ml_pipeline"""

import sqlite3

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from log import Log

DATABASE_FILE = "../data/raw/2016-Sep-09_sqlite.db"

ATTRIBUTES = [
    "country_code", #string
    "state_code", #string
    "region", #string
    "city", #string
    "zipcode", #string
    "status", #string
    "funding_rounds", #integer
    "funding_total_usd", #integer
    "founded_on", #date
    "first_funding_on", #date
    "last_funding_on", #date
    "closed_on", #date
    "employee_count" #integer (categorical)
    ]

def get_data(database_file):
    con = sqlite3.connect(DATABASE_FILE)
    con.text_factory = str
    attributes = ", ".join(ATTRIBUTES)
    sql = "SELECT %s FROM test" % attributes
    data = pd.read_sql_query(sql, con)
    return data

def clean_data(data):
    data.replace("", "NaN", inplace=True)
    #print(data.describe())
    data = data.apply(LabelEncoder().fit_transform)
    data.replace("NaN", np.nan, inplace=True)
    print(data.describe())
    #data = data.apply(StandardScaler().fit_transform)

def visualise_data(data):
    sns.pairplot(data[:100], hue="acquired", size=1.5)

def split_data(data):
    labels = data[["acquired"]]
    features = data.drop("acquired", 1)
    return (features, labels)

def analyse(features, labels):
    #pipeline = Pipeline([('')])
    print(features.describe())
    print(labels.describe())
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.5)
    clf = SVC()
    #print (features_train.describe())
    #print(labels_train.describe())
    clf.fit(features_train, labels_train)
    score = clf.score(features_test, labels_test)
    return score

def main():
    log = Log(__file__).logger
    log.info("initiated process")
    data = get_data(DATABASE_FILE)
    log.info("collected data")
    clean_data(data)
    #visualise_data(data)
    #log.info("visualised data")
    #features, labels = split_data(data)
    #log.info("split data")
    #score = analyse(features, labels)
    #log.info("created score")
    #print(score)
    #log.info("finished process")

if __name__ == "__main__":
    main()
