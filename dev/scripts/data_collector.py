#!/Anaconda3/env/honours python

"""data_collector"""

#standard modules

import sqlite3

#third party modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#local modules

class DataCollector(object):

    def __init__(self, database_file, attributes, log=None):
        self.database_file = database_file
        self.attributes = attributes
        self.log = log
        temp = self.get_data(database_file, attributes)
        self.data = self.clean_data(temp, attributes)
        self.features, self.labels = self.split_data(self.data)

    def get_data(self, database_file, attributes):
        self.log.info("data collection starting")
        try:
            con = sqlite3.connect(database_file)
            con.text_factory = str
            sel_attributes = ", ".join(attributes)
            sql = """
                    SELECT %s
                    FROM organizations
                    WHERE funding_rounds > 0
                    AND country_code = "USA"
                """ % sel_attributes
            data = pd.read_sql_query(sql, con)
        except:
            self.log.error("data collection failed", exc_info=1)
        self.log.info("data collection successful")
        return data

    def clean_data(self, data, attributes):
        self.log.info("data cleaning starting")
        try:
            temp = data["status"].str.get_dummies()
            temp.columns = ['is_' + col for col in temp.columns]
            data = pd.concat([data,temp],axis=1)
            le_columns = {}
            for column in data:
                if attributes.get(column) == "string":
                    le = LabelEncoder()
                    #data[column].replace("", "NaN", inplace=True)
                    data[column] = le.fit_transform(data[column])
                    le_columns[column] = le
                elif attributes.get(column) == "int":
                    data[column] = pd.to_numeric(
                        data[column],
                        errors="ignore")
                elif attributes.get(column) == "date":
                    data[column] = pd.to_datetime(
                        data[column],
                        format="%Y-%m-%d",
                        errors="coerce")
                    data["today"] = "2016-09-09" #NOTE: FIX THIS LATER
                    data["today"] = pd.to_datetime(
                        data["today"],
                        format="%Y-%m-%d",
                        errors="coerce")
                    data[column+"_days"] = data["today"] - data[column]
                    data[column+"_days"] = data[column+"_days"].apply(lambda x: x / np.timedelta64(1,'D'))
                    data = data.drop(column, axis = 1)
            data = data.drop(["today","status","is_operating","is_closed","is_ipo","country_code"], axis=1)
            data.replace("", np.nan, inplace=True)
            for column in data:
                data[column].fillna(data[column].mean(), inplace=True)
        except:
            self.log.error("data cleaning failed", exc_info=1)
        self.log.info("data cleaning successful")
        return data

    def split_data(self, data):
        self.log.info("data splitting started")
        try:
            labels = data[["is_acquired"]]
            features = data.drop("is_acquired", 1)
        except:
            self.log.error("data splitting failed")
        self.log.info("data splitting successful")
        return (features, labels)
