#!/Anaconda3/env/honours python

"""controller"""

# standard modules

# third party modules

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

#local modules

from model_comparer import ModelComparer
from data_collector import DataCollector
from log import Log

#Log

LOG_FILE = __file__

#DataCollector

DATABASE_FILE = "../data/raw/2016-Sep-09_sqlite.db"
ATTRIBUTES = {}
ATTRIBUTES["country_code"] = "string"
ATTRIBUTES["state_code"] = "string"
ATTRIBUTES["region"] = "string"
ATTRIBUTES["city"] = "string"
ATTRIBUTES["zipcode"] = "string"
ATTRIBUTES["employee_count"] = "string"
ATTRIBUTES["status"] = "blah"
ATTRIBUTES["funding_rounds"] = "int"
ATTRIBUTES["funding_total_usd"] = "int"
ATTRIBUTES["founded_on"] = "date"
ATTRIBUTES["first_funding_on"] = "date"
ATTRIBUTES["last_funding_on"] = "date"
ATTRIBUTES["closed_on"] = "date"

#ModelComparer

SCORING = 'roc_auc'
MODELS = []
MODELS.append(('LDA', LinearDiscriminantAnalysis()))
MODELS.append(('LR', LogisticRegression(class_weight="balanced")))
MODELS.append(('NB', GaussianNB()))
MODELS.append(('RF', RandomForestClassifier(class_weight="balanced")))
MODELS.append(('KNN', KNeighborsClassifier()))
MODELS.append(('CART', DecisionTreeClassifier(class_weight="balanced")))

#Controller

def main():
    log = Log(LOG_FILE).logger
    log.info("initiated process")
    dc = DataCollector(DATABASE_FILE,ATTRIBUTES, log=log)
    mc = ModelComparer(
        dc.features, dc.labels,
        MODELS, SCORING,
        plot=True, log=log)
    log.info("finished process")

if __name__ == "__main__":
    main()
