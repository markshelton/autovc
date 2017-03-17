#standard modules
import logging

#third party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#--utility
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

#--preprocessing
from sklearn.preprocessing import StandardScaler

#--feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

#--model selection
from sklearn.model_selection import KFold, cross_val_score

#--classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

#local modules
from logManager import logged
import analysis.dataPreparer as dp
import dbLoader as db

#constants
files = {}
files["thirteen"] = {}
files["thirteen"]["database_file"] = "collection/thirteen/output/2013-Dec.db"
files["thirteen"]["flat_raw_file"] = "analysis/output/flatten/raw/thirteen.csv"
files["thirteen"]["flat_clean_file"] = "analysis/output/flatten/clean/thirteen.csv"
files["thirteen"]["flatten_config"] = "analysis/config/flatten/thirteen.sql"
files["sixteen"] = {}
files["sixteen"]["database_file"] = "collection/sixteen/output/2016-Sep.db"
files["sixteen"]["flat_raw_file"] = "analysis/output/flatten/raw/sixteen.csv"
files["sixteen"]["flat_clean_file"] = "analysis/output/flatten/clean/sixteen.csv"
files["sixteen"]["flatten_config"] = "analysis/config/flatten/sixteen.sql"

output_table = "combo"
merge_config = "analysis/config/flatten/merge.sql"
database_file = "analysis/output/combo.db"

chosen = "outcome_operating_bool"

#logger
log = logging.getLogger(__name__)

@logged
def prepare_data(database_file, merge_config, output_table, files):
    #db.clear_files(database_file)
    #for file_name, file in files.items():
        #dp.flatten_file(file["database_file"], file["flatten_config"], file["flat_raw_file"], file_name)
        #dp.clean_file(file["flat_raw_file"], file["flat_clean_file"])
        #dp.load_file(database_file, file["flat_clean_file"], file_name)
    #dp.merge(database_file, merge_config)
    return dp.export_dataframe(database_file, output_table)

@logged
def alter_chosen(df):
    df = df.loc[df['company_operating_bool'] == 1]
    return df

@logged
def split_data(df, chosen):
    Y = df[chosen]
    drop_columns = [col for col in list(df) if col.startswith(("key","outcome","index"))]
    drop_columns.append(chosen)
    X = df.drop(drop_columns, axis=1)
    return (X, Y)

def main():
    #load data
    df = prepare_data(database_file, merge_config, output_table, files)
    df = alter_chosen(df)
    X, y = split_data(df, chosen)

    from sklearn import metrics
    mc_scorer = metrics.make_scorer(metrics.matthews_corrcoef)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    #preamble

    print('\n')
    print('-'*20)
    print("Classification Results")
    print("\n")

    print("Features Date:", "December 2013")
    print("Labels Date:", "September 2016")
    print("Selected Label:", chosen)
    print("Selected Classifier:", "Random Forest")

    #classification

    from sklearn.ensemble import RandomForestClassifier

    RF = RandomForestClassifier(n_estimators=10)
    RF_fit = RF.fit(X_train, y_train)
    RF_pred = RF_fit.predict(X_test)
    RF_prob = RF_fit.predict_proba(X_test)

    ## Feature importances are helpful
    print('\n')
    print('Feature Importances (>0.01):')
    zipped = list(zip(list(X), RF_fit.feature_importances_))
    zipped.sort(key = lambda t: t[1],reverse=True)
    zipped = [(k,v) for (k,v) in zipped if v >= 0.01]
    for i, j in zipped: print("{}: {:.4f}".format(i, j))

    print("\n")

    print('Classification Report:')
    print(metrics.classification_report(y_test, RF_pred))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, RF_pred))
    print('Test Accuracy:', RF_fit.score(X_test, y_test))
    print('Test Matthews corrcoef', metrics.matthews_corrcoef(y_test, RF_pred))

    RF_scores = cross_val_score(RF, X, y, cv=5, scoring=mc_scorer)
    print('\nCross-validation scores:', RF_scores)
    print("CV Avg Matthews CC: %0.2f (+/- %0.2f)" % (RF_scores.mean(), RF_scores.std() * 2))
    print('-'*20)
    print('\n')

if __name__ == "__main__":
    main()