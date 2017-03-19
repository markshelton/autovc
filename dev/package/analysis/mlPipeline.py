#%% general setup
import pandas as pd
import sys; sys.path.append('C:/Users/mark/Documents/GitHub/honours/dev/package')
import logging; log = logging.getLogger(__name__)

#load data
output_table = "combo"
database_file = "analysis/output/combo.db"
import analysis.dataPreparer as dp
df = dp.export_dataframe(database_file, output_table)

#%%constrain data
df = df.loc[df['company_operating_bool'] == 1]
#df = df.loc[df['offices_headquarters_country_dummy_usa'] == 1]
#df = df.loc[df['company_founded_date'] > 1262275200]

#%%select features, labels
chosen = "outcome_exit_bool"
drops = [col for col in list(df) if col.startswith(("key","outcome","index"))]
drops.append(chosen)
X = df.drop(drops, axis=1)
y = df[chosen]

#%%train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#%%preamble
print('\n','-'*20)
print("Classification Results")
print("\n")
print("Features Date:", "December 2013")
print("Labels Date:", "September 2016")
print("Selected Label:", chosen)
print("Selected Classifier:", "Random Forest")

#%%Classification
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=10)
RF_fit = RF.fit(X_train, y_train)
RF_pred = RF_fit.predict(X_test)
RF_prob = RF_fit.predict_proba(X_test)

#%%#%% Feature importances
print('\n')
print('Feature Importances (>0.01):')
zipped = list(zip(list(X), RF_fit.feature_importances_))
zipped.sort(key = lambda t: t[1],reverse=True)
zipped = [(k,v) for (k,v) in zipped if v >= 0.01]
for i, j in zipped: print("{}: {:.4f}".format(i, j))

#%%Test results
from sklearn import metrics
mc_scorer = metrics.make_scorer(metrics.matthews_corrcoef)
classification_report = metrics.classification_report(y_test, RF_pred)
confusion_matrix = metrics.confusion_matrix(y_test, RF_pred)
test_accuracy = RF_fit.score(X_test, y_test)
test_matthews = metrics.matthews_corrcoef(y_test, RF_pred)
print('\nClassification Report:', classification_report)
print('Confusion Matrix:', confusion_matrix)
print('Test Accuracy:', test_accuracy)
print('Test Matthews corrcoef', test_matthews)

#%%Cross-fold validation
from sklearn.model_selection import KFold, cross_val_score
RF_scores = cross_val_score(RF, X, y, cv=5, scoring=mc_scorer)
print('\nCross-validation scores:', RF_scores)
print("CV Avg Matthews CC: %0.2f (+/- %0.2f)" % (RF_scores.mean(), RF_scores.std() * 2))
print('-'*20, '\n')

#%%Plot results
import matplotlib.pyplot as plt