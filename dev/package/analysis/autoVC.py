#native
import os
import sqlite3
from datetime import date, timedelta
from collections import defaultdict
import csv
import warnings
import logging

#third party
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint
from sklearn.externals import joblib
import numpy as np
import pandas as pd

#local
import sqlManager as sm
import analysis.dataPreparer as dp
from logManager import logged
import dbLoader as db

log = logging.getLogger(__name__)

warnings.simplefilter("ignore")

def generate_scorer(y_true, y_pred, **kwargs):
    roc = metrics.roc_auc_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    mc = metrics.matthews_corrcoef(y_true, y_pred)
    prc = metrics.average_precision_score(y_true, y_pred)
    log.info("roc: {:.3f} | prc: {:.3f} | f1: {:.3f} | mc: {:.3f} |".format(roc, prc, f1, mc))
    return f1

def reliability_curve(y_true, y_score, bins=10, normalize=False):

    if normalize: y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score, y_score <= threshold + bin_width / 2)
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
    return y_score_bin_mean, empirical_prob_pos

    plt.figure(0, figsize=(8, 8))
    plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect")
    for method, (y_score_bin_mean, empirical_prob_pos) in reliability_scores.items():
        scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        plt.plot(y_score_bin_mean[scores_not_nan],
                empirical_prob_pos[scores_not_nan], label=method)
    plt.ylabel("Empirical probability")
    plt.legend(loc=0)

    plt.subplot2grid((3, 1), (2, 0))
    for method, y_score_ in y_score.items():
        y_score_ = (y_score_ - y_score_.min()) / (y_score_.max() - y_score_.min())
        plt.hist(y_score_, range=(0, 1), bins=bins, label=method,
                histtype="step", lw=2)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.legend(loc='upper center', ncol=2)

class autoVC():

    @logged
    def __init__(self, output_folder, merge_config):
        self.output_folder = output_folder
        self.master_test_merge_config = merge_config

    @logged
    def load_master(self, file_path, file_date, feature_config, label_config, merge_config):
        self.master_path = file_path
        self.master_date = file_date
        self.master_feature_config = feature_config
        self.master_label_config = label_config
        self.master_merge_config = merge_config

    @logged
    def load_test(self, file_path, file_date, label_config):
        self.test_path = file_path
        self.test_date = file_date
        self.test_label_config = label_config

    @logged
    def _create_scorer(self):

            scorer = metrics.make_scorer(generate_scorer)
            return scorer

    @logged
    def _make_estimator(self, cv=3):
        def _create_pipe():

            #Create Pipeline
            pipe = Pipeline(steps=[
                ("imputer", Imputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("thresholder", VarianceThreshold()),
                ("extracter", None),
                ("clf", None)])

            #Hyperparameter tuning
            param_dist = [
                dict(
                    extracter = [PCA(), SelectKBest()],
                    clf = [GaussianNB()]),
                dict(
                    clf = [RandomForestClassifier(n_estimators=20, class_weight="balanced")],
                    clf__criterion = ["gini", "entropy"],
                    clf__max_depth = [5, 10, 20]),
                dict(
                    extracter = [PCA(), SelectKBest()],
                    clf = [LogisticRegression()],
                    clf__penalty = ["l1", "l2"],
                    clf__C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]),
                dict(
                    extracter = [PCA(), SelectKBest()],
                    clf = [KNeighborsClassifier()],
                    clf__weights = ['uniform','distance'],
                    clf__n_neighbors = [3, 5, 7, 10],),
                dict(
                    clf = [DecisionTreeClassifier()],
                    clf__criterion = ["gini", "entropy"],
                    clf__max_depth = [5, 10, 20],
                    clf__class_weight = [None, "balanced"]),
                dict(
                    extracter = [PCA(), SelectKBest()],
                    clf = [SVC()],
                    clf__kernel = ["linear", "rbf", "poly", "sigmoid"],
                    clf__C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]),
                dict(
                    extracter = [PCA(), SelectKBest()],
                    clf = [SGDClassifier()],
                    clf__penalty = ["l1", "l2"],
                    clf__alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]),
                dict(
                    extracter = [PCA(), SelectKBest()],
                    clf = [MLPClassifier()],
                    clf__alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000])
            ]

            return pipe, param_dist

        pipe, param_dist = _create_pipe()
        self.scorer = self._create_scorer() #metrics.make_scorer(metrics.roc_auc_score) #
        clf = GridSearchCV(
            estimator= pipe, cv=cv, param_grid=param_dist, verbose = 2,#param_distributions=param_dist, n_iter = 2,
            scoring = self.scorer, return_train_score=False, n_jobs=-1)
        return clf

    @logged
    def _get_slice(self, input_path, slice_date, slice_type):
        slice_date = slice_date.strftime("%Y-%m-%d")
        tables = sm.get_tables(input_path)
        input_abs_path = os.path.abspath(input_path)
        input_short_path = os.path.basename(input_path).split(".")[0]
        output_path = "{0}{1}/{2}_{3}.db".format(self.output_folder, input_short_path, slice_date, slice_type)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        db.clear_files(output_path)
        with sqlite3.connect(output_path) as conn:
            c = conn.cursor()
            c.execute("ATTACH DATABASE '{0}' AS src;".format(input_path))
            c.execute("DROP TABLE IF EXISTS SliceDate;")
            c.execute("CREATE TABLE SliceDate (Value DATETIME);")
            c.execute("INSERT INTO SliceDate (Value) VALUES ('{0}');".format(slice_date))
            for table in tables:
                c.execute("SELECT sql FROM src.sqlite_master WHERE type='table' AND name='{0}';".format(table))
                c.execute(c.fetchone()[0])
                try: c.execute("INSERT INTO main.{0} SELECT * FROM src.{0} WHERE created_at < '{1}';".format(table, slice_date))
                except:
                    try: c.execute("INSERT INTO main.{0} SELECT * FROM src.{0} WHERE started_on < '{1}';".format(table, slice_date))
                    except: c.execute("INSERT INTO main.{0} SELECT * FROM src.{0};".format(table, slice_date))
        return output_path

    @logged
    def _create_slices(self, master_path, master_date, n_slices, period):
        dates = []
        for i in range(n_slices):
            feature_date = master_date - timedelta(days=365*period*(i+1))
            label_date = feature_date + timedelta(days=365*period)
            dates.append((feature_date, label_date))
        slice_paths = []
        for feature_date, label_date in dates:
            feature_path = self._get_slice(master_path, feature_date, "feature")
            label_path = self._get_slice(master_path, label_date, "label")
            slice_paths.append((feature_path, label_path))
        return slice_paths

    @logged
    def _prepare_data(self, feature_path, feature_config, label_path, label_config, merge_config, label, nrows=None):
        label_folder = os.path.dirname(label_path)
        label_base = os.path.basename(label_path).split(".")[0]
        label_raw_path = "{0}/{1}_raw.csv".format(label_folder, label_base)
        label_clean_path = "{0}/{1}_clean.csv".format(label_folder, label_base)
        dp.flatten_file(label_path, label_config, label_raw_path, "label")
        dp.clean_file(label_raw_path, label_clean_path, nrows=nrows)

        feature_folder = os.path.dirname(feature_path)
        feature_base = os.path.basename(feature_path).split(".")[0]
        feature_raw_path = "{0}/{1}_raw.csv".format(feature_folder,feature_base)
        feature_clean_path = "{0}/{1}_clean.csv".format(feature_folder,feature_base)
        dp.flatten_file(feature_path, feature_config, feature_raw_path, "feature")
        dp.clean_file(feature_raw_path, feature_clean_path, nrows=nrows)

        output_path = "{0}{1}_{2}.db".format(self.output_folder, feature_base, label_base)
        db.clear_files(output_path)
        dp.load_file(output_path, feature_clean_path, "feature")
        dp.load_file(output_path, label_clean_path, "label")
        dp.merge(output_path, merge_config)

        export_path = output_path.split(".")[0]+".csv"
        db.export_file(output_path, export_path, "merge")
        df = dp.export_dataframe(output_path, "merge")
        X, y = dp.label_dataset(df, label)
        return X, y

    @logged
    def _store_results(self, clf, feature_path = "", label_path = "", output_path = "output.csv"):
        if feature_path == "" and label_path == "":
            export_path = self.output_folder + output_path
        else:
            feature_base = os.path.basename(feature_path).split(".")[0]
            label_base = os.path.basename(label_path).split(".")[0]
            export_path = "{0}{1}_{2}.db".format(self.output_folder, feature_base, label_base)
            export_path = export_path.replace(".db","_results.csv")
        with open(export_path, 'w+', newline='') as outfile:
            results = clf.cv_results_
            writer = csv.writer(outfile)
            writer.writerow(results.keys())
            writer.writerows(zip(*results.values()))

    @logged
    def fit(self, period, n_slices, label, cv=3, warm=False, max_obs = None, store_path=None):

        self.period = period
        self.n_slices = n_slices
        self.label = label

        if warm:
            self.estimator = joblib.load(store_path)
            return self.estimator

        assert self.master_path is not None
        db.clear_files(self.output_folder)

        estimators = []
        datasets = {}

        clf = self._make_estimator(cv=cv)
        slice_paths = self._create_slices(self.master_path, self.master_date, n_slices, period)
        for feature_path, label_path in slice_paths: #N: n_slices
            print("Feature Path: {0}, Label Path: {1}".format(feature_path, label_path))
            X, y = self._prepare_data(
                feature_path, self.master_feature_config,
                label_path, self.master_label_config,
                self.master_merge_config, label, nrows=max_obs)
            datasets[feature_path] = (X, y)
            clf.fit(X, y)
            self._store_results(clf, feature_path, label_path)
            bst = clf.best_estimator_
            estimators.append(bst)
        print("Best Estimators:", estimators)

        results = defaultdict(list)
        for estimator in estimators: #N: n_slices
            print("Estimator:", estimator)
            for feature_path, label_path in slice_paths: #N: n_slices
                print("Feature Path: {0}, Label Path: {1}".format(feature_path, label_path))
                X, y = datasets[feature_path]
                scores = cross_val_score(estimator, X, y, cv=cv, scoring=self.scorer) #N: cv
                score = scores.mean()
                results[estimator].append(score)
                print('CV Scores (AUC):', scores),
                print("CV Avg (AUC): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)),
            results[estimator] = np.mean(results[estimator])

        best_estimator = max(results.keys(), key=(lambda k: results[k]))
        print("Best Estimator:", best_estimator.get_params())
        self.estimator = best_estimator
        if store_path is not None: joblib.dump(self.estimator, store_path)
        return self.estimator

    @logged
    def score(self, test_date = None, cv=3, max_obs = None, scorer=None):
        assert self.test_path is not None
        assert self.estimator is not None

        if test_date is not None:
            assert test_date <= self.test_date
            assert test_date - timedelta(days=365*self.period) <= self.master_date
        else: test_date = self.test_date

        feature_path = self._get_slice(self.master_path, test_date - timedelta(days=365*self.period), "feature")
        label_path = self._get_slice(self.test_path, test_date, "label")
        X, y = self._prepare_data(
            feature_path, self.master_feature_config,
            label_path, self.master_label_config,
            self.master_test_merge_config, self.label, nrows=max_obs)

        if scorer is None: scorer = self._create_scorer()
        self.scorer = scorer

        self.scores = cross_val_score(self.estimator, X, y, cv=cv, scoring=scorer)
        print(self.scores)

        self.y_pred = cross_val_predict(self.estimator, X, y, cv=cv, method = 'predict')

        self.classification_report = metrics.classification_report(y, self.y_pred)
        print(self.classification_report)

        self.confusion_matrix = metrics.confusion_matrix(y, self.y_pred)
        print(self.confusion_matrix)

        self.features = list(X)
        self.y_test = list(y)
        self.y_pred = list(self.y_pred)

        print('Feature Importances (Top 30):')
        try:
            zipped = list(zip(list(X), self.estimator.steps[-1][1].feature_importances_))
            zipped.sort(key = lambda t: t[1],reverse=True)
            self.feature_importances = zipped
            for i, j in self.feature_importances:
                print("{}: {:.4f}".format(i, j))
        except: print("Failed.")

        return self.scores

    @logged
    def save(self, output_file):
        with open(output_file, "w+") as file:
            for entry in self.__dict__.items():
                file.write(str(entry) + "\n")

@logged
def main():
    periods = [0.5, 1, 2, 3]
    for i in periods:
        build = str(i)+"/" if type(periods) is list else ""
        vc = autoVC(
            output_folder = "analysis/output/autoVC/"+build,
            merge_config = "analysis/config/master_test_merge.sql")
        vc.load_master(
            file_path = "analysis/input/master.db",
            file_date = date(2016, 9, 9),
            feature_config = "analysis/config/master_feature.sql",
            label_config = "analysis/config/master_label.sql",
            merge_config = "analysis/config/master_merge.sql")
        vc.fit(
            period = i,
            n_slices = 3,
            label = "outcome_exit_bool",
            cv=3,
            warm = False,
            max_obs = 10000,
            store_path = vc.output_folder+"estimator.pkl")
        vc.load_test(
            file_path = "analysis/input/test.db",
            file_date = date(2017, 4, 4),
            label_config = "analysis/config/test_label.sql")
        vc.score(cv=3, max_obs = 10000)
        vc.save(output_file = vc.output_folder+'record.txt')

def test():
    vc = autoVC(
        output_folder = "analysis/output/autoVC/",
        merge_config = "analysis/config/master_test_merge.sql")
    est = vc._make_estimator()
    from sklearn import datasets
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    est.fit(X, y)
    vc._store_results(est)
    print(est.best_estimator_)

if __name__ == "__main__":
    test()