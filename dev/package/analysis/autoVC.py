#native
import os
import sqlite3
from datetime import date, timedelta
from collections import defaultdict
import csv
import warnings
import logging
from functools import wraps


#third party
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, cross_val_predict, learning_curve, ParameterGrid
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, FunctionTransformer, MinMaxScaler, Binarizer
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

import scipy
import scipy.stats as st
import numpy as np
import pandas as pd
from dateutil import relativedelta


#local
import sqlManager as sm
import analysis.dataPreparer as dp
from logManager import logged
import dbLoader as db
import analysis.getStages as gs

log = logging.getLogger(__name__)
#warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.nan)

store_scores = None

def log_scores(f):
    def log_and_call(*args, **kwargs):
        global store_scores
        store_scores = "scores.csv"
        print(store_scores)
        f_result = f(*args, **kwargs)
        #store_scores = ""
        return f_result
    return log_and_call

def get_feature_weights(clf):
    if isinstance(clf, DecisionTreeClassifier): return clf.feature_importances_
    elif isinstance(clf, RandomForestClassifier): return clf.feature_importances_
    elif isinstance(clf, LogisticRegression): return clf.coef_
    elif isinstance(clf, SGDClassifier): return clf.coef_
    elif isinstance(clf, MLPClassifier): return clf.coefs_

def generate_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    f1 = metrics.f1_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    ck = metrics.cohen_kappa_score(y_true, y_pred)
    clf_type = type(estimator.named_steps["clf"]).__name__.split(".")[-1]
    log.info("f1: {0:.3f} | auc: {1:.3f} | ck: {2:.3f} | clf: {3}".format(f1, auc, ck, clf_type))
    if store_scores is not None:
        y_score = estimator.predict_proba(X)
        mc = metrics.matthews_corrcoef(y_true, y_pred)
        prc = metrics.average_precision_score(y_true, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score[:,1])
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y_true)
        clf = estimator.named_steps["clf"]
        params = estimator.get_params()
        weights = get_feature_weights(clf)
        df = pd.DataFrame({"Classifier": [clf],"Params": [params], "Weights": [weights],
            "AUC": [auc], "PRC": [prc], "F1": [f1], "MC": [mc],
            "ROC_FPR": [fpr], "ROC_TPR": [tpr], "ROC_Thresholds": [thresholds],
            "Train_Sizes": [train_sizes], "Train_Scores": [train_scores], "Test_Scores": [test_scores]})
        df.to_csv(store_scores, index=False, mode='a+', quoting = csv.QUOTE_ALL, header=(not os.path.exists(store_scores)))
    return f1

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
        self.results_paths = []

    @logged
    def load_test(self, file_path, file_date, label_config):
        self.test_path = file_path
        self.test_date = file_date
        self.test_label_config = label_config

    @logged
    def _make_estimator(self, cv=3, algorithm="NB", n_iter=10):

        @logged
        def _create_pipe():

            pipe = Pipeline(steps=[
                ("imputer", Imputer(strategy="median")),
                ('transformer', FunctionTransformer(np.log1p)),
                ('scaler', StandardScaler()),
                ('extractor', PCA()),
                ("clf", None)])

            return pipe

        @logged
        def _create_params(algorithm="NB"):

            ext = dict(extractor__n_components=st.randint(5, 100))
            pp = dict(**imp, **tf, **sc, **ext)
            param_dists = dict(
                NB = dict(**pp,
                    clf = [GaussianNB()]),
                RF = dict(**pp,
                    clf = [RandomForestClassifier(class_weight="balanced")],
                    clf__n_estimators = st.randint(10, 100),
                    clf__criterion = ["gini", "entropy"],
                    clf__max_depth = st.randint(5, 20)),
                LR = dict(**pp,
                    clf = [LogisticRegression(class_weight="balanced")],
                    clf__penalty = ["l1", "l2"],
                    clf__C = np.logspace(-4, 1, 6)),
                KNN = dict(**pp,
                    clf = [KNeighborsClassifier()],
                    clf__weights = ['uniform','distance'],
                    clf__n_neighbors = st.randint(5, 20)),
                DT = dict(**pp,
                    clf = [DecisionTreeClassifier(class_weight="balanced")],
                    clf__criterion = ["gini", "entropy"],
                    clf__max_depth = st.randint(5, 20)),
                SVM = dict(**pp,
                    clf = [SVC(probability=True, class_weight="balanced")],
                    clf__C = np.logspace(-4, 1, 6)),
                ANN = dict(**pp,
                    clf = [MLPClassifier()],
                    clf__alpha = np.logspace(-4, 1, 6)))
            params = param_dists[algorithm]
            return params

        pipe = _create_pipe()
        params = _create_params(algorithm)
        clf = RandomizedSearchCV(
            estimator= pipe, cv=cv, param_distributions=params, verbose = 1, n_iter=n_iter,
            scoring = generate_scorer, return_train_score=False, n_jobs=-1, pre_dispatch="2*n_jobs")
        return clf

    @logged
    def _get_slice(self, input_path, feature_date, label_date, slice_type):
        input_short_path = os.path.basename(input_path).split(".")[0]
        start_path = "{0}{1}/{2}".format(self.output_folder, input_short_path,feature_date)
        if slice_type == "feature":
            slice_date = feature_date
            output_path = "{0}/{1}.db".format(start_path, slice_type)
        elif slice_type == "label":
            slice_date = label_date
            output_path = "{0}/{1}/{2}.db".format(start_path, label_date, slice_type)
        else: raise ValueError("Wrong slice_type passed.")
        slice_date = slice_date.strftime("%Y-%m-%d")
        tables = sm.get_tables(input_path)
        input_abs_path = os.path.abspath(input_path)
        if os.path.isfile(output_path): return output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
            feature_path = self._get_slice(master_path, feature_date, label_date, slice_type ="feature")
            label_path = self._get_slice(master_path, feature_date, label_date, slice_type = "label")
            slice_paths.append((feature_path, label_path))
        return slice_paths

    def label_dataset(self, df, label):
        df = df.select_dtypes(['number'])
        df = df.loc[df['keys_company_status_operating_bool'] == 1]
        df = df.loc[df['confidence_context_broader_company_age_number'] <= 15]
        df = df.loc[df["confidence_validation_funding_rounds_number"] > 0]
        if label == "outcome_extra_funding_rounds_bool":
            df["outcome_extra_funding_rounds_number"] = df["outcome_funding_rounds_number"] - df["confidence_validation_funding_rounds_number"]
            df = df.loc[df['outcome_extra_funding_rounds_number'] >= 0]
            df["outcome_extra_funding_rounds_bool"] = np.where(df["outcome_extra_funding_rounds_number"] == 0, 0, 1)
        y = df[label]
        drops = [col for col in list(df) if col.startswith(("key","from","outcome","index"))]
        keys = [col for col in list(df) if col.startswith("key")]
        X = df.drop(drops, axis=1)
        #X = scipy.sparse.csr_matrix(X.values)
        return X, y, df[keys]

    @logged
    def _prepare_data(self, feature_path, feature_config, label_path, label_config, merge_config, label, nrows=None, override=False):
        feature_raw_path = feature_path.replace(".db", "_raw.csv")
        feature_clean_path = feature_path.replace(".db", "_clean.csv")
        if not override and not os.path.isfile(feature_raw_path): dp.flatten_file(feature_path, feature_config, feature_raw_path, "feature")
        if not override and not os.path.isfile(feature_clean_path):dp.clean_file(feature_raw_path, feature_clean_path, nrows=nrows)

        label_raw_path = label_path.replace(".db", "_raw.csv")
        label_clean_path = label_path.replace(".db", "_clean.csv")
        if not override and not os.path.isfile(label_raw_path): dp.flatten_file(label_path, label_config, label_raw_path, "label")
        if not override and not os.path.isfile(label_clean_path): dp.clean_file(label_raw_path, label_clean_path, nrows=nrows)

        output_path = label_path.replace("label.db", "merge.db")

        if not override and not os.path.isfile(output_path):
            dp.load_file(output_path, feature_clean_path, "feature")
            dp.load_file(output_path, label_clean_path, "label")
            dp.merge(output_path, merge_config)

        export_path = output_path.split(".")[0]+".csv"
        if not override and not os.path.isfile(export_path): db.export_file(output_path, export_path, "merge")
        df = dp.export_dataframe(output_path, "merge")
        return df

    @logged
    def _store_results(self, clf, label_path, algo):
        bst_score = "{:.3f}".format(clf.best_score_)
        label_folder = os.path.dirname(label_path)
        results_folder = "{0}/results/".format(label_folder)
        self.results_paths.append(results_folder)
        os.makedirs(os.path.dirname(results_folder), exist_ok=True)
        output_path = "{0}{1}_{2}.csv".format(results_folder, bst_score, algo)
        with open(output_path, 'w+', newline='') as outfile:
            results = clf.cv_results_
            writer = csv.writer(outfile)
            writer.writerow(results.keys())
            writer.writerows(zip(*results.values()))

    @logged
    def fit_all(self, slice_paths, label, max_obs, cv, n_iter):
        estimators, datasets = [], {} # <n_slices * n_algorithms>
        for feature_path, label_path in slice_paths: #N: n_slices
            log.info("Feature Path: {0}, Label Path: {1}".format(feature_path, label_path))
            df = self._prepare_data(
                feature_path, self.master_feature_config,
                label_path, self.master_label_config,
                self.master_merge_config, label, nrows=max_obs)
            X, y, keys = self.label_dataset(df, label)
            datasets[feature_path] = (X, y)
            for algo in self.algorithms:
                clf = self._make_estimator(algorithm=algo, n_iter=n_iter, cv=cv)
                clf.fit(X, y)
                bst_est = clf.best_estimator_
                try: print(bst_est.named_steps["clf"].feature_importances_)
                except: pass
                estimators.append(bst_est)
                self._store_results(clf, label_path, algo)
        return estimators, datasets

    @logged
    def get_datasets(self, max_obs_all, max_obs_best, feature_path, label_path, label):
        X, y = self.datasets[feature_path]
        if (max_obs_best is None and max_obs_all is not None) or max_obs_best > max_obs_all:
            df = self._prepare_data(
                feature_path, self.master_feature_config,
                label_path, self.master_label_config,
                self.master_merge_config, label = label, nrows=max_obs_best, override=True)
            X, y, keys = self.label_dataset(df, label)
        elif max_obs_best is not None:
            X, y = self.datasets[feature_path]
            df = pd.concat([X, y], axis=1)
            df = df.sample(max_obs_best)
            X, y = df[list(X)], df[label]
        return X, y

    @logged
    def fit_best(self, estimators, slice_paths, max_obs_all, max_obs_best, cv, label):

        @log_scores
        def log_cross_val_score(*args, **kwargs):
            return cross_val_score(*args, **kwargs)

        results = defaultdict(list)
        for estimator in estimators: #N: n_slices
            for feature_path, label_path in slice_paths: #N: n_slices
                log.info("Feature Path: {0}, Label Path: {1}".format(feature_path, label_path))
                X, y = self.get_datasets(max_obs_all, max_obs_best, feature_path, label_path, label)
                scores = log_cross_val_score(estimator, X, y, cv=cv, scoring=generate_scorer, n_jobs=1) #N: cv
                results[estimator].append(scores.mean())
                log.info('CV Scores: {0}'.format(scores))
                log.info("CV Avg: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            results[estimator] = np.mean(results[estimator])
        best_estimator = max(results.keys(), key=(lambda k: results[k]))
        log.info("Best Estimator:", best_estimator.get_params())
        return best_estimator

    @logged
    def fit(self, period, n_slices, label, cv=3, warm=False, max_obs_all = None, max_obs_best = None, n_iter = 10, store_path=None,algorithms=["NB"]):
        self.period = period
        self.n_slices = n_slices
        self.label = label
        self.algorithms = algorithms
        if warm:
            self.estimator = joblib.load(store_path)
            return self.estimator
        assert self.master_path is not None
        slice_paths = self._create_slices(self.master_path, self.master_date, n_slices, period)
        self.estimators, self.datasets = self.fit_all(slice_paths, label, max_obs_all, cv, n_iter)
        self.estimator = self.fit_best(self.estimators, slice_paths, max_obs_all, max_obs_best, cv, label)
        if store_path is not None:
            joblib.dump(self.estimator, store_path)
        return self.estimator

    @logged
    def score(self, test_date = None, cv=3, max_obs = None, scorer=None):

        assert self.test_path is not None
        assert self.estimator is not None
        if test_date is not None:
            assert test_date <= self.test_date
            assert test_date - timedelta(days=365*self.period) <= self.master_date
        else: test_date = self.test_date
        feature_date = test_date - timedelta(days=365*self.period)
        label_date = test_date
        feature_path = self._get_slice(self.master_path, feature_date, label_date, slice_type= "feature")
        label_path = self._get_slice(self.test_path, feature_date, label_date, slice_type= "label")
        df = self._prepare_data(
            feature_path, self.master_feature_config,
            label_path, self.test_label_config,
            self.master_test_merge_config, self.label, nrows=max_obs)
        X, y, keys = self.label_dataset(df, label)
        if scorer is None: scorer = _create_scorer()
        self.scores = cross_val_score(self.estimator, X, y, cv=cv, scoring=generate_scorer, n_jobs=-1, pre_dispatch="1*n_jobs")
        print(self.scores)
        self.y_pred = cross_val_predict(self.estimator, X, y, cv=cv, method = 'predict', n_jobs=-1, pre_dispatch="1*n_jobs")
        self.classification_report = metrics.classification_report(y, self.y_pred)
        print(self.classification_report)
        self.confusion_matrix = metrics.confusion_matrix(y, self.y_pred)
        print(self.confusion_matrix)
        self.features = list(X)
        self.y_test = list(y)
        self.y_pred = list(self.y_pred)
        print('Feature Importances (Top 30):') #FIX
        try:
            zipped = list(zip(list(X), self.estimator.steps[-1][1].feature_importances_))
            zipped.sort(key = lambda t: t[1],reverse=True)
            self.feature_importances = zipped
            for i, j in self.feature_importances:
                print("{}: {:.4f}".format(i, j))
        except: print("Failed.")
        return self.scores

    @logged
    def save(self, output_config, output_results):
        with open(output_config, "w+") as c_file:
            for entry in self.__dict__.items():
                c_file.write(str(entry) + "\n")
        df_full = pd.DataFrame()
        all_paths = []
        for path in self.results_paths:
            if os.path.isdir(path): all_paths.extend(db.get_files(path))
        for path in all_paths:
            df = pd.read_csv(path)
            df["feature_date"] = path.split("/")[5]
            df["label_date"] = path.split("/")[6]
            df_full = pd.concat([df_full, df], axis=0)
        df_full.to_csv(output_results)


@logged
def main():
    vc = autoVC(
        output_folder = "analysis/output/autoVC/{0}/".format(build),
        merge_config = "analysis/config/master_test_merge.sql")
    vc.load_master(
        file_path = "analysis/input/master.db",
        file_date = date(2016, 9, 9),
        feature_config = "analysis/config/master_feature.sql",
        label_config = "analysis/config/master_label.sql",
        merge_config = "analysis/config/master_merge.sql")
    vc.fit(
        warm = False,
        period = i, n_slices = 2, cv=3, n_iter = 10, max_obs_all = 50000, max_obs_best = None,
        label = "outcome_exit_bool",
        algorithms = ["NB", "DT", "LR", "KNN", "RF", "SVM", "ANN"],
        store_path = vc.output_folder+"estimator.pkl")
    if final_score:
        vc.load_test(
            file_path = "analysis/input/test.db",
            file_date = date(2017, 4, 4),
            label_config = "analysis/config/test_label.sql")
        vc.score(cv=3, max_obs = 50000)
        vc.save(
            output_config = vc.output_folder+str(i)+'_record.txt',
            output_results = vc.output_folder + str(i) +'_results.csv')


def generate_slices(time_slices, forecast_window, start_date, end_date):
    #TODO
    return dataset_slices

def generate_dataset(feature_slice, label_slice, max_observations, load_prev_files,
    database_path, feature_config, label_config, merge_config, output_folder):
    #TODO
    return X,y

def create_params(algorithm, pp_params, clf_params):
    #TODO
    return params

def fit_model(X, y, params, pipe, cv_folds, search_iterations, verbosity, log_scores):
    #TODO
    return log

def store_log(log, feature_slice, label_slice, cm, output_folder):
    #TODO
    return log

def new():
    cm = ConfigManager("analysis/config/experiments/5.yaml")
    dataset_slices = generate_slices(cm.time_slices, cm.forecast_window, cm.master_start_date, cm.master_end_date)
    for feature_slice, label_slice in dataset_slices:
        X,y = generate_dataset(feature_slice, label_slice, cm.max_observations, cm.load_prev_files, cm.master_path,
            cm.master_feature_config, cm.master_label_config, cm.master_merge_config, cm.output_folder)
        for algorithm in cm.algorithms:
            params = create_params(algorithm, cm.pp_params, cm.clf_params)
            log = fit_model(X, y, params, cm.pipe, cm.cv_folds, cm.search_iterations, cm.verbosity, cm.log_scores)
            store_log(log, feature_slice, label_slice, cm, cm.output_folder)

if __name__ == "__main__":
    main()