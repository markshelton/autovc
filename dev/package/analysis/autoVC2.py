#native
import os
import sqlite3
from datetime import date, timedelta
from collections import defaultdict
import csv
import warnings
import logging
import yaml
import pickle
from functools import wraps
import sys

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
from radar import random_date

#local
import sqlManager as sm
import analysis.dataPreparer as dp
from logManager import logged
import dbLoader as db
import analysis.getStages as gs

log = logging.getLogger(__name__)
#warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.nan)

master_log = pd.DataFrame()
flag_log_scores = False

features_stage_info = dict(
    Age = 'confidence_context_broader_company_age_number',
    FundingRounds = 'confidence_validation_funding_rounds_number',
    FundingRaised = 'confidence_validation_funding_raised_value_total_number',
    SeriesA = 'confidence_validation_funding_round_codes_list_a',
    SeriesB = 'confidence_validation_funding_round_codes_list_b',
    SeriesC = 'confidence_validation_funding_round_codes_list_c',
    SeriesD = 'confidence_validation_funding_round_codes_list_d',
    SeriesE = 'confidence_validation_funding_round_codes_list_e',
    SeriesF = 'confidence_validation_funding_round_codes_list_f',
    SeriesG = 'confidence_validation_funding_round_codes_list_g',
    SeriesH = 'confidence_validation_funding_round_codes_list_h',
    Closed = "keys_company_status_closed_bool",
    Acquired = "keys_company_status_acquired_bool",
    IPO = "keys_company_status_ipo_bool"
)

label_stage_info = dict(
    Age = 'outcome_age_number',
    FundingRounds = 'outcome_funding_rounds_number',
    FundingRaised = 'outcome_funding_raised_value_total_number',
    SeriesA = 'outcome_funding_round_codes_list_a',
    SeriesB = 'outcome_funding_round_codes_list_b',
    SeriesC = 'outcome_funding_round_codes_list_c',
    SeriesD = 'outcome_funding_round_codes_list_d',
    SeriesE = 'outcome_funding_round_codes_list_e',
    SeriesF = 'outcome_funding_round_codes_list_f',
    SeriesG = 'outcome_funding_round_codes_list_g',
    SeriesH = 'outcome_funding_round_codes_list_h',
    Closed = "outcome_closed_bool",
    Acquired = "outcome_acquired_bool",
    IPO = "outcome_ipo_bool"
)


class ConfigManager(object):

    @logged
    def __init__(self, config_path):
        file_content = self.load_yaml(config_path)
        self.__dict__.update(file_content)

    def load_yaml(self, path):

        def join(loader, node):
            seq = loader.construct_sequence(node)
            return ''.join([str(i) for i in seq])

        yaml.add_constructor('!join', join)
        if os.path.exists(path):
            with open(path, 'rt') as f:
                return yaml.load(f.read())

    def log_config(self):
        for entry in self.__dict__.items():
            log.info(entry)

@logged
def get_config(path):
    return ConfigManager(path)

@logged
def generate_dates(time_slices, forecast_windows, start_date, end_date, load_prev_files, output_slices_path):
    if load_prev_files and os.path.exists(output_slices_path):
        with open(output_slices_path, "rb") as prev_slices:
            dataset_slices = pickle.load(prev_slices)
        return dataset_slices
    else:
        forecast_windows = [timedelta(weeks=x*52) for x in forecast_windows]
        largest_window = max(forecast_windows)
        effective_end_date = end_date - largest_window
        feature_slices = [random_date(start_date, effective_end_date) for x in range(time_slices)]
        dataset_slices = []
        for feature_slice in feature_slices:
            for forecast_window in forecast_windows:
                label_slice = feature_slice + forecast_window
                dataset_slices.append((feature_slice, label_slice))
        os.makedirs(os.path.dirname(output_slices_path), exist_ok=True)
        with open(output_slices_path, "wb+") as save_path:
            pickle.dump(dataset_slices, save_path)
        return dataset_slices

@logged
def get_slice(input_path, output_folder, feature_date, label_date, slice_type):
    input_short_path = os.path.basename(input_path).split(".")[0]
    start_path = "{0}{1}/{2}".format(output_folder, input_short_path,feature_date)
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
                except:
                    try: c.execute("INSERT INTO main.{0} SELECT * FROM src.{0} WHERE patent_date < '{1}';".format(table, slice_date))
                    except: c.execute("INSERT INTO main.{0} SELECT * FROM src.{0};".format(table, slice_date))
    return output_path

@logged
def apply_constraints(df):
    df = df.loc[df['keys_company_stage_group'] == "Included"]
    age_old_cutoff = df["confidence_context_broader_company_age_number"][df["keys_company_stage"] == "Series D+"].quantile(0.75)
    df = df.loc[df['confidence_context_broader_company_age_number'] <= age_old_cutoff]
    return df

@logged
def create_stages(df, **features):
    df2 = df.copy()
    df2["keys_company_stage"] = "Other"
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2[features["Closed"]] >= 1), "Closed", df2["keys_company_stage"])
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2[features["Acquired"]] >= 1), "Acquired", df2["keys_company_stage"])
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2[features["IPO"]] >= 1), "IPO", df2["keys_company_stage"])
    df2["keys_company_stage_series-d+"] = df2[[features["SeriesD"],features["SeriesE"],features["SeriesF"],features["SeriesG"],features["SeriesH"]]].sum(axis=1)
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2["keys_company_stage_series-d+"] >= 1), "Series D+/PE", df2["keys_company_stage"])
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2[features["SeriesC"]] >= 1), "Series C", df2["keys_company_stage"])
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2[features["SeriesB"]] >= 1), "Series B", df2["keys_company_stage"])
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2[features["SeriesA"]] >= 1), "Series A", df2["keys_company_stage"])
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2[features["FundingRaised"]] >= 0), "Seed", df2["keys_company_stage"])
    age_new_cutoff = df2[features["Age"]][df2["keys_company_stage"] == "Seed"].quantile(0.75)
    df2["keys_company_stage"] = np.where((df2["keys_company_stage"] == "Other") & (df2[features["Age"]] <= age_new_cutoff), "Pre-Seed", df2["keys_company_stage"])
    group_stages = {"Other" : "Excluded", "Closed" : "Excluded", "IPO" : "Excluded", "Acquired" : "Excluded", "Pre-Seed" : "Included",
        "Seed" : "Included", "Series A" : "Included", "Series B" : "Included", "Series C" : "Included", "Series D+" : "Included"}
    df2["keys_company_stage_group"] = df2["keys_company_stage"].map(group_stages)
    ordinal_stages = {"Pre-Seed" : 1, "Seed" : 2, "Series A" : 3, "Series B" : 4, "Series C" : 5, "Series D+" : 6,"Other" : np.nan, "Closed" : -1, "IPO" : 7, "Acquired" : 8}
    df2["keys_company_stage_number"] = df2["keys_company_stage"].map(ordinal_stages)
    return df2[["keys_company_stage_group", "keys_company_stage","keys_company_stage_number"]]

@logged
def add_stages(df):
    feature_stages = create_stages(df, **features_stage_info)
    label_stages = create_stages(df, **label_stage_info)
    label_stages = label_stages.rename(columns={
        "keys_company_stage_group": "outcome_stage_group",
        "keys_company_stage":"outcome_stage",
        "keys_company_stage_number":"outcome_stage_number"})
    df = pd.concat([feature_stages, label_stages, df], axis=1)
    return df

@logged
def make_label(df):
    df["outcome_extra_stage_number"] = df["outcome_stage_number"] - df["keys_company_stage_number"]
    df["outcome_extra_stage_bool"] = np.where(df["outcome_extra_stage_number"] > 0, 1, 0)
    y = df["outcome_extra_stage_bool"]
    print("Feature:", df["keys_company_stage_number"].value_counts())
    print("Label:", df["outcome_stage_number"].value_counts())
    print("Merge:", df["outcome_extra_stage_bool"].value_counts())
    return y

@logged
def filter_features(df):
    df = df.select_dtypes(['number'])
    drops = [col for col in list(df) if col.startswith(("key","from","outcome","index"))]
    X = df.drop(drops, axis=1)
    all_nan = {x:{np.nan:0} for x in X.columns[X.isnull().all()].tolist()}
    X = X.replace(all_nan)
    return X

@logged
def finalise_dataset(df):
    df = add_stages(df)
    df = apply_constraints(df)
    y = make_label(df)
    keys = df[[col for col in list(df) if col.startswith(("key", "outcome"))]]
    X = filter_features(df)
    X = X.sort_index(axis=1)
    return X, y, keys

@logged
def generate_dataset(feature_date, label_date, feature_input_path, label_input_path, feature_config, label_config,
        merge_config, output_folder, max_observations, load_prev_files):
    #create_slices
    feature_path = get_slice(feature_input_path, output_folder, feature_date, label_date, slice_type = "feature")
    label_path = get_slice(label_input_path, output_folder, feature_date, label_date, slice_type = "label")

    #flatten_slices
    feature_raw_path = feature_path.replace(".db", "_raw.csv")
    if not load_prev_files or not os.path.isfile(feature_raw_path): dp.flatten_file(feature_path, feature_config, feature_raw_path, "feature")
    label_raw_path = label_path.replace(".db", "_raw.csv")
    if not load_prev_files or not os.path.isfile(label_raw_path): dp.flatten_file(label_path, label_config, label_raw_path, "label")

    #clean_slices
    feature_clean_path = feature_path.replace(".db", "_clean.csv")
    if not load_prev_files or not os.path.isfile(feature_clean_path):dp.clean_file(feature_raw_path, feature_clean_path, nrows=max_observations)
    label_clean_path = label_path.replace(".db", "_clean.csv")
    if not load_prev_files or not os.path.isfile(label_clean_path): dp.clean_file(label_raw_path, label_clean_path, nrows=max_observations)

    #merge_slices
    output_path = label_path.replace("label.db", "merge.db")
    if not load_prev_files or not os.path.isfile(output_path):
        dp.load_file(output_path, feature_clean_path, "feature")
        dp.load_file(output_path, label_clean_path, "label")
        dp.merge(output_path, merge_config)

    #export_merged
    export_path = output_path.split(".")[0]+".csv"
    if not load_prev_files or not os.path.isfile(export_path): db.export_file(output_path, export_path, "merge")

    #finalise_merged
    df = dp.export_dataframe(output_path, "merge")
    X, y, keys = finalise_dataset(df)
    return X, y, keys

@logged
def create_params(algorithm, pp_params, clf_params):
    clf_param = clf_params[algorithm]
    params = dict(**pp_params, **clf_param)
    return params

def log_scores(f):
    def log_and_call(*args, **kwargs):
        global flag_log_scores
        flag_log_scores = True
        f_result = f(*args, **kwargs)
        return f_result
    return log_and_call

def get_weights(clf):
    if type(clf) in [RandomForestClassifier, DecisionTreeClassifier]: return clf.feature_importances_
    elif type(clf) in [LogisticRegression, LinearSVC]: return clf.coef_[0]
    else: return None

def scorer(estimator, X, y_true):
    global master_log
    y_pred = estimator.predict(X)
    f1 = metrics.f1_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    ck = metrics.cohen_kappa_score(y_true, y_pred)
    clf_type = type(estimator.named_steps["clf"]).__name__.split(".")[-1]
    log.info("f1: {0:.3f} | auc: {1:.3f} | ck: {2:.3f} | clf: {3}".format(f1, auc, ck, clf_type))
    if flag_log_scores:
        y_score = estimator.predict_proba(X)
        mc = metrics.matthews_corrcoef(y_true, y_pred)
        prc = metrics.average_precision_score(y_true, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score[:,1])
        auc_scorer = metrics.make_scorer(metrics.roc_auc_score)
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y_true, cv= 3, scoring = auc_scorer, train_sizes=np.linspace(0.1, 1.0, 10))
        clf = estimator.named_steps["clf"]
        weights = get_weights(clf)
        params = estimator.get_params()
        df = pd.DataFrame({"Classifier": [clf],"Params": [params], "Weights": [weights],
            "AUC": [auc], "PRC": [prc], "F1": [f1], "MC": [mc],
            "ROC_FPR": [fpr], "ROC_TPR": [tpr], "ROC_Thresholds": [thresholds],
            "Train_Sizes": [train_sizes], "Train_Scores": [train_scores], "Test_Scores": [test_scores]})
        if master_log.empty: master_log = df
        else: master_log = pd.concat([master_log, df], axis=0)
    return auc

@log_scores
def logged_fit(clf, *args, **kwargs):
    return clf.fit(*args, **kwargs)

@logged
def fit_score_model(X, y, params, pipe_steps, cv_folds, search_iterations, verbosity, n_jobs, log_scores):
    pipe = Pipeline(steps=pipe_steps)
    clf = RandomizedSearchCV(
        estimator= pipe, param_distributions=params, scoring = scorer, n_iter=search_iterations, cv=cv_folds,
        verbose = verbosity, return_train_score=False, n_jobs=n_jobs, pre_dispatch="2*n_jobs")
    if log_scores: logged_fit(clf, X, y)
    else: clf.fit(X, y)
    temp = pd.DataFrame(clf.cv_results_)
    results = pd.DataFrame(columns=list(temp))
    for row in temp.iterrows():
        new = pd.concat([row[1].to_frame().T] * cv_folds, axis=0, ignore_index=True)
        results = pd.concat([results, new], axis=0, ignore_index=True)
    return results

@logged
def store_log(feature_slice, label_slice, X, y, keys, cm, output_log, results=None):
    global master_log
    if master_log.empty: return False
    master_log["feature_slice"] = feature_slice
    master_log["label_slice"] = label_slice
    master_log["feature_names"] = [list(X)] * len(master_log.index)
    master_log["label_name"] = [list(y)] * len(master_log.index)
    master_log["feature_stage"] = [keys["keys_company_stage"]] * len(master_log.index)
    master_log["feature_stage_number"] = [keys["keys_company_stage_number"]]  * len(master_log.index)
    master_log["label_stage"] = [keys["outcome_stage"]]  * len(master_log.index)
    master_log["label_stage_number"] = [keys["outcome_stage_number"]]  * len(master_log.index)
    for k, v in cm.__dict__.items():
        try: master_log[k] = v
        except: pass
    master_log.reset_index(inplace=True)
    if results is not None: master_log = pd.concat([master_log, results], axis=1)
    if os.path.exists(output_log):
        old_master_log = pd.read_pickle(output_log)
        master_log = pd.concat([master_log, old_master_log], axis=0)
    master_log.to_pickle(output_log)
    master_log = pd.DataFrame()
    return True

@logged
def create_pipelines():
    dataset_slices = generate_dates(
        cm.time_slices_create, cm.forecast_windows, cm.master_start_date,
        cm.master_end_date, cm.load_prev_files_create, cm.output_slices_path_create)
    for feature_date, label_date in dataset_slices:
        X, y, keys = generate_dataset(
            feature_date, label_date, cm.master_path, cm.master_path, cm.master_feature_config,
            cm.master_label_config, cm.master_merge_config, cm.output_folder_create,
            cm.max_observations_create, cm.load_prev_files_create)
        for algorithm in cm.algorithms:
            params = create_params(algorithm, cm.pp_params, cm.clf_params)
            results = fit_score_model(
                X, y, params, cm.pipe_steps, cm.cv_folds_create, cm.search_iterations,
                cm.verbosity, cm.n_jobs, cm.log_scores)
            store_log(feature_date, label_date, X, y, keys, cm, cm.output_log_create, results)
    pipelines = pd.read_pickle(cm.output_log_create)
    return pipelines

def rank_pipelines(pipelines, number_selected, criteria):
    top_pipelines = pipelines.sort_values(by=criteria, ascending=False)[0:number_selected]
    return top_pipelines

def get_best_pipeline(pipelines, criteria):
    pipelines["Params_str"] = pipelines["Params"].astype(str)
    best_params_str = pipelines.groupby("Params_str")[criteria].mean().sort_values(ascending=False).index[0]
    best_pipeline = pipelines.loc[pipelines["Params_str"] == best_params_str].reset_index().ix[0]
    return best_pipeline


@log_scores
def logged_cv_score(*args, **kwargs):
    return cross_val_score(*args, **kwargs)

@logged
def select_pipeline(pipelines = None):
    if pipelines is None: pipelines = pd.read_pickle(cm.output_log_create)
    finalist_pipelines = rank_pipelines(pipelines, cm.top_pipelines_select, cm.pipeline_criteria_select)
    dataset_slices = generate_dates(
        cm.time_slices_select, cm.forecast_windows, cm.master_start_date,
        cm.master_end_date, cm.load_prev_files_select, cm.output_slices_path_select)
    for feature_date, label_date in dataset_slices:
        X, y, keys = generate_dataset(
            feature_date, label_date, cm.master_path, cm.master_path, cm.master_feature_config,
            cm.master_label_config, cm.master_merge_config, cm.output_folder_select,
            cm.max_observations_select, cm.load_prev_files_select)
        for index, pipeline in finalist_pipelines.iterrows():
            pipe = Pipeline(steps=cm.pipe_steps).set_params(**pipeline["Params"])
            if cm.log_scores: logged_cv_score(pipe, X, y, scoring=scorer, cv=cm.cv_folds_select, verbose=cm.verbosity, n_jobs=cm.n_jobs)
            else: cross_val_score(pipe, X, y, scoring=scorer, cv=cm.cv_folds_select, verbose=cm.verbosity, n_jobs=cm.n_jobs)
            store_log(feature_date, label_date, X, y, keys, cm, cm.output_log_select)
    finalist_pipelines = pd.read_pickle(cm.output_log_select)
    best_pipeline = get_best_pipeline(finalist_pipelines, cm.pipeline_criteria_evaluate)
    return best_pipeline

@logged
def evaluate_pipeline(best_pipeline = None):
    if best_pipeline is None:
        finalist_pipelines = pd.read_pickle(cm.output_log_select)
        best_pipeline = get_best_pipeline(finalist_pipelines, cm.pipeline_criteria_evaluate)
    train_slices = generate_dates(
        cm.time_slices_evaluate, cm.forecast_windows, cm.master_start_date,
        cm.master_end_date, cm.load_prev_files_evaluate, cm.output_slices_path_evaluate)
    for feature_date_train, label_date_train in train_slices:
        X_train, y_train, keys = generate_dataset(
            feature_date_train, label_date_train,
            cm.master_path, cm.master_path,
            cm.master_feature_config, cm.master_label_config, cm.master_merge_config,
            cm.output_folder_evaluate, cm.max_observations_evaluate, cm.load_prev_files_evaluate)
        pipe = Pipeline(steps=cm.pipe_steps).set_params(**best_pipeline["Params"])
        if log_scores: logged_fit(pipe, X_train, y_train)
        else: pipe.fit(X_train, y_train)
        X_test, y_test, keys = generate_dataset(
            cm.test_date - (label_date_train-feature_date_train) , cm.test_date,
            cm.master_path, cm.test_path,
            cm.master_feature_config, cm.test_label_config, cm.final_merge_config,
            cm.output_folder_evaluate, cm.max_observations_evaluate, cm.load_prev_files_evaluate)
        scorer(pipe, X_test, y_test)
        store_log(feature_date_train, label_date_train, X_test, y_test, keys, cm, cm.output_log_evaluate)
    results = pd.read_pickle(cm.output_log_evaluate)
    return results

def main():
    global cm
    cm = get_config("analysis/config/experiments/{0}.yaml".format(sys.argv[1]))
    pipelines = create_pipelines() if cm.create_pipelines_flag else None
    best_pipeline = select_pipeline(pipelines) if cm.select_pipeline_flag else None
    results = evaluate_pipeline(best_pipeline) if cm.evaluate_pipeline_flag else None

if __name__ == "__main__":
    main()