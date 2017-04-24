import numpy as np
import pandas as pd

def create_stages(df, **f):

    def label_stage(row):
        if row[f["Closed"]] >= 1: return "Closed"
        elif row[f["Acquired"]] >= 1: return "Acquired"
        elif row[f["IPO"]] >= 1: return "IPO"
        elif row["keys_company_stage_series-d+"] >= 1: return "Series D+/PE"
        elif row[f["SeriesC"]] >= 1: return "Series C"
        elif row[f["SeriesB"]] >= 1: return "Series B"
        elif row[f["SeriesA"]] >= 1: return "Series A"
        elif row[f["Seed"]] >= 1: return "Seed"
        elif row["keys_company_stage_pre-seed"] >= 1: return "Pre-Seed"
        else: return "Other"

    def separate_ages(ages):
        from sklearn.mixture import GaussianMixture
        ages = ages.dropna()
        age_stacked = np.vstack(ages)
        mix = GaussianMixture(n_components=2)
        mix.fit(age_stacked)
        pred = mix.predict(age_stacked)
        max_index = list(mix.means_).index(max(mix.means_))
        if max_index == 1: pred = [0 if x==1 else 1 for x in pred]
        return pd.Series(pred, index=ages.index)

    def label_stage_other(row, pred):
        try: return "New" if pred[row.name] == 1 else "Other"
        except: return row["keys_company_stage"]

    df_new = df.copy()
    for x in f.values():
        if x not in df_new: df_new[x] = 0
    df_new["keys_company_stage_series-d+"] = df_new[[f["SeriesD"],f["SeriesE"],f["SeriesF"],f["SeriesG"],f["SeriesH"],f["PE"]]].sum(axis=1)
    df_new["keys_company_stage_pre-seed"] = df_new[[f["Convertible"], f["NonEquity"],f["ProductCF"],f["EquityCF"],f["Angel"],f["Grant"]]].sum(axis=1)
    df_new["keys_company_stage_other"] = df_new[[f["Debt"], f["Secondary"], f["Undisclosed"]]].sum(axis=1)
    df_new["keys_company_stage"] = df_new.apply(lambda row: label_stage(row), axis=1)
    pred_type = separate_ages(df_new[f["Age"]].loc[df_new["keys_company_stage"] == "Other"])
    df_new["keys_company_stage"] = df_new.apply(lambda row: label_stage_other(row, pred_type), axis=1)
    list_factors = dict(
        Included = ["New", "Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Series D+/PE", "Other"],
        Excluded = ["Closed", "Acquired", "IPO"])
    factors = {}
    for k,l in list_factors.items():
        for v in l: factors[v] = k
    df_new["keys_company_stage_group"] = df_new["keys_company_stage"].map(factors)
    ordinal_stages = {
        "New" : 0, "Pre-Seed" : 1, "Seed" : 2, "Series A" : 3, "Series B" : 4, "Series C" : 5, "Series D+/PE" : 6,
        "Other" : np.nan, "Closed" : -1, "IPO" : 7, "Acquired" : 8}
    df_new["keys_company_stage_number"] = df_new["keys_company_stage"].map(ordinal_stages)

    return df_new[["keys_company_stage_group", "keys_company_stage","keys_company_stage_number"]]