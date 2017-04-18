import numpy as np
import pandas as pd

def create_stages(df):

    def label_stage(row):
        if row["keys_company_status_closed_bool"] >= 1: return "Closed"
        elif row["keys_company_status_acquired_bool"] >= 1: return "Acquired"
        elif row["keys_company_status_ipo_bool"] >= 1: return "IPO"
        elif row["keys_company_stage_series-d+"] >= 1: return "Series D+/PE"
        elif row["confidence_validation_funding_round_codes_list_c"] >= 1: return "Series C"
        elif row["confidence_validation_funding_round_codes_list_b"] >= 1: return "Series B"
        elif row["confidence_validation_funding_round_codes_list_a"] >= 1: return "Series A"
        elif row["confidence_validation_funding_round_types_list_seed"] >= 1: return "Seed"
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

    df["keys_company_stage_series-d+"] = df[[
        'confidence_validation_funding_round_codes_list_d',
        'confidence_validation_funding_round_codes_list_e',
        'confidence_validation_funding_round_codes_list_f',
        'confidence_validation_funding_round_codes_list_g',
        'confidence_validation_funding_round_codes_list_h',
        'confidence_validation_funding_round_types_list_private_equity']].sum(axis=1)
    df["keys_company_stage_pre-seed"] = df[[
        'confidence_validation_funding_round_types_list_convertible_note',
        'confidence_validation_funding_round_types_list_non_equity_assistance',
        'confidence_validation_funding_round_types_list_product_crowdfunding',
        'confidence_validation_funding_round_types_list_equity_crowdfunding',
        'confidence_validation_funding_round_types_list_angel',
        'confidence_validation_funding_round_types_list_grant']].sum(axis=1)
    df["keys_company_stage_other"] = df[[
        'confidence_validation_funding_round_types_list_debt_financing',
        'confidence_validation_funding_round_types_list_secondary_market',
        'confidence_validation_funding_round_types_list_undisclosed']].sum(axis=1)
    df["keys_company_stage"] = df.apply(lambda row: label_stage(row), axis=1)
    pred_type = separate_ages(df["confidence_context_broader_company_age_number"].loc[df["keys_company_stage"] == "Other"])
    df["keys_company_stage"] = df.apply(lambda row: label_stage_other(row, pred_type), axis=1)
    list_factors = dict(
        Included = ["New", "Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Series D+/PE", "Other"],
        Excluded = ["Closed", "Acquired", "IPO"])
    factors = {}
    for k,l in list_factors.items():
        for v in l: factors[v] = k
    df["keys_company_stage_group"] = df["keys_company_stage"].map(factors)

    return df[["keys_company_stage_group", "keys_company_stage"]]