import analysis.dataPreparer as dp
import dbLoader as db
from logManager import logged
import numpy as np
import pandas as pd
import requests
import json
import os
from unidecode import unidecode

@logged
def go_patents(df, df_old):

    def get_patents(company_name, index):
        end_date = "2013-01-01"
        q = '{"_and":[{"_begins":{"assignee_organization":"%s"}},{"_lte":{"patent_date":"%s"}}]}' % (company_name, end_date)
        f = '["assignee_first_seen_date","patent_num_combined_citations","patent_num_cited_by_us_patents","patent_type","patent_date"]'
        base = "http://www.patentsview.org/api/assignees/query?"
        path = unidecode("{}q={}&f={}".format(base, q, f))
        response = requests.get(path).json()
        if response["total_assignee_count"] == 0: return pd.DataFrame()
        patents_ll = [v["patents"] for v in response["assignees"]]
        patents = [item for sublist in patents_ll for item in sublist]
        series = pd.Series(dict(
            count_number = len(patents),
            first_date = min([v["assignee_first_seen_date"] for v in response["assignees"]]),
            citations_total_number = sum([int(v["patent_num_combined_citations"]) for v in patents]),
            citations_average_number = np.mean([int(v["patent_num_combined_citations"]) for v in patents]),
            cited_by_total_number = sum([int(v["patent_num_cited_by_us_patents"]) for v in patents]),
            cited_by_average_number = np.mean([int(v["patent_num_cited_by_us_patents"]) for v in patents]),
            type_list = ";".join([v["patent_type"] for v in patents])))
        series.name = index
        df = series.to_frame().T
        new_names = [(i,"potential_structural_patents_"+i) for i in list(df)]
        df.rename(columns = dict(new_names), inplace=True)
        return df

    names = ["count_number", "first_date", "citations_total_number","citations_average_number","cited_by_total_number","cited_by_average_number","type_list"]
    new_names = ["potential_structural_patents_"+i for i in names]
    new = pd.DataFrame(columns = new_names)
    for counter, (index, series) in enumerate(df.iterrows()):
        try:
            print(counter, series["keys_name_id"])
            if series["keys_permalink_id"] in list(df_old.index): temp = df_old.loc[[series["keys_permalink_id"]]]
            else: temp = get_patents(series["keys_name_id"], series["keys_permalink_id"])
            if temp.empty: temp = pd.DataFrame(columns=new_names, index=[series["keys_permalink_id"]])
            new = new.append(temp)
            if counter % 500 == 5:
                os.makedirs(os.path.dirname(output_file),exist_ok=True)
                new.to_csv(output_file, mode="w+", index=True)
                db.clear_files(output_database_file)
                dp.load_file(output_database_file, output_file, output_table, index=True)
        except: print("Error")
    return new

source_database_file = "analysis/output/combo.db"
source_table = "combo"
output_file = "analysis/output/extra/patents.csv"
output_database_file = "analysis/output/extra.db"
output_table = "patents"

def main():
    if os.path.exists(output_database_file):
        df_old = dp.export_dataframe(output_database_file, output_table, index=True)
    else: df_old = pd.DataFrame()
    df = dp.export_dataframe(source_database_file, source_table)
    df = go_patents(df, df_old)
    db.clear_files(output_file)
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    df.to_csv(output_file, mode="w+", index=True)
    db.clear_files(output_database_file)
    dp.load_file(output_database_file, output_file, output_table, index=True)

if __name__ == "__main__":
    main()