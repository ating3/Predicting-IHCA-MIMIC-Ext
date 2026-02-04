import preprocessing_query
import os
import pandas as pd
from icdmappings import Validator
from icdmappings import Mapper
import polars as pl

icu_dir = os.getenv("ICU_PATH")
hosp_dir = os.getenv("HOSP_PATH")

#TIME INDEPENDENT
procedure_path = os.path.join(icu_dir, "procedureevents.csv")
#has procedure codes

icustays_path = os.path.join(icu_dir, "icustays.csv")
#has ICU duration and admission type

admissions_path = os.path.join(hosp_dir, "admissions.csv")
#has ethnicity

patients_path = os.path.join(hosp_dir, "patients.csv")
#has age(anchor_age) and sex

diagnosis_path = os.path.join(hosp_dir, "diagnoses_icd.csv")
#presenting illness

#TIME DEPENDENT
chart_events_path = os.path.join(icu_dir, "chartevents.csv")
vitals=['HR', 'RR', 'SpO2', 'SBP', 'DBP', 'MAP']
#charttime, item_id mapping(temp: 223761, HR: 220045, RR: 220210, SpO2: 220277, SBP: 220179, DBP: 220180, Mean Arterial Pressure: 220181)

procedure_df = pd.read_csv(procedure_path)
icustays_df = pd.read_csv(icustays_path)
patients_df = pd.read_csv(patients_path)
admissions_df = pd.read_csv(admissions_path)
diagnosis_df = pd.read_csv(diagnosis_path)

procedure_df["starttime"] = pd.to_datetime(procedure_df["starttime"])
icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])
icustays_df["outtime"] = pd.to_datetime(icustays_df["outtime"])


ihca_list = preprocessing_query.create_ihca_list(procedure_df, icustays_df)
controls_list = preprocessing_query.create_controls_list(icustays_df, ihca_list)
mapper = Mapper()

controls_metadata = preprocessing_query.create_metadata_table(controls_list, icustays_df, patients_df, admissions_df, diagnosis_df, mapper)
#controls_metadata.to_csv("controls_metadata.csv", index=False)
ihca_metadata = preprocessing_query.create_metadata_table(ihca_list, icustays_df, patients_df, admissions_df, diagnosis_df, mapper)
#ihca_metadata.to_csv("ihca_metadata.csv", index=False)

def create_vitals_table(input_list, label, hours_before=13):
    #creates time dependent data tables
    #label: classification of patient string "ctrl":0 vs "ihca":1
    #input_list: ihca_list or controls_list above
    #hours_before: how many hours before cardiac arrest event to start tracking
    processed_vitals = {}
    for vital_name in vitals:
        vital_df = preprocessing_query.query_vitals_vectorized(input_list, vital_name, chart_events_path)
        feature_df = preprocessing_query.feature_extract_vital(input_list, vital_df, vital_name, label, hours_before)
        feature_df = (
            feature_df
            .unique(subset=["subject_id", "stay_id", "bin_hour"])  # Keep only unique rows
        )
        processed_vitals[vital_name] = feature_df
    combined_df = processed_vitals["HR"].rename({"valuenum": "HR"})
    for vital in vitals:
        df = processed_vitals[vital].rename({"valuenum": vital})
        # Only select the join keys and the vital column (not bin_time or other columns)
        combined_df = combined_df.join(
            df.select(["subject_id", "stay_id", "bin_hour", vital]),
            on=["subject_id", "stay_id", "bin_hour"],
            how="full",
            coalesce=True
        )

    combined_df = combined_df.select(["subject_id", "stay_id", "bin_hour", "bin_time", 'HR', 'RR', 'SpO2', 'SBP', 'DBP', 'MAP']).sort("subject_id")
    for vital in vitals:
        combined_df = preprocessing_query.impute_patient_average_column(combined_df, vital)
    combined_df = combined_df.with_columns(pl.lit(1 if label == "ihca" else 0).alias("label")) 
    return combined_df



ihca_output = create_vitals_table(ihca_list,"ihca")
ctrl_output = create_vitals_table(controls_list,"ctrl")
combined_df = pl.concat(
    [ihca_output, ctrl_output]
)

metadata = pl.from_pandas(pd.concat([controls_metadata, ihca_metadata]))
final_df = combined_df.join(metadata, how="left", on="subject_id")

final_df.write_csv("output.csv")
