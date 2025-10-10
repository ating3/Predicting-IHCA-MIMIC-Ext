import query
import os
import pandas as pd
from icdmappings import Validator
from icdmappings import Mapper

icu_dir = "/home/anson/URC-mimic-project/physionet.org/files/mimic-iv/3.1/icu/"
hosp_dir = "/home/anson/URC-mimic-project/physionet.org/files/mimic-iv/3.1/hosp"

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
#charttime, item_id mapping(temp: 223761, HR: 220045, RR: 220210, SpO2: 220277, SBP: 220179, DBP: 220180, Mean Arterial Pressure: 220181)

procedure_df = pd.read_csv(procedure_path)
icustays_df = pd.read_csv(icustays_path)
patients_df = pd.read_csv(patients_path)
admissions_df = pd.read_csv(admissions_path)
diagnosis_df = pd.read_csv(diagnosis_path)

procedure_df["starttime"] = pd.to_datetime(procedure_df["starttime"])
icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])
icustays_df["outtime"] = pd.to_datetime(icustays_df["outtime"])


ihca_list = query.create_ihca_list(procedure_df, icustays_df)
controls_list = query.create_controls_list(icustays_df, ihca_list)
mapper = Mapper()

controls_metadata = query.create_metadata_table(controls_list, icustays_df, patients_df, admissions_df, diagnosis_df, mapper)
#print(controls_metadata)
ihca_metadata = query.create_metadata_table(ihca_list, icustays_df, patients_df, admissions_df, diagnosis_df, mapper)
#print(ihca_metadata)