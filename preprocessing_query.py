import pandas as pd
import os
import datetime as dt
from icdmappings import Validator
from icdmappings import Mapper
import polars as pl

vitals_code = {
    #"temp": 223761,
    "HR": 220045,
    "RR": 220210,  # Respiratory Rate
    "SpO2": 220277,
    "SBP": 220179,
    "DBP": 220180,
    "MAP": 220181,  # Mean Arterial Pressure
}

def create_ihca_list(procedure_df, icustays_df):
    #grabs patients with cardiac resuscitation procedure and relevant icu stay, filters for los > 1.0 and 
    #subject_id: patient id
    #reference: beginning of cardiac resuscitation event
    #stay_id: relevant icu stay of ihca event

    #get time of resuccitation event, this will be referencetime
    ihca_events = procedure_df[procedure_df["itemid"] == 225466][["subject_id", "starttime"]].copy()
    merged = ihca_events.merge(icustays_df[['subject_id', 'stay_id', 'intime', 'outtime', 'los', 'first_careunit']], on='subject_id', how='left')
    in_stay = (merged['starttime'] >= merged['intime']) & (merged['starttime'] <= merged['outtime']) & (merged['los'] >= 1.0)
    merged = merged[in_stay]
    merged = merged.rename(columns={'starttime': 'referencetime'})
    merged["referencetime"] = merged["referencetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged = merged.drop_duplicates(subset=["subject_id", "stay_id", "referencetime"])

    return merged[['subject_id', 'stay_id', 'referencetime']]

def create_controls_list(icustays_df, ihca_list, n = 956):
    controls = icustays_df[icustays_df['los'] >= 1.0].copy()
    merged = controls.merge(ihca_list[['subject_id','stay_id']], on=['subject_id', 'stay_id'], how='left', indicator=True)
    merged = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
    merged = merged.rename(columns={'outtime': 'referencetime'})
    merged["referencetime"] = merged["referencetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    controls = merged[['subject_id', 'stay_id', 'referencetime']]
    controls = controls.drop_duplicates(subset=["subject_id", "stay_id", "referencetime"])
    
    if int(len(controls)) > n:
        controls = controls.sample(n=n, random_state=42)
        return controls
    
    return controls

def create_metadata_table(patient_list, icustays_df, patients_df, admissions_df, diagnosis_df, mapper):
    '''Creates table of independent metadata features for each patient.'''
    df = patient_list[['subject_id', 'stay_id']]
    df = df.merge(icustays_df[['subject_id', 'stay_id', 'first_careunit']], on=['subject_id', 'stay_id'], how='left').rename(columns={'first_careunit': 'admission_type'})
    df = df.merge(patients_df[['subject_id', 'gender', 'anchor_age']], on=['subject_id'], how='left').rename(columns={'anchor_age': 'age', 'gender': 'sex'})
    
    admissions_unique = admissions_df[['subject_id', 'race']].groupby('subject_id').first().reset_index()
    df = df.merge(admissions_unique, on=['subject_id'], how='left')
    stay_to_hadm = icustays_df.set_index('stay_id')['hadm_id']
    df['hadm_id'] = df['stay_id'].map(stay_to_hadm)

    diagnosis_unique = diagnosis_df[diagnosis_df['seq_num'] == 1][['subject_id', 'icd_code', 'icd_version', 'seq_num', 'hadm_id']]
    df = df.merge(diagnosis_unique, on=['subject_id', 'hadm_id'], how='left')
    
    icd9_mask = df['icd_version'] == 9
    icd10_mask = df['icd_version'] == 10

    mapped_icd10 = mapper.map(df.loc[icd9_mask, 'icd_code'], source="icd9", target="icd10")

    mapped_icd10 = [x[0] if isinstance(x, list) else x for x in mapped_icd10]
    
    df.loc[icd9_mask, 'diagnosis'] = mapped_icd10
    df.loc[icd10_mask, 'diagnosis'] = df.loc[icd10_mask, 'icd_code']
    
    return df[['subject_id', 'admission_type', 'sex', 'age', 'race', 'diagnosis']]    

def query_vitals_vectorized(patient_df, vital, chart_events_path, hours_before=24):
    '''Returns a polars df of a specific vital for hours_before hours before the reference time for all patients.'''
    patient_df = patient_df.copy()
    patient_df["referencetime"] = pd.to_datetime(patient_df["referencetime"])
    patient_df["start_time"] = patient_df["referencetime"] - pd.Timedelta(hours=hours_before)
    
    patient_pl = pl.from_pandas(patient_df[["subject_id", "stay_id", "referencetime", "start_time"]]).lazy()
    
    vitals_lazy = (
        pl.scan_csv(chart_events_path)
        .with_columns(
            pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )
        .filter(pl.col("itemid") == vitals_code[vital])
    )
    
    joined = vitals_lazy.join(
        patient_pl,
        on=["subject_id", "stay_id"],
        how="inner"
    )
    
    filtered = joined.filter(
        (pl.col("charttime") >= pl.col("start_time")) &
        (pl.col("charttime") <= pl.col("referencetime"))
    ).select(["subject_id", "stay_id", "charttime", "valuenum"])
    
    df = filtered.collect()
    return df

def feature_extract_vital(patient_df, vitals_df, vital, condition, cutoff):

    patient_df = patient_df.copy()
    vitals_df = vitals_df.lazy()
    patient_df["referencetime"] = pd.to_datetime(patient_df["referencetime"])

    # create hourly bins
    all_rows = []
    for _, row in patient_df.iterrows():
        ref_time = row["referencetime"]
        subject_id = row["subject_id"]
        stay_id = row["stay_id"]
        for hour_subtract in range(0, 24):
            bin_time = ref_time - pd.Timedelta(hours=hour_subtract)
            all_rows.append({
                "subject_id": subject_id,
                "stay_id": stay_id,
                "bin_hour": hour_subtract,
                "bin_time": bin_time
            })
    bins = pl.from_pandas(pd.DataFrame(all_rows)).lazy()

    # join vitals
    joined = bins.join(vitals_df, on=["subject_id","stay_id"], how="left")

    # delta_hour filter
    joined = joined.with_columns(
        ((pl.col("charttime") - pl.col("bin_time")).dt.total_seconds() / 3600).alias("delta_hour")
    ).filter(
        ((pl.col("delta_hour") > -1) & (pl.col("delta_hour") <= 0)) | pl.col("delta_hour").is_null()
    )

    # aggregation per hour per patient
    if condition == "ctrl":
        agg_expr = get_agg_expr("valuenum", "mean")
    elif condition == "ihca":
        if vital in {"SBP","DBP","MAP","SpO2"}:
            agg_expr = get_agg_expr("valuenum", "min")
        elif vital == "HR":
            agg_expr = get_agg_expr("valuenum", conditional=True, threshold=80)
        elif vital == "RR":
            agg_expr = get_agg_expr("valuenum", conditional=True, threshold=12)
        else:
            raise ValueError("Invalid vital sign.")
    else:
        raise ValueError("Invalid condition group provided.")

    aggregated = joined.group_by(["subject_id","stay_id","bin_hour"]).agg(agg_expr)
    aggregated = aggregated.sort(["subject_id","stay_id","bin_hour"])

    # rejoin bins to keep all hours
    aggregated = bins.join(
        aggregated.select(["subject_id","stay_id","bin_hour","valuenum"]),
        on=["subject_id","stay_id","bin_hour"], how="left"
    )

    # LOCF per patient
    aggregated = aggregated.with_columns(
        pl.col("valuenum").backward_fill().over(["subject_id","stay_id"])
    )

    # filter cutoff
    aggregated = aggregated.filter(pl.col("bin_hour") <= cutoff)

    return aggregated.collect()

def get_agg_expr(col, agg_type=None, conditional=False, threshold=None):
    expr = pl.col(col)

    if conditional:
        if threshold is None:
            raise ValueError("Conditional aggregation requires a threshold value.")
        else:
            return (
                # implements conditional feature extraction
                pl.when(pl.min(col) >= threshold).then(pl.max(col)).otherwise(
                    pl.when(pl.max(col) <= threshold).then(pl.min(col)).otherwise(pl.min(col)))
                ) 
    if agg_type == "mean":
        return expr.mean()
    elif agg_type == "min":
        return expr.min()
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")

def impute_patient_average_column(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """
    Impute missing values with patient/stay average, then global average.
    """
    df_lazy = df.lazy()
    
    df_filled = (
        df_lazy
        .with_columns([
            # Compute patient/stay mean
            pl.col(col_name).mean().over(["subject_id", "stay_id"]).alias(f"{col_name}_mean"),
            # Compute global mean
            pl.col(col_name).mean().alias(f"{col_name}_global_mean")
        ])
        .with_columns(
            # Fill with patient mean first, then global mean
            pl.col(col_name)
              .fill_null(pl.col(f"{col_name}_mean"))
              .fill_null(pl.col(f"{col_name}_global_mean"))
        )
        .drop([f"{col_name}_mean", f"{col_name}_global_mean"])
        .collect()
    )
    
    return df_filled
