
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

MONITORING_FILE = (
    PROCESSED_DATA / "astronaut_monitoring_results.csv"
)

HEALTH_FILE = (
    PROCESSED_DATA / "astronaut_health_ml_ready.csv"
)

AUTOENCODER_FILE = (
    PROJECT_ROOT / "artifacts" / "models" /
    "astronaut_autoencoder.pth"
)

ISOLATION_FOREST_FILE = (
    PROJECT_ROOT / "artifacts" / "models" /
    "isolation_forest.joblib"
)


# --------------------------------------------------
# REQUIRED COLUMNS
# --------------------------------------------------

MONITORING_COLUMNS = [
    "Sample Name",
    "SUBJECT_ID",
    "TIMEPOINT",
    "MISSION_DAY",
    "ISOLATION_FOREST_STATUS",
    "ISOLATION_FOREST_SCORE",
    "AUTOENCODER_STATUS",
    "RECONSTRUCTION_ERROR",
    "MONITORING_PRIORITY_SCORE",
    "MONITORING_PRIORITY",
    "EXPLANATIONS",
]

HEALTH_COLUMNS = [
    "Sample Name",
    "SUBJECT_ID",
    "TIMEPOINT",
    "MISSION_DAY",
    "hemoglobin_value_percent",
    "hematocrit_value_percent",
    "red_blood_cell_count_value_million_per_microliter",
    "white_blood_cell_count_value_thousand_per_microliter",
    "platelet_count_value_thousand_per_microliter",
]


# --------------------------------------------------
# LOAD MONITORING DATA
# --------------------------------------------------

@st.cache_data
def load_monitoring_data():

    if not MONITORING_FILE.exists():
        raise FileNotFoundError(
            f"Monitoring dataset not found: {MONITORING_FILE}"
        )

    df = pd.read_csv(MONITORING_FILE)

    missing_columns = [
        column for column in MONITORING_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing monitoring columns: {missing_columns}"
        )

    df["MISSION_DAY"] = pd.to_numeric(
        df["MISSION_DAY"], errors="coerce"
    )

    return df.sort_values(
        ["SUBJECT_ID", "MISSION_DAY"]
    ).reset_index(drop=True)


# --------------------------------------------------
# LOAD HEALTH DATA
# --------------------------------------------------

@st.cache_data
def load_health_data():

    if not HEALTH_FILE.exists():
        raise FileNotFoundError(
            f"Health dataset not found: {HEALTH_FILE}"
        )

    df = pd.read_csv(HEALTH_FILE)

    missing_columns = [
        column for column in HEALTH_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing health columns: {missing_columns}"
        )

    df["MISSION_DAY"] = pd.to_numeric(
        df["MISSION_DAY"], errors="coerce"
    )

    return df.sort_values(
        ["SUBJECT_ID", "MISSION_DAY"]
    ).reset_index(drop=True)


# --------------------------------------------------
# LOAD BOTH DATASETS
# --------------------------------------------------

def load_all_data():

    monitoring_df = load_monitoring_data()
    health_df = load_health_data()

    if monitoring_df["Sample Name"].duplicated().any():
        raise ValueError(
            "Duplicate sample names found in monitoring data."
        )

    if health_df["Sample Name"].duplicated().any():
        raise ValueError(
            "Duplicate sample names found in health data."
        )

    if set(monitoring_df["Sample Name"]) != set(
        health_df["Sample Name"]
    ):
        raise ValueError(
            "Monitoring and health datasets contain "
            "different sample records."
        )

    return monitoring_df, health_df


# --------------------------------------------------
# DATA SUMMARY
# --------------------------------------------------

def get_dashboard_summary(monitoring_df):

    return {
        "astronaut_count": monitoring_df[
            "SUBJECT_ID"
        ].nunique(),

        "observation_count": len(monitoring_df),

        "review_count": int(
            monitoring_df["MONITORING_PRIORITY"].isin(
                ["Review Recommended", "Priority Review"]
            ).sum()
        ),

        "priority_count": int(
            (
                monitoring_df["MONITORING_PRIORITY"]
                == "Priority Review"
            ).sum()
        ),

        "routine_count": int(
            (
                monitoring_df["MONITORING_PRIORITY"]
                == "Routine Monitoring"
            ).sum()
        ),

        "model_count": sum([
            AUTOENCODER_FILE.exists(),
            ISOLATION_FOREST_FILE.exists(),
        ]),
    }
