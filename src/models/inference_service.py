
from pathlib import Path

import pandas as pd
import streamlit as st

from src.models.predict import AstronautHealthPredictor


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "astronaut_health_features_ml.csv"
)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_predictor():
    """Load trained models once and reuse them."""

    predictor = AstronautHealthPredictor()
    predictor.load_models()

    return predictor


# --------------------------------------------------
# LOAD MODEL INPUT FEATURES
# --------------------------------------------------

@st.cache_data
def load_feature_data():
    """Load the 18 features required by the ML models."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_FILE}"
        )

    features = pd.read_csv(FEATURE_FILE)

    if features.shape[1] != 18:
        raise ValueError(
            f"Expected 18 features, found {features.shape[1]}."
        )

    return features


# --------------------------------------------------
# LIVE MODEL INFERENCE
# --------------------------------------------------

def get_live_predictions():
    """
    Generate fresh predictions using:
    1. Autoencoder
    2. Isolation Forest
    3. Experimental Risk Scoring Engine
    """

    predictor = load_predictor()

    features = load_feature_data()

    predictions = predictor.predict(features)

    results = pd.DataFrame({
        "ISOLATION_FOREST_STATUS":
            predictions["isolation_forest_status"],

        "ISOLATION_FOREST_SCORE":
            predictions["isolation_forest_score"],

        "AUTOENCODER_STATUS":
            predictions["autoencoder_status"],

        "RECONSTRUCTION_ERROR":
            predictions["reconstruction_error"],
    })

    risk_results = pd.DataFrame(
        predictions["risk_results"]
    )

    results = pd.concat(
        [results, risk_results],
        axis=1
    )

    return results


# --------------------------------------------------
# COMBINE METADATA WITH LIVE PREDICTIONS
# --------------------------------------------------

def get_live_monitoring_data(metadata_df):
    """
    Combine astronaut observation metadata
    with actual ML model predictions.
    """

    predictions = get_live_predictions()

    if len(metadata_df) != len(predictions):
        raise ValueError(
            "Metadata and prediction record counts do not match."
        )

    live_df = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            predictions.reset_index(drop=True)
        ],
        axis=1
    )

    return live_df
