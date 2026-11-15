
import numpy as np
import pandas as pd
import pytest

from src.models.predict import AstronautHealthPredictor


@pytest.fixture(scope="module")
def predictor():

    model = AstronautHealthPredictor()
    model.load_models()

    return model


@pytest.fixture(scope="module")
def sample_features():

    project_root = predictor_project_root()

    data_path = (
        project_root
        / "data/processed/astronaut_health_features_ml.csv"
    )

    return pd.read_csv(data_path)


def predictor_project_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def test_predictor_loads(predictor):

    assert predictor.is_loaded is True


def test_prediction_output_keys(predictor, sample_features):

    results = predictor.predict(sample_features)

    expected_keys = {
        "isolation_forest_status",
        "isolation_forest_score",
        "autoencoder_status",
        "reconstruction_error",
        "autoencoder_threshold",
        "risk_results"
    }

    assert set(results.keys()) == expected_keys


def test_prediction_count(predictor, sample_features):

    results = predictor.predict(sample_features)

    assert len(results["risk_results"]) == len(sample_features)


def test_risk_scores_are_valid(predictor, sample_features):

    results = predictor.predict(sample_features)

    scores = [
        item["monitoring_priority_score"]
        for item in results["risk_results"]
    ]

    assert all(score in (0, 20, 30, 50, 60, 80) for score in scores)


def test_reconstruction_errors_are_finite(predictor, sample_features):

    results = predictor.predict(sample_features)

    assert np.isfinite(results["reconstruction_error"]).all()
