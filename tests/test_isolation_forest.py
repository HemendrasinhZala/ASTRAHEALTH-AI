
import numpy as np
import pytest

from src.models.isolation_forest import AstronautHealthIsolationForest


def test_model_fit():

    X = np.random.default_rng(42).normal(size=(100, 18))

    model = AstronautHealthIsolationForest()
    result = model.fit(X)

    assert model.is_fitted is True
    assert result is model


def test_prediction_output():

    X = np.random.default_rng(42).normal(size=(100, 18))

    model = AstronautHealthIsolationForest()
    model.fit(X)

    predictions = model.predict(X[:10])

    assert len(predictions) == 10
    assert set(predictions).issubset({
        "Anomaly Flag",
        "No Anomaly Flag"
    })


def test_anomaly_scores():

    X = np.random.default_rng(42).normal(size=(100, 18))

    model = AstronautHealthIsolationForest()
    model.fit(X)

    scores = model.anomaly_scores(X[:10])

    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))


def test_predict_before_fit():

    model = AstronautHealthIsolationForest()

    X = np.zeros((5, 18))

    with pytest.raises(RuntimeError):
        model.predict(X)
