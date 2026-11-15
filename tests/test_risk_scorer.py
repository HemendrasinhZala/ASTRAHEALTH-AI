
from src.risk_engine.risk_scorer import AstronautHealthRiskScorer


def test_routine_monitoring():
    scorer = AstronautHealthRiskScorer()

    result = scorer.calculate_score(
        autoencoder_status="No Anomaly Flag",
        isolation_forest_status="No Anomaly Flag",
        reconstruction_error=0.2,
        autoencoder_threshold=0.5
    )

    assert result["monitoring_priority_score"] == 0
    assert result["monitoring_priority"] == "Routine Monitoring"


def test_review_recommended():
    scorer = AstronautHealthRiskScorer()

    result = scorer.calculate_score(
        autoencoder_status="Anomaly Flag",
        isolation_forest_status="No Anomaly Flag",
        reconstruction_error=0.6,
        autoencoder_threshold=0.5
    )

    assert result["monitoring_priority_score"] == 30
    assert result["monitoring_priority"] == "Review Recommended"


def test_priority_review():
    scorer = AstronautHealthRiskScorer()

    result = scorer.calculate_score(
        autoencoder_status="Anomaly Flag",
        isolation_forest_status="Anomaly Flag",
        reconstruction_error=0.9,
        autoencoder_threshold=0.5
    )

    assert result["monitoring_priority_score"] == 80
    assert result["monitoring_priority"] == "Priority Review"


def test_not_a_clinical_diagnosis():
    scorer = AstronautHealthRiskScorer()

    result = scorer.calculate_score(
        autoencoder_status="No Anomaly Flag",
        isolation_forest_status="No Anomaly Flag",
        reconstruction_error=0.2,
        autoencoder_threshold=0.5
    )

    assert result["is_clinical_diagnosis"] is False
