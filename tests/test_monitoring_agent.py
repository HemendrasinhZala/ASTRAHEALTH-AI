
from src.agents.monitoring_agent import orchestrate_monitoring


def test_priority_review_requires_human_approval():
    result = orchestrate_monitoring({
        "Sample Name": "C003_R+194",
        "SUBJECT_ID": "C003",
        "TIMEPOINT": "R+194",
        "MISSION_DAY": 194,
        "ISOLATION_FOREST_STATUS": "No Anomaly Flag",
        "AUTOENCODER_STATUS": "Anomaly Flag",
        "RECONSTRUCTION_ERROR": 0.5414,
        "MONITORING_PRIORITY_SCORE": 30,
        "MONITORING_PRIORITY": "Review Recommended",
        "EXPLANATIONS": ["Autoencoder anomaly flag"],
    })

    assert result["subject_id"] == "C003"
    assert result["monitoring_priority"] == "Review Recommended"
    assert result["approval_required"] is True
    assert result["workflow"][-1]["status"] == "pending"
    assert result["is_clinical_diagnosis"] is False


def test_routine_monitoring_does_not_require_approval():
    result = orchestrate_monitoring({
        "Sample Name": "C001_R+194",
        "SUBJECT_ID": "C001",
        "TIMEPOINT": "R+194",
        "MISSION_DAY": 194,
        "ISOLATION_FOREST_STATUS": "No Anomaly Flag",
        "AUTOENCODER_STATUS": "No Anomaly Flag",
        "RECONSTRUCTION_ERROR": 0.1253,
        "MONITORING_PRIORITY_SCORE": 0,
        "MONITORING_PRIORITY": "Routine Monitoring",
        "EXPLANATIONS": [],
    })

    assert result["monitoring_priority"] == "Routine Monitoring"
    assert result["approval_required"] is False
    assert result["workflow"][-1]["status"] == "not_required"
