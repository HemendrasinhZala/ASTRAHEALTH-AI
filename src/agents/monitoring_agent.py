
"""Backend orchestration for ASTRAHEALTH AI monitoring decisions.

This module coordinates existing model outputs into a reviewable workflow.
It does not provide medical diagnosis or independently calculate model risk.
"""

from typing import Any, Mapping


def _get(record: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a value using the first matching column/key name."""
    for key in keys:
        if key in record:
            value = record[key]
            # Convert common pandas/numpy scalar values to Python scalars.
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except (ValueError, TypeError):
                    pass
            return value
    return default


def orchestrate_monitoring(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build a structured agent workflow from one live prediction record."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping or dictionary-like row")

    sample = _get(record, "Sample Name", "sample_name", default="Unknown")
    subject = _get(record, "SUBJECT_ID", "subject_id", default="Unknown")
    timepoint = _get(record, "TIMEPOINT", "timepoint", default="Unknown")
    mission_day = _get(record, "MISSION_DAY", "mission_day")

    if_status = str(_get(
        record, "ISOLATION_FOREST_STATUS", "isolation_forest_status",
        default="Not Available"
    ))
    ae_status = str(_get(
        record, "AUTOENCODER_STATUS", "autoencoder_status",
        default="Not Available"
    ))
    reconstruction_error = _get(
        record, "RECONSTRUCTION_ERROR", "reconstruction_error"
    )
    score = _get(
        record, "MONITORING_PRIORITY_SCORE", "monitoring_priority_score",
        default=0
    )
    priority = str(_get(
        record, "MONITORING_PRIORITY", "monitoring_priority",
        default="Routine Monitoring"
    ))
    explanations = _get(
        record, "EXPLANATIONS", "explanations", default=[]
    )

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    if priority == "Priority Review":
        recommendation = (
            "Escalate this observation for priority human review. "
            "Verify the source measurements and review the participant's "
            "available history."
        )
    elif priority == "Review Recommended":
        recommendation = (
            "Flag this observation for human review. Compare it with the "
            "participant's previous observations and verify the measurements."
        )
    else:
        recommendation = (
            "Continue routine monitoring and review subsequent observations "
            "when available."
        )

    if isinstance(explanations, str):
        explanation_items = [explanations] if explanations else []
    elif explanations is None:
        explanation_items = []
    else:
        try:
            explanation_items = list(explanations)
        except TypeError:
            explanation_items = [str(explanations)]

    return {
        "sample_name": str(sample),
        "subject_id": str(subject),
        "timepoint": str(timepoint),
        "mission_day": mission_day,
        "model_outputs": {
            "isolation_forest_status": if_status,
            "autoencoder_status": ae_status,
            "reconstruction_error": reconstruction_error,
        },
        "monitoring_priority_score": score,
        "monitoring_priority": priority,
        "explanations": explanation_items,
        "recommendation": recommendation,
        "approval_required": priority != "Routine Monitoring",
        "workflow": [
            {"agent": "Data Collection Agent", "status": "completed"},
            {"agent": "Anomaly Analysis Agent", "status": "completed"},
            {"agent": "Risk Assessment Agent", "status": "completed"},
            {"agent": "Recommendation Agent", "status": "completed"},
            {"agent": "Human Review", "status": (
                "pending" if priority != "Routine Monitoring"
                else "not_required"
            )},
        ],
        "is_clinical_diagnosis": False,
        "disclaimer": (
            "Experimental decision-support output based on historical "
            "astronaut blood measurements. Not a medical diagnosis."
        ),
    }
