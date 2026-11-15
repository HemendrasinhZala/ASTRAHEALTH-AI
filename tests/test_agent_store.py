
from src.agents.agent_store import (
    get_recent_executions,
    save_agent_execution,
    update_approval,
)


def sample_result():
    return {
        "sample_name": "C003_R+194",
        "subject_id": "C003",
        "timepoint": "R+194",
        "monitoring_priority_score": 30,
        "monitoring_priority": "Review Recommended",
        "approval_required": True,
        "recommendation": "Flag this observation for human review.",
        "is_clinical_diagnosis": False,
    }


def test_save_and_retrieve_execution(tmp_path):
    db_path = tmp_path / "test_agent.db"

    execution_id = save_agent_execution(sample_result(), db_path)
    records = get_recent_executions(db_path=db_path)

    assert execution_id > 0
    assert len(records) == 1
    assert records[0]["sample_name"] == "C003_R+194"
    assert records[0]["approval_status"] == "pending"
    assert records[0]["result"]["subject_id"] == "C003"


def test_update_human_approval(tmp_path):
    db_path = tmp_path / "test_agent.db"

    execution_id = save_agent_execution(sample_result(), db_path)

    updated = update_approval(
        execution_id,
        "approved",
        "Reviewed for demonstration.",
        db_path,
    )

    records = get_recent_executions(db_path=db_path)

    assert updated is True
    assert records[0]["approval_status"] == "approved"
    assert records[0]["reviewer_note"] == "Reviewed for demonstration."


def test_invalid_approval_status_is_rejected(tmp_path):
    db_path = tmp_path / "test_agent.db"

    execution_id = save_agent_execution(sample_result(), db_path)

    try:
        update_approval(execution_id, "maybe", db_path=db_path)
        assert False, "Expected ValueError"
    except ValueError:
        pass
