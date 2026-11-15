
"""SQLite persistence for ASTRAHEALTH AI agent executions and approvals."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "storage" / "agent_decisions.db"


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_store(db_path: str | Path | None = None) -> None:
    """Create the execution table if it does not already exist."""
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_name TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                priority TEXT NOT NULL,
                priority_score REAL NOT NULL,
                approval_status TEXT NOT NULL DEFAULT 'pending',
                reviewer_note TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_executions_created
            ON agent_executions(created_at)
            """
        )


def save_agent_execution(
    result: Mapping[str, Any],
    db_path: str | Path | None = None,
) -> int:
    """Persist one orchestrated monitoring result and return its database ID."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping or dictionary")

    initialize_store(db_path)

    sample_name = str(result.get("sample_name", "Unknown"))
    subject_id = str(result.get("subject_id", "Unknown"))
    priority = str(result.get("monitoring_priority", "Routine Monitoring"))

    try:
        score = float(result.get("monitoring_priority_score", 0))
    except (TypeError, ValueError):
        score = 0.0

    approval_status = (
        "pending" if result.get("approval_required", False)
        else "not_required"
    )
    now = datetime.now(timezone.utc).isoformat()

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO agent_executions (
                sample_name, subject_id, priority, priority_score,
                approval_status, result_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_name,
                subject_id,
                priority,
                score,
                approval_status,
                json.dumps(dict(result), default=str),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)


def update_approval(
    execution_id: int,
    status: str,
    reviewer_note: str = "",
    db_path: str | Path | None = None,
) -> bool:
    """Update a saved execution's human approval decision."""
    allowed_statuses = {"approved", "rejected"}

    if status not in allowed_statuses:
        raise ValueError("status must be 'approved' or 'rejected'")

    initialize_store(db_path)
    now = datetime.now(timezone.utc).isoformat()

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE agent_executions
            SET approval_status = ?, reviewer_note = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, reviewer_note, now, int(execution_id)),
        )
        return cursor.rowcount == 1


def get_recent_executions(
    limit: int = 20,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the newest saved executions, including their structured result."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    initialize_store(db_path)

    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM agent_executions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["result"] = json.loads(item.pop("result_json"))
        results.append(item)

    return results
