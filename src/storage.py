from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.models import ScenarioPack


class LocalStorage:
    def __init__(self, db_path: str = "data/crisissim.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenarios (
                    scenario_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_id TEXT NOT NULL,
                    phase_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    decision_text TEXT NOT NULL,
                    constraints TEXT,
                    evaluation_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_scenario(self, scenario: ScenarioPack) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scenarios (scenario_id, created_at, payload)
                VALUES (?, datetime('now'), ?)
                """,
                (scenario.scenario_id, json.dumps(scenario.model_dump(), ensure_ascii=False)),
            )
            conn.commit()

    def get_latest_scenario(self) -> ScenarioPack | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload
                FROM scenarios
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload"])
        return ScenarioPack.model_validate(payload)

    def save_decision(self, record: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (
                    scenario_id,
                    phase_id,
                    question,
                    decision_text,
                    constraints,
                    evaluation_json,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("scenario_id", ""),
                    record.get("phase_id", ""),
                    record.get("question", ""),
                    record.get("decision_text", ""),
                    record.get("constraints", ""),
                    json.dumps(record.get("evaluation", {}), ensure_ascii=False),
                    record.get("timestamp", ""),
                ),
            )
            conn.commit()

    def get_decisions(self, scenario_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT scenario_id, phase_id, question, decision_text, constraints, evaluation_json, timestamp
                FROM decisions
                WHERE scenario_id = ?
                ORDER BY id ASC
                """,
                (scenario_id,),
            ).fetchall()

        decisions = []
        for row in rows:
            decisions.append(
                {
                    "scenario_id": row["scenario_id"],
                    "phase_id": row["phase_id"],
                    "question": row["question"],
                    "decision_text": row["decision_text"],
                    "constraints": row["constraints"],
                    "evaluation": json.loads(row["evaluation_json"]),
                    "timestamp": row["timestamp"],
                }
            )
        return decisions
