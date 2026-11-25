from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Any, Generator


class ExperimentStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    config_json TEXT NOT NULL,
                    result_json TEXT,
                    error_message TEXT,
                    logs_json TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "experiment_id": row["experiment_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "config": json.loads(row["config_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_message": row["error_message"],
            "logs": json.loads(row["logs_json"]),
        }

    def create(self, experiment: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments
                    (experiment_id, status, created_at, config_json, logs_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment["experiment_id"],
                    experiment["status"],
                    experiment["created_at"],
                    json.dumps(experiment["config"]),
                    json.dumps(experiment.get("logs", [])),
                ),
            )

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def update(self, experiment_id: str, **fields: Any) -> dict[str, Any] | None:
        column_map = {
            "status": "status",
            "started_at": "started_at",
            "completed_at": "completed_at",
            "result": "result_json",
            "error_message": "error_message",
            "logs": "logs_json",
        }
        updates: list[str] = []
        values: list[Any] = []
        for field, value in fields.items():
            column = column_map.get(field)
            if column is None:
                raise ValueError(f"Unsupported experiment field: {field}")
            updates.append(f"{column} = ?")
            values.append(
                json.dumps(value) if field in {"result", "logs"} else value
            )
        if not updates:
            return self.get(experiment_id)
        values.append(experiment_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE experiment_id = ?",
                values,
            )
        return self.get(experiment_id)

    def list(
        self, page: int, page_size: int, status: str | None = None
    ) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        with self._connect() as connection:
            counts = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM experiments GROUP BY status"
                )
            }
            where = "WHERE status = ?" if status else ""
            params: tuple[Any, ...] = (status,) if status else ()
            total = connection.execute(
                f"SELECT COUNT(*) FROM experiments {where}", params
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM experiments {where}
                ORDER BY created_at DESC, experiment_id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, page_size, (page - 1) * page_size),
            ).fetchall()
        return [self._decode(row) for row in rows], total, counts

    def recover_interrupted(self) -> None:
        message = "Experiment interrupted by backend restart."
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE experiments
                SET status = 'failed',
                    completed_at = COALESCE(completed_at, created_at),
                    error_message = ?
                WHERE status IN ('queued', 'running')
                """,
                (message,),
            )
