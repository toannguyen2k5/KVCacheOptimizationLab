from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from api.schemas import ExperimentConfig, RulerBenchmarkConfig
from api.services.application_facade import ApplicationFacade
from api.services.event_hub import ExperimentEventHub
from api.services.experiment_store import ExperimentStore

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentService:
    def __init__(
        self,
        store: ExperimentStore | None = None,
        facade: ApplicationFacade | None = None,
        events: ExperimentEventHub | None = None,
    ) -> None:
        if store is None:
            default_path = (
                Path.home()
                / ".kvcache-optimization-lab"
                / "experiments.sqlite3"
            )
            store = ExperimentStore(
                os.environ.get("KV_CACHE_EXPERIMENT_DB", str(default_path))
            )
        self.store = store
        self.facade = facade or ApplicationFacade()
        self.events = events or ExperimentEventHub()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kv-experiment"
        )
        self._shutdown = False
        self._lock = threading.Lock()

    def model_info(self) -> dict[str, Any]:
        return self.facade.model_info()

    def start(self) -> None:
        self.store.recover_interrupted()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def create(
        self, config: ExperimentConfig | RulerBenchmarkConfig
    ) -> dict[str, Any]:
        record = {
            "experiment_id": f"exp_{uuid4().hex}",
            "status": "queued",
            "created_at": utc_now(),
            "config": config.model_dump(),
            "logs": [],
        }
        self.store.create(record)
        self._add_log(record["experiment_id"], "Experiment queued.")
        self._publish_status(record["experiment_id"], "queued")
        try:
            self._executor.submit(self._execute, record["experiment_id"])
        except RuntimeError as error:
            self.store.update(
                record["experiment_id"],
                status="failed",
                completed_at=utc_now(),
                error_message="Experiment could not be scheduled.",
            )
            logger.exception("Could not schedule experiment")
            raise RuntimeError("Experiment could not be scheduled.") from error
        return {
            "experiment_id": record["experiment_id"],
            "status": "queued",
            "created_at": record["created_at"],
        }

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        return self.store.get(experiment_id)

    def list(
        self, page: int, page_size: int, status: str | None = None
    ) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        return self.store.list(page, page_size, status)

    def logs(self, experiment_id: str) -> list[dict[str, str]] | None:
        experiment = self.store.get(experiment_id)
        return experiment["logs"] if experiment else None

    def _add_log(self, experiment_id: str, message: str) -> None:
        experiment = self.store.get(experiment_id)
        if experiment is None:
            return
        entry = {"timestamp": utc_now(), "message": message}
        logs = [*experiment["logs"], entry][-500:]
        self.store.update(experiment_id, logs=logs)
        self.events.publish(
            experiment_id,
            {
                "event": "log",
                "experiment_id": experiment_id,
                "timestamp": entry["timestamp"],
                "data": entry,
            },
        )

    def _publish_status(self, experiment_id: str, status: str) -> None:
        self.events.publish(
            experiment_id,
            {
                "event": "status",
                "experiment_id": experiment_id,
                "timestamp": utc_now(),
                "data": {"status": status},
            },
        )

    def _execute(self, experiment_id: str) -> None:
        experiment = self.store.get(experiment_id)
        if experiment is None:
            return
        started_at = utc_now()
        self.store.update(
            experiment_id,
            status="running",
            started_at=started_at,
        )
        self._publish_status(experiment_id, "running")
        self._add_log(experiment_id, "Experiment started.")
        try:
            result = self.facade.run_experiment(experiment["config"])
            completed_at = utc_now()
            self.store.update(
                experiment_id,
                status="completed",
                completed_at=completed_at,
                result=result,
                error_message=None,
            )
            self._add_log(experiment_id, "Experiment completed.")
            self._publish_status(experiment_id, "completed")
            self.events.publish(
                experiment_id,
                {
                    "event": "result_ready",
                    "experiment_id": experiment_id,
                    "timestamp": completed_at,
                    "data": {},
                },
            )
        except Exception:
            logger.exception("Experiment %s failed", experiment_id)
            failed_at = utc_now()
            self.store.update(
                experiment_id,
                status="failed",
                completed_at=failed_at,
                error_message="Experiment execution failed. Check backend logs.",
            )
            self._add_log(experiment_id, "Experiment failed.")
            event = {
                "event": "error",
                "experiment_id": experiment_id,
                "timestamp": failed_at,
                "data": {"message": "Experiment execution failed."},
            }
            self.events.publish(experiment_id, event)
            self._publish_status(experiment_id, "failed")
