from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from api.schemas import ExperimentConfig, RulerBenchmarkConfig
from core.ruler_benchmark import ruler_dataset_options

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/config-options")
def get_config_options() -> dict[str, object]:
    return {
        "generation": {
            "max_new_tokens": {"minimum": 1, "maximum": 1024, "default": 200},
            "temperature": None,
            "sampling": None,
        },
        "methods": ["xkv", "xkv_sr"],
        "prompt": {"minimum_length": 1, "maximum_length": 20000},
        "ruler": ruler_dataset_options(),
    }


@router.post("/validate")
def validate_experiment(config: ExperimentConfig) -> dict[str, object]:
    return {"valid": True, "config": config.model_dump()}


@router.post("", status_code=202)
def create_experiment(
    config: ExperimentConfig, request: Request
) -> dict[str, object]:
    return request.app.state.experiment_service.create(config)


@router.post("/ruler", status_code=202)
def create_ruler_benchmark(
    config: RulerBenchmarkConfig, request: Request
) -> dict[str, object]:
    dataset_options = ruler_dataset_options()
    if not dataset_options["available"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RULER_DATA_NOT_READY",
                "message": "Prepared RULERv1-ns data is not available.",
                "details": {
                    "data_dir_env": dataset_options["data_dir_env"],
                    "task": dataset_options["task"],
                },
            },
        )
    return request.app.state.experiment_service.create(config)


@router.get("")
def list_experiments(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Literal[
        "queued", "running", "completed", "failed", "cancelled"
    ]
    | None = None,
) -> dict[str, object]:
    items, total, counts = request.app.state.experiment_service.list(
        page, page_size, status
    )
    return {
        "items": [public_experiment(item) for item in items],
        "total": total,
        "page": page,
        "page_size": page_size,
        "counts": counts,
    }


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str, request: Request) -> dict[str, object]:
    experiment = request.app.state.experiment_service.get(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "EXPERIMENT_NOT_FOUND",
                "message": "Experiment not found.",
                "details": [],
            },
        )
    return public_experiment(experiment)


@router.get("/{experiment_id}/logs")
def get_experiment_logs(experiment_id: str, request: Request) -> dict[str, object]:
    logs = request.app.state.experiment_service.logs(experiment_id)
    if logs is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "EXPERIMENT_NOT_FOUND",
                "message": "Experiment not found.",
                "details": [],
            },
        )
    return {"experiment_id": experiment_id, "items": logs}


def public_experiment(experiment: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_id": experiment["experiment_id"],
        "status": experiment["status"],
        "created_at": experiment["created_at"],
        "started_at": experiment["started_at"],
        "completed_at": experiment["completed_at"],
        "config": experiment["config"],
        "error_message": experiment["error_message"],
    }
