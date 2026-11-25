from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/experiments", tags=["results"])


@router.get("/{experiment_id}/result")
def get_experiment_result(
    experiment_id: str, request: Request
) -> dict[str, object]:
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
    if experiment["status"] != "completed" or experiment["result"] is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RESULT_NOT_READY",
                "message": "Experiment result is not available yet.",
                "details": {"status": experiment["status"]},
            },
        )

    result = experiment["result"]
    return {
        "experiment_id": experiment_id,
        "inference": {
            "generated_text": result.get("generated_text"),
            "reference_generated_text": result.get("reference_generated_text"),
            "prompt_tokens": result.get("prompt_tokens"),
            "generated_tokens": result.get("Generated_tokens"),
            "latency_seconds": result.get("latency_seconds"),
        },
        "metrics": {
            "memory": result.get("memory_metrics"),
            "compression": result.get("compress_metrics"),
            "reconstruction": result.get("reconstruction_metrics"),
            "accuracy": result.get("accuracy_metrics"),
            "benchmark": result.get("ruler_metrics"),
        },
    }
