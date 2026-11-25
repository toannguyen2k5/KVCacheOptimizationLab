from fastapi import APIRouter, Request

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(request: Request) -> dict[str, list[dict[str, object]]]:
    return {"items": [request.app.state.experiment_service.model_info()]}
