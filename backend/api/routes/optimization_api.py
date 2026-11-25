from fastapi import APIRouter, HTTPException

from api.schemas import OptimizationValidationRequest

router = APIRouter(prefix="/optimization", tags=["optimization"])


@router.get("/methods")
def list_methods() -> dict[str, list[dict[str, object]]]:
    return {
        "items": [
            {
                "id": "xkv",
                "name": "XKV",
                "description": "Existing XKV KV Cache optimization pipeline.",
                "available": True,
                "parameters": [],
            },
            {
                "id": "xkv_sr",
                "name": "XKV-SR",
                "description": (
                    "XKV low-rank KV compression with query-guided selective "
                    "chunk reconstruction support. Current inference still "
                    "reconstructs a dense cache before generation."
                ),
                "available": True,
                "parameters": [],
            },
        ]
    }


@router.post("/validate")
def validate_method_config(
    request: OptimizationValidationRequest,
) -> dict[str, object]:
    if request.parameters:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "UNSUPPORTED_PARAMETERS",
                "message": (
                    "The selected optimization method does not expose "
                    "configurable parameters."
                ),
                "details": [],
            },
        )
    return {"valid": True, "method_id": request.method_id, "parameters": {}}
