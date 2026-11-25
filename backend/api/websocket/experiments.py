from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["experiments"])


@router.websocket("/ws/experiments/{experiment_id}")
async def experiment_updates(websocket: WebSocket, experiment_id: str) -> None:
    service = websocket.app.state.experiment_service
    experiment = service.get(experiment_id)
    if experiment is None:
        await websocket.close(code=1008, reason="Experiment not found.")
        return

    await websocket.accept()
    _, stream = service.events.subscribe(experiment_id)
    await websocket.send_json(
        {
            "event": "status",
            "experiment_id": experiment_id,
            "timestamp": experiment["created_at"],
            "data": {"status": experiment["status"]},
        }
    )
    if experiment["status"] in {"completed", "failed", "cancelled"}:
        if experiment["status"] == "completed":
            await websocket.send_json(
                {
                    "event": "result_ready",
                    "experiment_id": experiment_id,
                    "timestamp": experiment["completed_at"],
                    "data": {},
                }
            )
        elif experiment["status"] == "failed":
            await websocket.send_json(
                {
                    "event": "error",
                    "experiment_id": experiment_id,
                    "timestamp": experiment["completed_at"],
                    "data": {"message": experiment["error_message"]},
                }
            )
        await stream.aclose()
        await websocket.close(code=1000)
        return
    try:
        async for event in stream:
            await websocket.send_json(event)
            if event["event"] == "result_ready" or (
                event["event"] == "status"
                and event["data"].get("status") in {"failed", "cancelled"}
            ):
                await stream.aclose()
                await websocket.close(code=1000)
                return
    except WebSocketDisconnect:
        await stream.aclose()
