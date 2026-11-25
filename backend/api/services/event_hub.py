import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any


class ExperimentEventHub:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]]] = defaultdict(set)

    def subscribe(
        self, experiment_id: str
    ) -> tuple[asyncio.Queue[dict[str, Any]], AsyncIterator[dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        subscriber = (loop, queue)
        self._subscribers[experiment_id].add(subscriber)

        async def stream() -> AsyncIterator[dict[str, Any]]:
            try:
                while True:
                    yield await queue.get()
            finally:
                self._subscribers[experiment_id].discard(subscriber)
                if not self._subscribers[experiment_id]:
                    self._subscribers.pop(experiment_id, None)

        return queue, stream()

    def publish(self, experiment_id: str, event: dict[str, Any]) -> None:
        for loop, queue in tuple(self._subscribers.get(experiment_id, ())):
            def enqueue(target: asyncio.Queue[dict[str, Any]] = queue) -> None:
                if target.full():
                    try:
                        target.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                target.put_nowait(event)

            if not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(enqueue)
                except RuntimeError:
                    subscribers = self._subscribers.get(experiment_id)
                    if subscribers is not None:
                        subscribers.discard((loop, queue))
                        if not subscribers:
                            self._subscribers.pop(experiment_id, None)
