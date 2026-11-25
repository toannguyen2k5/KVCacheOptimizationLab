import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ApplicationFacade:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model_manager: Any = None
        self._model_status = "not_loaded"
        self._model_load_error: str | None = None

    def model_info(self) -> dict[str, Any]:
        with self._lock:
            status = self._model_status
            model_manager = self._model_manager
            error = self._model_load_error

        model = getattr(model_manager, "current_model", None)
        device = None
        dtype = None
        if model is not None:
            try:
                parameter = next(model.parameters())
                device = str(parameter.device)
                dtype = str(parameter.dtype).removeprefix("torch.")
            except (AttributeError, StopIteration, TypeError):
                pass

        return {
            "id": "qwen3-8b",
            "name": "Qwen3-8B",
            "status": status,
            "device": device,
            "dtype": dtype,
            "error": error if status == "error" else None,
        }

    def run_experiment(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._model_manager is None:
                self._model_status = "loading"
                try:
                    from models.qwen3_loader import Model_Qwen

                    self._model_manager = Model_Qwen()
                    self._model_manager.load_model()
                    self._model_status = "loaded"
                    self._model_load_error = None
                except Exception:
                    self._model_manager = None
                    self._model_status = "error"
                    self._model_load_error = "Model loading failed; check backend logs."
                    logger.exception("Model loading failed")
                    raise

            model_manager = self._model_manager

        from core.inference_engine import InferenceEngine
        from core.optimization_engine import OptimizationEngine
        from optimizers.xkv.xkv_optimizer import XKVOptimizer
        from optimizers.xkv_sr.xkv_sr_optimizer import XKV_SROptimizer

        optimization_engine = OptimizationEngine()
        optimization_engine.register("xkv", XKVOptimizer())
        optimization_engine.register("xkv_sr", XKV_SROptimizer())
        inference_engine = InferenceEngine(model_manager, optimization_engine)
        if config.get("benchmark") == "ruler_v1_ns":
            from core.ruler_benchmark import run_ruler_benchmark

            return run_ruler_benchmark(inference_engine, config)
        return inference_engine.generate(
            prompt=config["prompt"],
            algorithm=config["method"],
            max_new_tokens=config["max_new_tokens"],
        )
