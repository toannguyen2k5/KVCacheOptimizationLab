import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app
from api.services.experiment_service import ExperimentService
from api.services.experiment_store import ExperimentStore


class FakeFacade:
    def __init__(self, fail: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = fail
        self.received_config: dict[str, object] | None = None

    def model_info(self) -> dict[str, object]:
        return {
            "id": "qwen3-8b",
            "name": "Qwen3-8B",
            "status": "not_loaded",
            "device": None,
            "dtype": None,
            "error": None,
        }

    def run_experiment(self, config: dict[str, object]) -> dict[str, object]:
        self.received_config = config
        self.started.set()
        self.release.wait(timeout=5)
        if self.fail:
            raise RuntimeError("model path should not be exposed")
        return {
            "generated_text": "hello",
            "Generated_tokens": 1,
            "memory_metrics": {"memory_raw_cache": 3.0},
            "compress_metrics": None,
            "reconstruction_metrics": None,
            "accuracy_metrics": {
                "token_agreement": 1.0,
                "exact_sequence_match": True,
                "compared_tokens": 1,
            },
            "reference_generated_text": "hello",
        }


class ExperimentAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.facade = FakeFacade()
        store = ExperimentStore(Path(self.temp_dir.name) / "experiments.sqlite3")
        self.service = ExperimentService(store=store, facade=self.facade)
        self.client_context = TestClient(create_app(self.service))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.facade.release.set()
        self.client_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def test_health_does_not_load_model(self) -> None:
        self.assertEqual(self.client.get("/api/v1/health").json()["status"], "ok")
        model = self.client.get("/api/v1/models").json()["items"][0]
        self.assertEqual(model["status"], "not_loaded")
        self.assertIsNone(model["device"])

    def test_validation_errors_use_shared_error_shape(self) -> None:
        response = self.client.post(
            "/api/v1/experiments/validate",
            json={"prompt": " ", "max_new_tokens": 0, "method": "other"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "VALIDATION_ERROR")
        self.assertIn("details", response.json())

    def test_xkv_sr_is_advertised_and_validated(self) -> None:
        options = self.client.get("/api/v1/experiments/config-options")
        self.assertIn("xkv_sr", options.json()["methods"])

        methods = self.client.get("/api/v1/optimization/methods")
        self.assertIn(
            "xkv_sr",
            [method["id"] for method in methods.json()["items"]],
        )

        validation = self.client.post(
            "/api/v1/experiments/validate",
            json={"prompt": "test", "method": "xkv_sr", "max_new_tokens": 1},
        )
        self.assertEqual(validation.status_code, 200)
        self.assertEqual(validation.json()["config"]["method"], "xkv_sr")

        method_validation = self.client.post(
            "/api/v1/optimization/validate",
            json={"method_id": "xkv_sr"},
        )
        self.assertEqual(method_validation.status_code, 200)
        self.assertEqual(method_validation.json()["method_id"], "xkv_sr")

    def test_xkv_sr_experiment_config_reaches_inference_facade(self) -> None:
        response = self.client.post(
            "/api/v1/experiments",
            json={"prompt": "test", "method": "xkv_sr", "max_new_tokens": 1},
        )
        self.assertEqual(response.status_code, 202)
        self.assertTrue(self.facade.started.wait(timeout=2))
        self.assertIsNotNone(self.facade.received_config)
        self.assertEqual(self.facade.received_config["method"], "xkv_sr")

    def test_ruler_smoke_queues_only_when_prepared_data_exists(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            task_file = Path(data_dir) / "niah_single_1" / "test.jsonl"
            task_file.parent.mkdir(parents=True)
            task_file.write_text(
                '{"question":"Find the needle.","expected_answer":["blue"]}\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"KV_CACHE_RULER_DATA_DIR": data_dir}):
                options = self.client.get(
                    "/api/v1/experiments/config-options"
                ).json()["ruler"]
                self.assertTrue(options["available"])
                self.assertEqual(options["available_samples"], 1)

                response = self.client.post(
                    "/api/v1/experiments/ruler",
                    json={"method": "xkv_sr"},
                )
                self.assertEqual(response.status_code, 202)
                self.assertTrue(self.facade.started.wait(timeout=2))
                self.assertEqual(
                    self.facade.received_config["benchmark"], "ruler_v1_ns"
                )
                self.assertEqual(
                    self.facade.received_config["task"], "niah_single_1"
                )

    def test_ruler_smoke_rejects_missing_prepared_data(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post(
                "/api/v1/experiments/ruler",
                json={"method": "xkv"},
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "RULER_DATA_NOT_READY")

    def test_create_returns_immediately_and_completed_result_is_normalized(self) -> None:
        response = self.client.post(
            "/api/v1/experiments",
            json={"prompt": "test", "method": "xkv", "max_new_tokens": 12},
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")
        self.assertTrue(self.facade.started.wait(timeout=2))

        result_response = self.client.get(
            f"/api/v1/experiments/{response.json()['experiment_id']}/result"
        )
        self.assertEqual(result_response.status_code, 409)

        self.facade.release.set()
        experiment_id = response.json()["experiment_id"]
        for _ in range(100):
            experiment = self.client.get(
                f"/api/v1/experiments/{experiment_id}"
            ).json()
            if experiment["status"] == "completed":
                break
            threading.Event().wait(0.01)
        self.assertEqual(experiment["status"], "completed")

        result = self.client.get(
            f"/api/v1/experiments/{experiment_id}/result"
        ).json()
        self.assertEqual(result["inference"]["generated_text"], "hello")
        self.assertEqual(
            result["inference"]["reference_generated_text"], "hello"
        )
        self.assertIsNone(result["inference"]["prompt_tokens"])
        self.assertEqual(result["metrics"]["memory"]["memory_raw_cache"], 3.0)
        self.assertEqual(result["metrics"]["accuracy"]["token_agreement"], 1.0)

    def test_unknown_experiment_is_404(self) -> None:
        response = self.client.get("/api/v1/experiments/missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "EXPERIMENT_NOT_FOUND")

    def test_unknown_endpoint_uses_shared_error_shape(self) -> None:
        response = self.client.get("/api/v1/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "HTTP_ERROR")
        self.assertIn("details", response.json())

    def test_interrupted_experiments_are_marked_failed_on_restart(self) -> None:
        experiment_id = "exp_interrupted"
        self.service.store.create(
            {
                "experiment_id": experiment_id,
                "status": "running",
                "created_at": "2026-01-01T00:00:00+00:00",
                "config": {"prompt": "test", "method": "xkv", "max_new_tokens": 1},
                "logs": [],
            }
        )
        self.service.start()
        experiment = self.service.get(experiment_id)
        self.assertEqual(experiment["status"], "failed")
        self.assertIn("backend restart", experiment["error_message"])

    def test_websocket_sends_initial_status_and_failure_event(self) -> None:
        failing = FakeFacade(fail=True)
        failing.release.set()
        service = ExperimentService(
            store=ExperimentStore(Path(self.temp_dir.name) / "failing.sqlite3"),
            facade=failing,
        )
        with TestClient(create_app(service)) as client:
            created = client.post(
                "/api/v1/experiments",
                json={"prompt": "test", "max_new_tokens": 1},
            ).json()
            with client.websocket_connect(
                f"/ws/experiments/{created['experiment_id']}"
            ) as socket:
                messages = []
                for _ in range(8):
                    message = socket.receive_json()
                    messages.append(message)
                    if message["event"] == "error":
                        break
                self.assertEqual(messages[0]["event"], "status")
                errors = [message for message in messages if message["event"] == "error"]
                self.assertTrue(errors)
                self.assertNotIn("model path", str(errors[0]))


if __name__ == "__main__":
    unittest.main()
