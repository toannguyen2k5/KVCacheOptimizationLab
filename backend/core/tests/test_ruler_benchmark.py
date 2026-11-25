import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.ruler_benchmark import run_ruler_benchmark, score_ruler_answer


class FakeInferenceEngine:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self, prompt: str, algorithm: str, max_new_tokens: int
    ) -> dict[str, object]:
        self.prompts.append(prompt)
        return {
            "memory_metrics": {
                "memory_raw_cache": 4.0,
                "peak_memory": 8.0,
            },
            "compress_metrics": {"compression_raito": 1.5},
            "reconstruction_metrics": {
                "key_mse": 0.1,
                "mse_scope": "dense_full_cache",
            },
            "accuracy_metrics": {
                "token_agreement": 0.5,
                "matching_tokens": 1,
                "compared_tokens": 2,
                "exact_sequence_match": False,
                "reference_generation_time": 0.2,
            },
            "Generated_tokens": 2,
            "generated_text": "The answer is BLUE.",
        }


class RulerBenchmarkTests(unittest.TestCase):
    def test_scoring_matches_rulerv1_all_and_part_rules(self) -> None:
        prediction = "The answer is Blue."
        self.assertEqual(
            score_ruler_answer(prediction, ["blue", "green"], "all"),
            0.5,
        )
        self.assertEqual(
            score_ruler_answer(prediction, ["blue", "green"], "part"),
            1.0,
        )

    def test_prepared_samples_run_through_optimizer_engine_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            task_file = Path(data_dir) / "niah_single_1" / "test.jsonl"
            task_file.parent.mkdir(parents=True)
            task_file.write_text(
                json.dumps(
                    {
                        "index": 0,
                        "question": "Find the needle. Answer:",
                        "expected_answer": ["blue"],
                        "length": 4096,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            engine = FakeInferenceEngine()
            config = {
                "method": "xkv_sr",
                "sample_limit": 10,
                "max_new_tokens": 64,
                "context_length": 4096,
            }
            with patch.dict(
                os.environ, {"KV_CACHE_RULER_DATA_DIR": data_dir}
            ):
                result = run_ruler_benchmark(engine, config)

        self.assertEqual(engine.prompts, ["Find the needle. Answer:"])
        self.assertEqual(result["ruler_metrics"]["task_accuracy"], 1.0)
        self.assertEqual(result["ruler_metrics"]["sample_count"], 1)
        self.assertEqual(result["accuracy_metrics"]["token_agreement"], 0.5)
        self.assertEqual(result["memory_metrics"]["peak_memory"], 8.0)


if __name__ == "__main__":
    unittest.main()
