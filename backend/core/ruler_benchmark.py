import json
import os
from pathlib import Path
from statistics import mean
from typing import Any


RULER_DATA_DIR_ENV = "KV_CACHE_RULER_DATA_DIR"
RULER_TASK = "niah_single_1"
RULER_CONTEXT_LENGTH = 4096
RULER_SAMPLE_LIMIT = 10
RULER_MAX_NEW_TOKENS = 64


def ruler_dataset_file() -> Path | None:
    data_dir = os.environ.get(RULER_DATA_DIR_ENV)
    if not data_dir:
        return None
    return Path(data_dir).expanduser() / RULER_TASK / "test.jsonl"


def ruler_dataset_options() -> dict[str, Any]:
    dataset_file = ruler_dataset_file()
    available_samples = 0
    if dataset_file is not None and dataset_file.is_file():
        with dataset_file.open("r", encoding="utf-8") as file:
            available_samples = sum(bool(line.strip()) for line in file)
    return {
        "available": available_samples > 0,
        "task": RULER_TASK,
        "context_length": RULER_CONTEXT_LENGTH,
        "sample_limit": RULER_SAMPLE_LIMIT,
        "max_new_tokens": RULER_MAX_NEW_TOKENS,
        "data_dir_env": RULER_DATA_DIR_ENV,
        "available_samples": available_samples,
    }


def _read_samples(dataset_file: Path, sample_limit: int) -> list[dict[str, Any]]:
    samples = []
    with dataset_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            question = sample.get("question")
            answers = sample.get("expected_answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"RULER sample at line {line_number} has no valid question."
                )
            if isinstance(answers, str):
                answers = [answers]
            if (
                not isinstance(answers, list)
                or not answers
                or any(not isinstance(answer, str) or not answer for answer in answers)
            ):
                raise ValueError(
                    f"RULER sample at line {line_number} has invalid expected_answer."
                )
            sample["question"] = question
            sample["expected_answer"] = answers
            samples.append(sample)
            if len(samples) >= sample_limit:
                break
    if not samples:
        raise ValueError(f"No RULER samples found in {dataset_file}.")
    return samples


def score_ruler_answer(
    prediction: str, expected_answers: list[str], match_type: str = "all"
) -> float:
    matches = [
        answer.lower() in prediction.lower() for answer in expected_answers
    ]
    if match_type == "all":
        return sum(matches) / len(matches)
    if match_type == "part":
        return float(any(matches))
    raise ValueError(f"Unsupported RULER match type: {match_type}")


def _average_numeric_metrics(
    metric_rows: list[dict[str, Any]], *, peak_keys: frozenset[str] = frozenset()
) -> dict[str, Any] | None:
    if not metric_rows:
        return None
    keys = set.intersection(*(set(row) for row in metric_rows))
    aggregate: dict[str, Any] = {}
    for key in keys:
        values = [row[key] for row in metric_rows]
        if all(isinstance(value, (int, float)) for value in values):
            aggregate[key] = (
                max(values) if key in peak_keys else mean(values)
            )
        elif all(value == values[0] for value in values):
            aggregate[key] = values[0]
    return aggregate


def run_ruler_benchmark(inference_engine: Any, config: dict[str, Any]) -> dict[str, Any]:
    dataset_file = ruler_dataset_file()
    if dataset_file is None:
        raise FileNotFoundError(
            f"Set {RULER_DATA_DIR_ENV} to the prepared RULERv1-ns setup directory."
        )
    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"RULER data not found: {dataset_file}. Prepare task {RULER_TASK} first."
        )

    samples = _read_samples(dataset_file, int(config["sample_limit"]))
    predictions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for sample in samples:
        result = inference_engine.generate(
            prompt=sample["question"],
            algorithm=config["method"],
            max_new_tokens=int(config["max_new_tokens"]),
        )
        prediction = result["generated_text"]
        score = score_ruler_answer(
            prediction,
            sample["expected_answer"],
            match_type="all",
        )
        predictions.append(
            {
                "index": sample.get("index"),
                "length": sample.get("length"),
                "expected_answer": sample["expected_answer"],
                "prediction": prediction,
                "score": score,
            }
        )
        results.append(result)

    accuracy_rows = [
        result["accuracy_metrics"]
        for result in results
        if result.get("accuracy_metrics") is not None
    ]
    compared_tokens = sum(row["compared_tokens"] for row in accuracy_rows)
    matching_tokens = sum(row["matching_tokens"] for row in accuracy_rows)
    accuracy_metrics = {
        "token_agreement": (
            matching_tokens / compared_tokens if compared_tokens else None
        ),
        "exact_sequence_match": all(
            row["exact_sequence_match"] for row in accuracy_rows
        ),
        "exact_sequence_match_rate": mean(
            float(row["exact_sequence_match"]) for row in accuracy_rows
        )
        if accuracy_rows
        else None,
        "compared_tokens": compared_tokens,
        "reference_generation_time": sum(
            row["reference_generation_time"] for row in accuracy_rows
        ),
    }
    task_accuracy = mean(item["score"] for item in predictions)
    return {
        "memory_metrics": _average_numeric_metrics(
            [result["memory_metrics"] for result in results],
            peak_keys=frozenset({"peak_memory"}),
        ),
        "compress_metrics": _average_numeric_metrics(
            [result["compress_metrics"] for result in results]
        ),
        "reconstruction_metrics": _average_numeric_metrics(
            [
                result["reconstruction_metrics"]
                for result in results
                if result.get("reconstruction_metrics") is not None
            ]
        ),
        "accuracy_metrics": accuracy_metrics,
        "ruler_metrics": {
            "benchmark": "RULERv1-ns",
            "task": RULER_TASK,
            "context_length": int(config["context_length"]),
            "sample_count": len(predictions),
            "task_accuracy": task_accuracy,
            "correct_samples": sum(item["score"] == 1.0 for item in predictions),
            "samples": predictions,
        },
        "Generated_tokens": sum(result["Generated_tokens"] for result in results),
        "generated_text": (
            f"RULER {RULER_TASK}: {task_accuracy:.1%} "
            f"({sum(item['score'] == 1.0 for item in predictions)}/"
            f"{len(predictions)} fully correct)"
        ),
        "reference_generated_text": None,
    }
