# Backend API

Install the API dependencies from the repository root:

```powershell
python -m pip install -r backend/api/requirements.txt
```

Run the API from the repository root. The `--app-dir backend` option keeps
the import paths used by the existing model and inference modules intact:

```powershell
python -m uvicorn api.app:app --app-dir backend --reload
```

The API is available at `http://localhost:8000`; OpenAPI documentation is at
`/docs`. The development frontend origins default to ports 5173 on localhost.
Set `FRONTEND_ORIGINS` to a comma-separated allowlist to override them.

Experiment metadata is stored in
`%USERPROFILE%\.kvcache-optimization-lab\experiments.sqlite3` on Windows
(`~/.kvcache-optimization-lab/experiments.sqlite3` elsewhere). Set
`KV_CACHE_EXPERIMENT_DB` to change the database location.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | API health |
| GET | `/api/v1/models` | Configured model status, without loading it |
| GET | `/api/v1/experiments/config-options` | Supported prompt and generation options |
| POST | `/api/v1/experiments/validate` | Validate prompt and generation config |
| POST | `/api/v1/experiments` | Queue an experiment; returns HTTP 202 |
| POST | `/api/v1/experiments/ruler` | Queue the prepared RULERv1-ns smoke benchmark |
| GET | `/api/v1/experiments` | Paginated history, optional status filter |
| GET | `/api/v1/experiments/{id}` | Experiment status and configuration |
| GET | `/api/v1/experiments/{id}/result` | Completed inference and actual backend metrics |
| GET | `/api/v1/experiments/{id}/logs` | Bounded experiment lifecycle logs |
| GET | `/api/v1/optimization/methods` | Integrated optimization methods |
| POST | `/api/v1/optimization/validate` | Validate supported optimization configuration |
| WS | `/ws/experiments/{id}` | Status, log, completion, and failure events |

The optimization methods endpoint advertises `xkv` and `xkv_sr`; choose either
value in the experiment `method` field. Temperature/sampling and per-method
parameters are omitted because the current inference interface does not accept
them. XKV-SR uses query-guided sparse reconstruction during Qwen3 autoregressive
decoding. Its reconstruction MSE is measured only over prompt tokens selected
for attention and reports the average selected-token fraction; the standard XKV
method continues to report dense-cache reconstruction metrics. Cancellation and
token-level progress are not advertised because the current interface does not
provide safe cancellation or progress callbacks. Metrics absent from the
existing inference result remain `null`.

Completed runs also include greedy token agreement and exact-sequence match
against the same model decoding with its full, uncompressed KV cache. This is a
reference-agreement metric, not task accuracy against a labeled answer, and
requires an additional full-cache generation pass.

## RULERv1-ns smoke benchmark

The setup page can run a 4K `niah_single_1` smoke test (up to 10 samples)
through the selected XKV method. Prepare the RULERv1-ns dataset using the
NeMo-Skills `chsieh/ruler-remove-prefix` branch and its `ns prepare_data ruler`
command; for example:

```powershell
ns prepare_data ruler `
  --cluster=local `
  --expname=ruler-data-qwen3-8b-4k `
  --data_dir=D:\ruler-data `
  --setup=qwen3-8b-4096 `
  --tokenizer_path=D:\xldl\KVCacheOptimizationLab\weights\Qwen3-8B `
  --max_seq_length=4096 `
  --tasks=niah_single_1 `
  --num_samples=10
```

Set `KV_CACHE_RULER_DATA_DIR` to the generated setup directory containing
`niah_single_1\test.jsonl`, then start the backend from that same PowerShell
session:

```powershell
$env:KV_CACHE_RULER_DATA_DIR = (Resolve-Path '.\data\ruler\qwen3-8b-4096').Path
```

The setup page checks for this file and only enables the RULER run when data is
available. Its score is exposed as `metrics.benchmark.task_accuracy` and can be
compared between runs with the same task, context length, and sample count.

RULERv1-ns data preparation invokes Linux shell commands, so prepare it in the
NeMo-Skills-supported Linux/WSL environment and point the Windows backend at an
accessible copy of the generated setup directory. The backend applies RULERv1's
case-insensitive substring matching to the expected answer. It deliberately
runs the prepared samples through this project's inference engine rather than
`ns eval`, so XKV/XKV-SR is actually active. This smoke run is one task only; it
is not the full 13-task RULER score.

Request validation errors use the shape
`{"code":"VALIDATION_ERROR","message":"...","details":[...]}`. Other API errors
use the same `code`, `message`, and `details` envelope.

Create a run with:

```json
{
  "prompt": "Explain how a KV Cache reduces repeated transformer computation.",
  "method": "xkv",
  "max_new_tokens": 200
}
```

The immediate response has HTTP 202:

```json
{
  "experiment_id": "exp_<uuid>",
  "status": "queued",
  "created_at": "2026-11-15T10:00:00+00:00"
}
```

`GET /api/v1/experiments/{id}/result` is available only after completion.
Values the existing inference engine does not return (for example prompt token
count or total latency) are `null`; metrics are passed through from its actual
result rather than recomputed by the API.
