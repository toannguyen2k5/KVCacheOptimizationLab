from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.ruler_benchmark import (
    RULER_CONTEXT_LENGTH,
    RULER_MAX_NEW_TOKENS,
    RULER_SAMPLE_LIMIT,
    RULER_TASK,
)


ExperimentStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
OptimizationMethodId = Literal["xkv", "xkv_sr"]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(APIModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    method: OptimizationMethodId = "xkv"
    max_new_tokens: int = Field(default=200, ge=1, le=1024)

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must contain non-whitespace characters.")
        return value


class RulerBenchmarkConfig(ExperimentConfig):
    prompt: str = "RULERv1-ns NIAH smoke test (4K, up to 10 samples)"
    benchmark: Literal["ruler_v1_ns"] = "ruler_v1_ns"
    task: Literal["niah_single_1"] = RULER_TASK
    context_length: Literal[4096] = RULER_CONTEXT_LENGTH
    sample_limit: int = Field(default=RULER_SAMPLE_LIMIT, ge=1, le=RULER_SAMPLE_LIMIT)
    max_new_tokens: int = Field(default=RULER_MAX_NEW_TOKENS, ge=1, le=1024)


class OptimizationValidationRequest(APIModel):
    method_id: OptimizationMethodId
    parameters: dict[str, object] = Field(default_factory=dict)


class ExperimentCreateResponse(APIModel):
    experiment_id: str
    status: ExperimentStatus
    created_at: str


class ExperimentListResponse(APIModel):
    items: list[dict[str, object]]
    total: int
    page: int
    page_size: int
    counts: dict[str, int]
