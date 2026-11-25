export type ExperimentStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type OptimizationMethodId = "xkv" | "xkv_sr";

export interface ApiError {
  code: string;
  message: string;
  details: unknown;
}

export interface ModelInfo {
  id: string;
  name: string;
  status: "loaded" | "not_loaded" | "loading" | "error";
  device: string | null;
  dtype: string | null;
  error?: string | null;
}

export interface ExperimentConfig {
  prompt: string;
  method: OptimizationMethodId;
  max_new_tokens: number;
  benchmark?: "ruler_v1_ns";
  task?: string;
  context_length?: number;
  sample_limit?: number;
}

export interface Experiment {
  experiment_id: string;
  status: ExperimentStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  config: ExperimentConfig;
  error_message: string | null;
}

export interface ExperimentList {
  items: Experiment[];
  total: number;
  page: number;
  page_size: number;
  counts: Partial<Record<ExperimentStatus, number>>;
}

export interface ExperimentResult {
  experiment_id: string;
  inference: {
    generated_text: string | null;
    reference_generated_text: string | null;
    prompt_tokens: number | null;
    generated_tokens: number | null;
    latency_seconds: number | null;
  };
  metrics: {
    memory: Record<string, unknown> | null;
    compression: Record<string, unknown> | null;
    reconstruction: Record<string, unknown> | null;
    accuracy: Record<string, unknown> | null;
    benchmark: Record<string, unknown> | null;
  };
}

export interface ExperimentLog {
  timestamp: string;
  message: string;
}

export interface ExperimentEvent {
  event: "status" | "progress" | "log" | "result_ready" | "error";
  experiment_id: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export interface ConfigOptions {
  generation: {
    max_new_tokens: { minimum: number; maximum: number; default: number };
    temperature: null;
    sampling: null;
  };
  methods: string[];
  prompt: { minimum_length: number; maximum_length: number };
  ruler?: {
    available: boolean;
    task: string;
    context_length: number;
    sample_limit: number;
    max_new_tokens: number;
    data_dir_env: string;
    available_samples: number;
  };
}

export interface OptimizationMethod {
  id: OptimizationMethodId;
  name: string;
  description: string;
  available: boolean;
  parameters: Array<Record<string, unknown>>;
}
