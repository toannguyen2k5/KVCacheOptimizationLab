import { useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Boxes,
  Check,
  ChevronDown,
  Cpu,
  Database,
  Download,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  Plus,
  RefreshCw,
  Terminal,
  Zap,
} from "lucide-react";
import {
  Link,
  NavLink,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  ExperimentTable,
  formatDate,
  LoadingState,
  MetricCard,
  ModelStatus,
  StatePanel,
  StatusBadge,
} from "./components";
import { useExperiment } from "./hooks/useExperiment";
import { experimentService } from "./services/experimentService";
import { getErrorMessage } from "./services/apiClient";
import { modelService } from "./services/modelService";
import { optimizationService } from "./services/optimizationService";
import { resultService } from "./services/resultService";
import type {
  ConfigOptions,
  Experiment,
  ExperimentConfig,
  ExperimentList,
  ExperimentResult,
  ModelInfo,
  OptimizationMethod,
  OptimizationMethodId,
} from "./types/api";

const navigation = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/analysis", label: "KV Cache Analysis", icon: Database },
  { to: "/setup", label: "Experiment Setup", icon: FlaskConical },
  { to: "/optimization-lab", label: "Optimization Lab", icon: Zap },
  { to: "/evaluation", label: "Evaluation", icon: BarChart3 },
];

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link className="brand" to="/">
          <span className="brand-mark"><Boxes size={21} /></span>
          <span>
            <strong>KV LAB</strong>
            <small>Optimization Studio</small>
          </span>
        </Link>

        <div className="nav-caption">WORKSPACE</div>
        <nav className="primary-nav">
          {navigation.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <span className="note-icon"><Activity size={16} /></span>
            <div>
              <strong>Research workspace</strong>
              <span>Qwen3 · XKV / XKV-SR pipeline</span>
            </div>
          </div>
          <span className="sidebar-version">KV CACHE OPTIMIZATION LAB</span>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span>/</span><strong>Experiments</strong></div>
          <div className="topbar-right">
            <span className="environment-tag"><span /> LOCAL ENVIRONMENT</span>
            <span className="avatar">KL</span>
          </div>
        </header>
        <div className="page-container">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/setup" element={<SetupPage />} />
            <Route path="/optimization-lab" element={<OptimizationLabPage />} />
            <Route path="/optimization-lab/:experimentId" element={<OptimizationLabPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

function PageHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action && <div className="heading-action">{action}</div>}
    </div>
  );
}

function useOverviewData() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [experiments, setExperiments] = useState<ExperimentList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelItems, history] = await Promise.all([
        modelService.list(),
        experimentService.list(),
      ]);
      setModels(modelItems);
      setExperiments(history);
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  return { models, experiments, error, loading, refresh };
}

function OverviewPage() {
  const { models, experiments, error, loading, refresh } = useOverviewData();
  const model = models[0];
  const counts = experiments?.counts ?? {};

  return (
    <>
      <PageHeading
        eyebrow="SYSTEM OVERVIEW"
        title="KV Cache Optimization Lab"
        description="Monitor your model, launch experiments, and inspect real inference metrics."
        action={
          <button className="button button-primary" onClick={() => void refresh()}>
            <RefreshCw size={16} /> Refresh
          </button>
        }
      />
      {error && <StatePanel title="Backend unavailable" message={error} tone="error" action={<button className="button button-secondary" onClick={() => void refresh()}>Try again</button>} />}
      <section className="metric-grid overview-metrics">
        <MetricCard label="Experiments" value={experiments?.total ?? (loading ? "—" : 0)} description="Saved experiment history" icon={<FlaskConical size={17} />} />
        <MetricCard label="Running" value={counts.running ?? 0} description="Currently executing" icon={<Activity size={17} />} />
        <MetricCard label="Completed" value={counts.completed ?? 0} description="Results ready to inspect" icon={<Check size={17} />} />
        <MetricCard label="Model" value={model?.name ?? "Qwen3-8B"} description={model ? `${model.device ?? "Device unknown"} · ${model.dtype ?? "dtype unknown"}` : "Configured model"} icon={<Cpu size={17} />} />
      </section>

      <section className="panel model-panel">
        <div className="panel-title-row">
          <div className="panel-heading">
            <div className="panel-icon"><Cpu size={18} /></div>
            <div><h2>Model & connection</h2><p>Current API-reported runtime status</p></div>
          </div>
          {model && <ModelStatus status={model.status} />}
        </div>
        {loading && !model ? <LoadingState label="Connecting to backend…" /> : (
          <div className="model-details">
            <div><span className="eyebrow">MODEL</span><strong>{model?.name ?? "Qwen3-8B"}</strong></div>
            <div><span className="eyebrow">DEVICE</span><strong>{model?.device ?? "Not reported"}</strong></div>
            <div><span className="eyebrow">PRECISION</span><strong>{model?.dtype ?? "Not reported"}</strong></div>
            <div><span className="eyebrow">API</span><strong className={error ? "text-error" : "text-success"}>{error ? "Offline" : "Connected"}</strong></div>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="section-title-row">
          <div><div className="eyebrow">EXPERIMENT HISTORY</div><h2>Recent experiments</h2></div>
          <Link className="button button-secondary button-small" to="/optimization-lab">View all <ArrowUpRight size={15} /></Link>
        </div>
        {loading && !experiments ? <LoadingState /> : error ? null : (
          <ExperimentTable items={experiments?.items.slice(0, 6) ?? []} emptyMessage="No experiments yet. Start by submitting a prompt." />
        )}
      </section>

      <section className="quick-action">
        <div className="quick-action-icon"><Zap size={19} /></div>
        <div><strong>Ready to run a new experiment?</strong><span>Choose an optimization method and launch a run.</span></div>
        <Link className="button button-primary" to="/setup"><Plus size={16} /> New experiment</Link>
      </section>
    </>
  );
}

function SetupPage() {
  const navigate = useNavigate();
  const [options, setOptions] = useState<ConfigOptions | null>(null);
  const [availableMethods, setAvailableMethods] = useState<OptimizationMethod[]>([]);
  const [selectedMethod, setSelectedMethod] = useState<OptimizationMethodId | "">("");
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(200);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fieldError, setFieldError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      experimentService.options(),
      optimizationService.methods(),
    ]).then(([configOptions, methods]) => {
      setOptions(configOptions);
      const available = methods.filter(
        (method) => method.available && configOptions.methods.includes(method.id),
      );
      setAvailableMethods(available);
      setSelectedMethod((current) =>
        available.some((method) => method.id === current)
          ? current
          : available[0]?.id ?? "",
      );
      setMaxTokens(configOptions.generation.max_new_tokens.default);
    }).catch((requestError: unknown) => setError(getErrorMessage(requestError)))
      .finally(() => setLoadingOptions(false));
  }, []);

  const maxPromptLength = options?.prompt.maximum_length ?? 20_000;
  const selectedMethodDetails = availableMethods.find(
    (method) => method.id === selectedMethod,
  );
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    const trimmed = prompt.trim();
    if (!trimmed) {
      setFieldError("Enter a prompt before launching the experiment.");
      return;
    }
    if (trimmed.length > maxPromptLength) {
      setFieldError(`Prompt must be ${maxPromptLength.toLocaleString()} characters or fewer.`);
      return;
    }
    if (!options || !selectedMethodDetails || !selectedMethod) {
      setError("No optimization method is currently available from the backend.");
      return;
    }
    const config: ExperimentConfig = {
      prompt: trimmed,
      method: selectedMethod,
      max_new_tokens: maxTokens,
    };
    setSubmitting(true);
    setFieldError(null);
    try {
      await experimentService.validate(config);
      const created = await experimentService.create(config);
      navigate(`/optimization-lab/${encodeURIComponent(created.experiment_id)}`);
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  const runRulerSmoke = async () => {
    if (!selectedMethod || !options?.ruler?.available) return;
    setSubmitting(true);
    setError(null);
    try {
      const created = await experimentService.createRuler(selectedMethod);
      navigate(`/optimization-lab/${encodeURIComponent(created.experiment_id)}`);
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <PageHeading eyebrow="NEW RUN" title="Experiment setup" description="Choose an available KV cache optimization method and configure an inference run." />
      {error && <StatePanel title="Unable to prepare experiment" message={error} tone="error" />}
      {loadingOptions ? <LoadingState label="Loading backend options…" /> : (
        <div className="setup-layout">
          <form className="panel setup-form" onSubmit={(event) => void submit(event)}>
            <div className="form-section-heading">
              <span className="step-number">01</span>
              <div><h2>Input prompt</h2><p>Provide the text to send to Qwen3-8B.</p></div>
            </div>
            <label className="field-label" htmlFor="prompt">PROMPT <span className="required-mark">*</span></label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(event) => {
                setPrompt(event.target.value);
                setFieldError(null);
              }}
              maxLength={maxPromptLength + 1}
              placeholder="Ask a question or describe the task for your model…"
              rows={9}
              required
            />
            <div className={`field-meta ${fieldError ? "text-error" : ""}`}>
              <span>{fieldError ?? "Prompt content is sent to your configured local model."}</span>
              <span>{prompt.length.toLocaleString()} / {maxPromptLength.toLocaleString()}</span>
            </div>
            <div className="form-divider" />
            <div className="form-section-heading">
              <span className="step-number">02</span>
              <div><h2>Generation</h2><p>Only parameters accepted by the current inference interface are shown.</p></div>
            </div>
            <div className="form-row">
              <div>
                <label className="field-label" htmlFor="method">OPTIMIZATION METHOD</label>
                <div className="select-wrap">
                  <select
                    id="method"
                    value={selectedMethod}
                    onChange={(event) => {
                      const method = availableMethods.find(
                        (item) => item.id === event.target.value,
                      );
                      if (method) setSelectedMethod(method.id);
                    }}
                    disabled={!availableMethods.length}
                  >
                    {availableMethods.map((method) => (
                      <option key={method.id} value={method.id}>
                        {method.name}
                      </option>
                    ))}
                    {!availableMethods.length && <option value="">Unavailable</option>}
                  </select>
                  <ChevronDown size={16} />
                </div>
                <span className="field-help">
                  {selectedMethodDetails?.description ?? "No optimization methods are currently available."}
                </span>
              </div>
              <div>
                <label className="field-label" htmlFor="max-tokens">MAX NEW TOKENS</label>
                <input
                  id="max-tokens"
                  type="number"
                  min={options?.generation.max_new_tokens.minimum ?? 1}
                  max={options?.generation.max_new_tokens.maximum ?? 1024}
                  value={maxTokens}
                  onChange={(event) => setMaxTokens(Number(event.target.value))}
                  required
                />
                <span className="field-help">Greedy decoding · 1–{options?.generation.max_new_tokens.maximum ?? 1024} tokens.</span>
              </div>
            </div>
            <div className="form-actions">
              <Link className="button button-secondary" to="/">Cancel</Link>
              <button className="button button-primary" type="submit" disabled={submitting || !selectedMethodDetails}>
                {submitting ? <><span className="button-spinner" /> Submitting…</> : <><Zap size={16} /> Run experiment</>}
              </button>
            </div>
          </form>

          <aside className="setup-aside">
            <section className="panel summary-panel">
              <div className="eyebrow">RUN SUMMARY</div>
              <h3>Configuration</h3>
              <div className="summary-row"><span>Model</span><strong>Qwen3-8B</strong></div>
              <div className="summary-row"><span>Method</span><strong>{selectedMethodDetails?.name ?? "Unavailable"}</strong></div>
              <div className="summary-row"><span>Max new tokens</span><strong>{maxTokens || "—"}</strong></div>
              <div className="summary-row"><span>Prompt length</span><strong>{prompt.length.toLocaleString()} chars</strong></div>
              <div className="summary-note"><Gauge size={16} /><span>Generation runs in the backend worker. The API returns immediately.</span></div>
            </section>
            <section className="info-card">
              <div className="info-card-icon"><Database size={16} /></div>
              <strong>Local inference</strong>
              <p>Model loading is deferred until the first experiment. Browsing the dashboard does not load model weights.</p>
            </section>
            <section className="panel summary-panel">
              <div className="eyebrow">RULERv1-ns SMOKE TEST</div>
              <h3>NIAH · 4K context</h3>
              <p className="field-help">Runs up to 10 prepared samples with the selected optimization method.</p>
              {options?.ruler?.available ? (
                <>
                  <div className="summary-row"><span>Prepared samples</span><strong>{Math.min(options.ruler.sample_limit, options.ruler.available_samples)}</strong></div>
                  <button className="button button-secondary" type="button" onClick={() => void runRulerSmoke()} disabled={submitting || !selectedMethod}>
                    {submitting ? <><span className="button-spinner" /> Starting…</> : "Run RULER smoke test"}
                  </button>
                </>
              ) : (
                <p className="field-help">
                  Prepare RULERv1-ns data and set <code>{options?.ruler?.data_dir_env ?? "KV_CACHE_RULER_DATA_DIR"}</code> to the setup folder containing <code>niah_single_1\test.jsonl</code>. See the backend API README.
                </p>
              )}
            </section>
          </aside>
        </div>
      )}
    </>
  );
}

function OptimizationLabPage() {
  const { experimentId } = useParams();
  const [items, setItems] = useState<Experiment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!experimentId);
  const detail = useExperiment(experimentId);
  const progressValue = detail.events
    .filter((event) => event.event === "progress")
    .map((event) => event.data.progress)
    .find((value): value is number => typeof value === "number");

  useEffect(() => {
    if (experimentId) return;
    experimentService.list(1, 100).then((response) => setItems(response.items))
      .catch((requestError: unknown) => setError(getErrorMessage(requestError)))
      .finally(() => setLoading(false));
  }, [experimentId]);

  if (!experimentId) {
    return (
      <>
        <PageHeading eyebrow="LIVE MONITOR" title="Optimization lab" description="Review experiment state and open a run to inspect its live events." action={<Link className="button button-primary" to="/setup"><Plus size={16} /> New experiment</Link>} />
        {error && <StatePanel title="Could not load experiment history" message={error} tone="error" />}
        {loading ? <LoadingState /> : <section className="panel"><ExperimentTable items={items} /></section>}
      </>
    );
  }

  return (
    <>
      <PageHeading
        eyebrow="EXPERIMENT MONITOR"
        title="Run details"
        description={experimentId}
        action={<span className={`connection-pill ${detail.connected ? "connected" : ""}`}><span />{detail.connected ? "Live updates connected" : "Reconnecting…"}</span>}
      />
      {detail.error && <StatePanel title="Experiment error" message={detail.error} tone="error" />}
      {!detail.experiment && !detail.error ? <LoadingState label="Connecting to experiment…" /> : detail.experiment && (
        <>
          <section className="panel run-overview">
            <div className="run-overview-status"><StatusBadge status={detail.experiment.status} /><span>Created {formatDate(detail.experiment.created_at)}</span></div>
            <div className="run-config"><span>METHOD <strong>{detail.experiment.config.method.toUpperCase()}</strong></span><span>MAX TOKENS <strong>{detail.experiment.config.max_new_tokens}</strong></span></div>
            {progressValue !== undefined && <div className="progress-track"><span style={{ width: `${Math.min(100, Math.max(0, progressValue))}%` }} /></div>}
            <p className="run-status-message">{statusMessage(detail.experiment.status)}</p>
          </section>
          <div className="monitor-grid">
            <section className="panel">
              <div className="section-title-row"><div><div className="eyebrow">OUTPUT</div><h2>Generated text</h2></div><Terminal size={18} className="muted" /></div>
              {detail.result?.inference.generated_text ? <pre className="generated-output">{detail.result.inference.generated_text}</pre> : <div className="pending-output">{detail.experiment.status === "failed" ? "No result was produced." : "Output will appear when inference completes."}</div>}
              {detail.result && <div className="output-meta">{detail.result.inference.generated_tokens ?? "—"} generated tokens</div>}
            </section>
            <section className="panel">
              <div className="section-title-row"><div><div className="eyebrow">EVENT STREAM</div><h2>Run log</h2></div><span className="log-count">{detail.logs.length} / 500</span></div>
              <div className="log-viewer">
                {detail.logs.length ? detail.logs.map((log, index) => <div className="log-entry" key={`${log.timestamp}-${index}`}><time>{new Date(log.timestamp).toLocaleTimeString()}</time><span>{log.message}</span></div>) : <span className="muted">Waiting for backend events…</span>}
              </div>
            </section>
          </div>
          {detail.result && <ResultMetrics result={detail.result} />}
        </>
      )}
      <Link className="back-link" to="/optimization-lab">← Back to experiment history</Link>
    </>
  );
}

function statusMessage(status: Experiment["status"]): string {
  const messages = {
    queued: "Your experiment is waiting for the inference worker.",
    running: "The model is generating a response. This may take a while.",
    completed: "Inference completed. Results are ready.",
    failed: "The backend could not complete this experiment.",
    cancelled: "This experiment was cancelled.",
  };
  return messages[status];
}

function ResultMetrics({ result }: { result: ExperimentResult }) {
  const memory = result.metrics.memory ?? {};
  const compression = result.metrics.compression ?? {};
  const reconstruction = result.metrics.reconstruction ?? {};
  const benchmark = result.metrics.benchmark ?? {};
  const isSelective = reconstruction.mse_scope === "query_selected_prompt_tokens";
  return (
    <section className="panel results-strip">
      <div className="section-title-row"><div><div className="eyebrow">MEASURED OUTPUT</div><h2>Available metrics</h2></div><Link className="text-link" to={`/analysis?experiment=${encodeURIComponent(result.experiment_id)}`}>Open analysis <ArrowUpRight size={14} /></Link></div>
      <div className="metric-grid four">
        <MetricCard label="Cache before" value={numberValue(memory.memory_raw_cache)} unit="MB" />
        <MetricCard label="Cache after" value={numberValue(memory.memory_compressed_cache)} unit="MB" />
        <MetricCard label="Compression ratio" value={numberValue(compression.compression_raito)} unit="×" />
        <MetricCard label={isSelective ? "SR reconstruction" : "Reconstruction"} value={numberValue(reconstruction.reconstruction_time)} unit="s" />
        {isSelective && <MetricCard label="Prompt tokens used" value={percent(numberValue(reconstruction.selected_prompt_token_fraction))} unit="%" description="Average fraction retrieved during decode" />}
        {benchmark.benchmark === "RULERv1-ns" && <MetricCard label="RULER task accuracy" value={percent(numberValue(benchmark.task_accuracy))} unit="%" />}
      </div>
    </section>
  );
}

function AnalysisPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedId, setSelectedId] = useState(new URLSearchParams(window.location.search).get("experiment") ?? "");
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    experimentService.list(1, 100).then((response) => {
      const completed = response.items.filter((item) => item.status === "completed");
      setExperiments(completed);
      if (!selectedId && completed[0]) setSelectedId(completed[0].experiment_id);
    }).catch((requestError: unknown) => setError(getErrorMessage(requestError)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setResult(null);
      return;
    }
    resultService.get(selectedId).then(setResult)
      .catch((requestError: unknown) => setError(getErrorMessage(requestError)));
  }, [selectedId]);

  const memory = result?.metrics.memory ?? {};
  const compression = result?.metrics.compression ?? {};
  const reconstruction = result?.metrics.reconstruction ?? {};
  const accuracy = result?.metrics.accuracy ?? {};
  const benchmark = result?.metrics.benchmark ?? {};
  const isSelective = reconstruction.mse_scope === "query_selected_prompt_tokens";
  const memoryChart = [
    { name: "Original", value: numberValue(memory.memory_raw_cache) ?? 0 },
    { name: "Optimized", value: numberValue(memory.memory_compressed_cache) ?? 0 },
  ];
  const chartMax = Math.max(...memoryChart.map((item) => item.value), 1);

  return (
    <>
      <PageHeading eyebrow="CACHE INSPECTION" title="KV Cache analysis" description="Inspect cache memory and reconstruction metrics reported by the backend." />
      {error && <StatePanel title="Could not load analysis" message={error} tone="error" />}
      {loading ? <LoadingState /> : !experiments.length ? (
        <StatePanel title="No completed experiments" message="Complete an experiment to view its KV Cache measurements." action={<Link className="button button-primary" to="/setup">Create experiment</Link>} />
      ) : (
        <>
          <div className="analysis-toolbar panel">
            <div><span className="eyebrow">COMPLETED EXPERIMENT</span><p>Choose a run with a completed result.</p></div>
            <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
              {experiments.map((item) => <option key={item.experiment_id} value={item.experiment_id}>{item.experiment_id} · {formatDate(item.completed_at)}</option>)}
            </select>
          </div>
          {result && (
            <>
              <div className="analysis-prompt panel"><span className="eyebrow">PROMPT</span><p>{experiments.find((item) => item.experiment_id === selectedId)?.config.prompt}</p></div>
              <div className="analysis-grid">
                <div className="analysis-prompt panel"><span className="eyebrow">OPTIMIZED OUTPUT</span><p>{result.inference.generated_text ?? "Not provided"}</p></div>
                <div className="analysis-prompt panel"><span className="eyebrow">FULL-CACHE REFERENCE</span><p>{result.inference.reference_generated_text ?? "Not provided"}</p></div>
              </div>
              <div className="metric-grid four">
                <MetricCard label="Original cache" value={numberValue(memory.memory_raw_cache)} unit="MB" description="KV Cache before compression" />
                <MetricCard label="Compressed cache" value={numberValue(memory.memory_compressed_cache)} unit="MB" description="Storage reported by XKV optimizer" />
                <MetricCard label="Memory saved" value={numberValue(memory.memory_saved_cache)} unit="MB" description="Absolute memory difference" />
                <MetricCard label="Memory reduction" value={percent(numberValue(memory.memory_reduction_cache))} unit="%" description="Backend-reported reduction ratio" />
              </div>
              <div className="analysis-grid">
                <section className="panel chart-panel">
                  <div className="section-title-row"><div><div className="eyebrow">CACHE FOOTPRINT</div><h2>Memory comparison</h2></div><span className="chart-unit">MB</span></div>
                  <div className="chart-container bar-chart">
                    {memoryChart.map((item, index) => (
                      <div className="bar-row" key={item.name} title={`${item.name}: ${item.value} MB`}>
                        <div className="bar-label"><span>{item.name}</span><strong>{item.value.toLocaleString(undefined, { maximumFractionDigits: 3 })} MB</strong></div>
                        <div className="bar-rail"><span className={index === 0 ? "bar-original" : "bar-optimized"} style={{ width: `${(item.value / chartMax) * 100}%` }} /></div>
                      </div>
                    ))}
                  </div>
                </section>
                <section className="panel">
                  <div className="section-title-row"><div><div className="eyebrow">COMPRESSION</div><h2>Run metrics</h2></div></div>
                  <div className="detail-list">
                    <DetailRow label="Compression ratio" value={formatted(numberValue(compression.compression_raito), "×")} />
                    <DetailRow label="Compression time" value={formatted(numberValue(compression.compression_time), "s")} />
                    <DetailRow label={isSelective ? "SR reconstruction time" : "Reconstruction time"} value={formatted(numberValue(reconstruction.reconstruction_time), "s")} />
                    {isSelective && <DetailRow label="Prompt tokens used" value={formatted(percent(numberValue(reconstruction.selected_prompt_token_fraction)), "%")} />}
                    <DetailRow label="Peak memory" value={formatted(numberValue(memory.peak_memory), "MB")} />
                    <DetailRow label="Generated tokens" value={result.inference.generated_tokens ?? "Not provided"} />
                    <DetailRow label="Reference generation time" value={formatted(numberValue(accuracy.reference_generation_time), "s")} />
                    {typeof benchmark.task === "string" && <DetailRow label="RULER task" value={benchmark.task} />}
                    {numberValue(benchmark.sample_count) !== null && <DetailRow label="RULER samples" value={numberValue(benchmark.sample_count) ?? 0} />}
                  </div>
                </section>
              </div>
              {benchmark.benchmark === "RULERv1-ns" && (
                <section className="panel">
                  <div className="section-title-row"><div><div className="eyebrow">RULERv1-ns TASK SCORE</div><h2>{String(benchmark.task)} accuracy</h2></div></div>
                  <div className="metric-grid four">
                    <MetricCard label="Task accuracy" value={percent(numberValue(benchmark.task_accuracy))} unit="%" description="Official RULERv1 match scoring for this task" />
                    <MetricCard label="Fully correct samples" value={numberValue(benchmark.correct_samples)} />
                    <MetricCard label="Samples evaluated" value={numberValue(benchmark.sample_count)} />
                    <MetricCard label="Context length" value={numberValue(benchmark.context_length)} unit="tokens" />
                  </div>
                  <p className="data-note">This smoke test covers only one NIAH task and is not the full 13-task RULER aggregate.</p>
                </section>
              )}
              <section className="panel">
                <div className="section-title-row"><div><div className="eyebrow">FULL-CACHE REFERENCE AGREEMENT</div><h2>Generation accuracy</h2></div></div>
                <div className="metric-grid four">
                  <MetricCard label="Greedy token agreement" value={percent(numberValue(accuracy.token_agreement))} unit="%" description="Generated tokens matching the uncompressed full-cache output" />
                  <MetricCard
                    label="Exact sequence match"
                    value={accuracy.exact_sequence_match === true ? "Yes" : accuracy.exact_sequence_match === false ? "No" : null}
                    description="Includes the end-of-sequence position"
                  />
                  <MetricCard label="Tokens compared" value={numberValue(accuracy.compared_tokens)} />
                </div>
                <p className="data-note">This measures agreement with the same model using its full KV cache; it is not correctness against a ground-truth answer. The reference decode adds inference time.</p>
              </section>
              <section className="panel">
                <div className="section-title-row"><div><div className="eyebrow">RECONSTRUCTION QUALITY</div><h2>Quality metrics</h2></div></div>
                <div className="metric-grid four">
                  <MetricCard label={isSelective ? "Selected key MSE" : "Key MSE"} value={numberValue(reconstruction.key_mse)} description={isSelective ? "MSE on prompt keys retrieved during autoregressive decode" : "Mean squared reconstruction error"} />
                  <MetricCard label={isSelective ? "Selected value MSE" : "Value MSE"} value={numberValue(reconstruction.value_mse)} description={isSelective ? "MSE on prompt values retrieved during autoregressive decode" : "Mean squared reconstruction error"} />
                  <MetricCard label="Key relative error" value={numberValue(reconstruction.key_relative_error)} />
                  <MetricCard label="Value relative error" value={numberValue(reconstruction.value_relative_error)} />
                </div>
                <p className="data-note">Layer-level shapes, head metadata and grouping are not included by the current inference result.</p>
              </section>
            </>
          )}
        </>
      )}
    </>
  );
}

function EvaluationPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [results, setResults] = useState<ExperimentResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    experimentService.list(1, 100).then((response) => setExperiments(response.items.filter((item) => item.status === "completed")))
      .catch((requestError: unknown) => setError(getErrorMessage(requestError)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all(selected.map((id) => resultService.get(id)))
      .then((items) => { if (active) setResults(items); })
      .catch((requestError: unknown) => { if (active) setError(getErrorMessage(requestError)); });
    return () => { active = false; };
  }, [selected]);

  const metrics = useMemo(() => sharedMetrics(results), [results]);
  const changedPrompts = selected.length > 1 && new Set(
    selected.map((id) => experiments.find((item) => item.experiment_id === id)?.config.prompt),
  ).size > 1;
  const toggle = (id: string) => {
    setSelected((current) => current.includes(id)
      ? current.filter((value) => value !== id)
      : current.length < 4 ? [...current, id] : current);
  };
  const exportResults = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "kv-cache-experiment-comparison.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <PageHeading eyebrow="RESULT COMPARISON" title="Evaluation" description="Compare metrics with matching definitions across completed experiments." action={results.length > 0 && <button className="button button-secondary" onClick={exportResults}><Download size={16} /> Export JSON</button>} />
      {error && <StatePanel title="Could not load results" message={error} tone="error" />}
      {loading ? <LoadingState /> : !experiments.length ? <StatePanel title="Nothing to compare yet" message="Completed experiments appear here when their results are ready." action={<Link className="button button-primary" to="/setup">Create experiment</Link>} /> : (
        <div className="evaluation-layout">
          <section className="panel experiment-picker">
            <div className="section-title-row"><div><div className="eyebrow">EXPERIMENTS</div><h2>Select 2–4 runs</h2></div><span className="selection-count">{selected.length} selected</span></div>
            <div className="picker-list">
              {experiments.map((experiment) => (
                <label key={experiment.experiment_id} className={`picker-item ${selected.includes(experiment.experiment_id) ? "selected" : ""}`}>
                  <input type="checkbox" checked={selected.includes(experiment.experiment_id)} onChange={() => toggle(experiment.experiment_id)} disabled={!selected.includes(experiment.experiment_id) && selected.length >= 4} />
                  <span className="picker-check">{selected.includes(experiment.experiment_id) && <Check size={13} />}</span>
                  <span className="picker-copy"><strong>{experiment.experiment_id}</strong><small>{formatDate(experiment.completed_at)} · {experiment.config.method.toUpperCase()}</small></span>
                </label>
              ))}
            </div>
          </section>
          <section className="panel comparison-panel">
            <div className="section-title-row"><div><div className="eyebrow">COMPARISON TABLE</div><h2>Measured metrics</h2></div></div>
            {selected.length < 2 ? <StatePanel title="Choose at least two experiments" message="Select two or more completed runs to see comparable metrics." /> : (
              <>
                {changedPrompts && <div className="warning-banner"><AlertTriangleIcon /> The selected runs use different prompts. Compare results with that context in mind.</div>}
                {!metrics.length ? <StatePanel title="No shared numeric metrics" message="These experiments do not share numeric metrics with matching units." /> : (
                  <div className="table-wrap">
                    <table className="comparison-table">
                      <thead><tr><th>Metric</th>{results.map((result) => <th key={result.experiment_id}>{result.experiment_id}</th>)}</tr></thead>
                      <tbody>{metrics.map((metric) => <tr key={metric.key}><td>{metric.label}<span>{metric.unit}</span></td>{results.map((result) => <td key={result.experiment_id}>{new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(metric.values[result.experiment_id])}</td>)}</tr>)}</tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </section>
        </div>
      )}
    </>
  );
}

function AlertTriangleIcon() {
  return <span className="warning-mark">!</span>;
}

function sharedMetrics(results: ExperimentResult[]) {
  if (results.length < 2) return [];
  const reconstructionScopes = new Set(
    results.map(
      (result) =>
        result.metrics.reconstruction?.mse_scope ?? "dense_full_cache",
    ),
  );
  const comparableReconstruction = reconstructionScopes.size === 1;
  const isSelectiveReconstruction =
    reconstructionScopes.has("query_selected_prompt_tokens");
  const benchmarkKeys = results.map((result) => {
    const benchmark = result.metrics.benchmark;
    if (!benchmark) return null;
    return [
      benchmark.benchmark,
      benchmark.task,
      benchmark.context_length,
      benchmark.sample_count,
    ].join(":");
  });
  const comparableBenchmark =
    benchmarkKeys.every((key): key is string => key !== null) &&
    new Set(benchmarkKeys).size === 1;
  const candidates = [
    { label: "Original cache size", unit: "MB", key: "memory_raw_cache", group: "memory" },
    { label: "Compressed cache size", unit: "MB", key: "memory_compressed_cache", group: "memory" },
    { label: "Memory saved", unit: "MB", key: "memory_saved_cache", group: "memory" },
    { label: "Memory reduction", unit: "%", key: "memory_reduction_cache", group: "memory", percent: true },
    { label: "Compression ratio", unit: "×", key: "compression_raito", group: "compression" },
    { label: "Compression time", unit: "s", key: "compression_time", group: "compression" },
    { label: isSelectiveReconstruction ? "SR reconstruction time" : "Reconstruction time", unit: "s", key: "reconstruction_time", group: "reconstruction" },
    { label: isSelectiveReconstruction ? "Selected key MSE" : "Key MSE", unit: "", key: "key_mse", group: "reconstruction" },
    { label: isSelectiveReconstruction ? "Selected value MSE" : "Value MSE", unit: "", key: "value_mse", group: "reconstruction" },
    { label: "Full-cache token agreement", unit: "%", key: "token_agreement", group: "accuracy", percent: true },
    { label: "RULER task accuracy", unit: "%", key: "task_accuracy", group: "benchmark", percent: true },
  ];
  return candidates.flatMap((candidate) => {
    if (candidate.group === "reconstruction" && !comparableReconstruction) {
      return [];
    }
    if (candidate.group === "benchmark" && !comparableBenchmark) {
      return [];
    }
    const values: Record<string, number> = {};
    for (const result of results) {
      const source = result.metrics[candidate.group as keyof ExperimentResult["metrics"]];
      const rawValue = source?.[candidate.key];
      if (typeof rawValue !== "number" || !Number.isFinite(rawValue)) return [];
      values[result.experiment_id] = candidate.percent ? rawValue * 100 : rawValue;
    }
    return [{ ...candidate, values }];
  });
}

function DetailRow({ label, value }: { label: string; value: string | number }) {
  return <div className="detail-row"><span>{label}</span><strong>{value}</strong></div>;
}

function NotFoundPage() {
  return <StatePanel title="Page not found" message="This route is not part of the lab workspace." action={<Link className="button button-primary" to="/">Back to overview</Link>} />;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function percent(value: number | null): number | null {
  return value === null ? null : value * 100;
}

function formatted(value: number | null, unit: string): string {
  return value === null ? "Not provided" : `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value)} ${unit}`;
}