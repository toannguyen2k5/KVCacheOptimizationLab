import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  Clock3,
  LoaderCircle,
} from "lucide-react";
import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import type {
  Experiment,
  ExperimentStatus,
} from "./types/api";

const statusDetails: Record<
  ExperimentStatus,
  { label: string; icon: typeof Clock3 }
> = {
  queued: { label: "Queued", icon: Clock3 },
  running: { label: "Running", icon: LoaderCircle },
  completed: { label: "Completed", icon: CheckCircle2 },
  failed: { label: "Failed", icon: AlertCircle },
  cancelled: { label: "Cancelled", icon: CircleDashed },
};

export function StatusBadge({ status }: { status: ExperimentStatus }) {
  const { label, icon: Icon } = statusDetails[status];
  return (
    <span className={`status-badge status-${status}`}>
      <Icon size={14} className={status === "running" ? "spin" : undefined} />
      {label}
    </span>
  );
}

export function ModelStatus({ status }: { status: string }) {
  const active = status === "loaded";
  return (
    <span className={`model-status ${active ? "is-ready" : ""}`}>
      <span className="status-dot" />
      {status.replace("_", " ")}
    </span>
  );
}

export function MetricCard({
  label,
  value,
  unit,
  description,
  icon,
}: {
  label: string;
  value: number | string | null | undefined;
  unit?: string;
  description?: string;
  icon?: ReactNode;
}) {
  const formatted =
    typeof value === "number"
      ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(
          value,
        )
      : value;
  return (
    <article className="metric-card">
      <div className="metric-card-top">
        <span className="eyebrow">{label}</span>
        {icon && <span className="metric-icon">{icon}</span>}
      </div>
      <div className="metric-value">
        {formatted ?? <span className="muted">Not provided</span>}
        {formatted !== null && formatted !== undefined && unit && (
          <span className="metric-unit">{unit}</span>
        )}
      </div>
      {description && <p className="metric-description">{description}</p>}
    </article>
  );
}

export function StatePanel({
  title,
  message,
  tone = "neutral",
  action,
}: {
  title: string;
  message: string;
  tone?: "neutral" | "error";
  action?: ReactNode;
}) {
  return (
    <div className={`state-panel ${tone}`}>
      <div className="state-symbol">
        {tone === "error" ? <AlertCircle size={20} /> : <CircleDashed size={20} />}
      </div>
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
        {action && <div className="state-action">{action}</div>}
      </div>
    </div>
  );
}

export function ExperimentTable({
  items,
  emptyMessage = "Experiments you run will appear here.",
}: {
  items: Experiment[];
  emptyMessage?: string;
}) {
  if (!items.length) {
    return (
      <div className="table-empty">
        <p>{emptyMessage}</p>
        <Link className="text-link" to="/setup">
          Create an experiment
        </Link>
      </div>
    );
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Method</th>
            <th>Created</th>
            <th>Status</th>
            <th aria-label="Open experiment" />
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.experiment_id}>
              <td>
                <Link
                  className="experiment-link"
                  to={`/optimization-lab/${encodeURIComponent(item.experiment_id)}`}
                >
                  {item.experiment_id}
                </Link>
                <span className="table-subtitle">
                  {item.config.prompt.slice(0, 76)}
                  {item.config.prompt.length > 76 ? "…" : ""}
                </span>
              </td>
              <td><span className="method-chip">{item.config.method}</span></td>
              <td>{formatDate(item.created_at)}</td>
              <td><StatusBadge status={item.status} /></td>
              <td>
                <Link
                  aria-label={`Open ${item.experiment_id}`}
                  className="table-arrow"
                  to={`/optimization-lab/${encodeURIComponent(item.experiment_id)}`}
                >
                  →
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function LoadingState({ label = "Loading data…" }: { label?: string }) {
  return (
    <div className="loading-state">
      <LoaderCircle size={18} className="spin" />
      <span>{label}</span>
    </div>
  );
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "—"
    : date.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
}
