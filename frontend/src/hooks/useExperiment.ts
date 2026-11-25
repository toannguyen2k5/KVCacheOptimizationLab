import { useEffect, useState } from "react";
import { experimentService } from "../services/experimentService";
import { resultService } from "../services/resultService";
import { getErrorMessage } from "../services/apiClient";
import type { Experiment, ExperimentLog, ExperimentResult } from "../types/api";
import { useExperimentSocket } from "./useExperimentSocket";

export function useExperiment(experimentId: string | undefined) {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [logs, setLogs] = useState<ExperimentLog[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { events, connected } = useExperimentSocket(experimentId);

  useEffect(() => {
    if (!experimentId) return;
    let active = true;
    let timer: number | undefined;

    const refresh = async () => {
      try {
        const current = await experimentService.get(experimentId);
        if (!active) return;
        setExperiment(current);
        setError(current.error_message);
        if (current.status === "completed") {
          const completedResult = await resultService.get(experimentId);
          if (active) setResult(completedResult);
        }
        const currentLogs = await experimentService.logs(experimentId);
        if (active) setLogs(currentLogs);
        if (!["completed", "failed", "cancelled"].includes(current.status)) {
          timer = window.setTimeout(refresh, 2_000);
        }
      } catch (requestError) {
        if (active) setError(getErrorMessage(requestError));
      }
    };

    void refresh();
    return () => {
      active = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [experimentId]);

  useEffect(() => {
    const newLogs = events
      .filter((event) => event.event === "log")
      .map((event) => event.data as unknown as ExperimentLog);
    if (newLogs.length) setLogs((current) => [...current, ...newLogs].slice(-500));
  }, [events]);

  return { experiment, result, logs, error, connected, events };
}
