import { apiClient, getWebSocketUrl } from "./apiClient";
import type {
  ConfigOptions,
  Experiment,
  ExperimentConfig,
  ExperimentEvent,
  ExperimentList,
  ExperimentLog,
} from "../types/api";

export const experimentService = {
  async options(): Promise<ConfigOptions> {
    const response = await apiClient.get<ConfigOptions>(
      "/experiments/config-options",
    );
    return response.data;
  },

  async list(page = 1, pageSize = 50): Promise<ExperimentList> {
    const response = await apiClient.get<ExperimentList>("/experiments", {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  async get(experimentId: string): Promise<Experiment> {
    const response = await apiClient.get<Experiment>(
      `/experiments/${encodeURIComponent(experimentId)}`,
    );
    return response.data;
  },

  async validate(config: ExperimentConfig): Promise<void> {
    await apiClient.post("/experiments/validate", config);
  },

  async create(config: ExperimentConfig): Promise<{
    experiment_id: string;
    status: Experiment["status"];
    created_at: string;
  }> {
    const response = await apiClient.post("/experiments", config);
    return response.data;
  },

  async createRuler(method: ExperimentConfig["method"]): Promise<{
    experiment_id: string;
    status: Experiment["status"];
    created_at: string;
  }> {
    const response = await apiClient.post("/experiments/ruler", { method });
    return response.data;
  },

  async logs(experimentId: string): Promise<ExperimentLog[]> {
    const response = await apiClient.get<{ items: ExperimentLog[] }>(
      `/experiments/${encodeURIComponent(experimentId)}/logs`,
    );
    return response.data.items;
  },

  socketUrl(experimentId: string): string {
    return `${getWebSocketUrl()}/ws/experiments/${encodeURIComponent(experimentId)}`;
  },

  parseEvent(raw: MessageEvent<string>): ExperimentEvent | null {
    try {
      return JSON.parse(raw.data) as ExperimentEvent;
    } catch {
      return null;
    }
  },
};
