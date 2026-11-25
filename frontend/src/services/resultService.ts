import { apiClient } from "./apiClient";
import type { ExperimentResult } from "../types/api";

export const resultService = {
  async get(experimentId: string): Promise<ExperimentResult> {
    const response = await apiClient.get<ExperimentResult>(
      `/experiments/${encodeURIComponent(experimentId)}/result`,
    );
    return response.data;
  },
};
