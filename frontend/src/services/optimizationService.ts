import { apiClient } from "./apiClient";
import type { OptimizationMethod } from "../types/api";

export const optimizationService = {
  async methods(): Promise<OptimizationMethod[]> {
    const response = await apiClient.get<{ items: OptimizationMethod[] }>(
      "/optimization/methods",
    );
    return response.data.items;
  },
};
