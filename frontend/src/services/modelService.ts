import { apiClient } from "./apiClient";
import type { ModelInfo } from "../types/api";

export const modelService = {
  async list(): Promise<ModelInfo[]> {
    const response = await apiClient.get<{ items: ModelInfo[] }>("/models");
    return response.data.items;
  },
};
