import axios from "axios";
import type { ApiError } from "../types/api";

const baseURL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export const apiClient = axios.create({
  baseURL,
  timeout: 15_000,
  headers: { "Content-Type": "application/json" },
});

export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError<ApiError>(error)) {
    const responseMessage = error.response?.data?.message;
    return responseMessage || error.message || "The API request failed.";
  }
  if (error instanceof Error) return error.message;
  return "An unexpected error occurred.";
}

export function getWebSocketUrl(): string {
  const explicitBase = import.meta.env.VITE_WS_BASE_URL as string | undefined;
  if (explicitBase) return explicitBase.replace(/^http/, "ws").replace(/\/+$/, "");

  const apiUrl = new URL(baseURL, window.location.origin);
  const protocol = apiUrl.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${apiUrl.host}`;
}

