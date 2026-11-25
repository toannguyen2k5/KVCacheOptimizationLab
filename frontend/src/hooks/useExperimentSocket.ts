import { useEffect, useState } from "react";
import { experimentService } from "../services/experimentService";
import type { ExperimentEvent } from "../types/api";

export function useExperimentSocket(experimentId: string | undefined) {
  const [events, setEvents] = useState<ExperimentEvent[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!experimentId) return;

    let active = true;
    let socket: WebSocket | undefined;
    let reconnectTimer: number | undefined;
    let attempts = 0;
    let terminal = false;

    const connect = () => {
      if (!active) return;
      socket = new WebSocket(experimentService.socketUrl(experimentId));
      socket.onopen = () => {
        attempts = 0;
        if (active) setConnected(true);
      };
      socket.onmessage = (message: MessageEvent<string>) => {
        const event = experimentService.parseEvent(message);
        if (event && active) {
          if (
            event.event === "result_ready" ||
            event.event === "error" ||
            (event.event === "status" &&
              ["completed", "failed", "cancelled"].includes(
                String(event.data.status),
              ))
          ) {
            terminal = true;
          }
          setEvents((current) => [...current, event].slice(-500));
        }
      };
      socket.onclose = () => {
        if (!active) return;
        setConnected(false);
        if (terminal) return;
        const delay = Math.min(1_000 * 2 ** attempts, 15_000);
        attempts += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      active = false;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [experimentId]);

  return { events, connected };
}
