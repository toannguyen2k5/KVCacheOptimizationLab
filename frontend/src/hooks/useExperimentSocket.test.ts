import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useExperimentSocket } from "./useExperimentSocket";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  close = vi.fn(() => this.drop());

  constructor(readonly url: string) {
    MockWebSocket.instances.push(this);
  }

  open() {
    this.onopen?.({} as Event);
  }

  sendEvent(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent<string>);
  }

  drop() {
    this.onclose?.({} as CloseEvent);
  }
}

describe("useExperimentSocket", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("tracks connection state and parses backend events", () => {
    const { result, unmount } = renderHook(() => useExperimentSocket("exp_test"));
    const socket = MockWebSocket.instances[0];

    act(() => {
      socket.open();
      socket.sendEvent({
        event: "status",
        experiment_id: "exp_test",
        timestamp: "2026-01-01T00:00:00Z",
        data: { status: "running" },
      });
    });

    expect(result.current.connected).toBe(true);
    expect(result.current.events[0].data.status).toBe("running");
    unmount();
    expect(socket.close).toHaveBeenCalled();
  });

  it("reconnects after a dropped connection and stops on cleanup", () => {
    vi.useFakeTimers();
    const { unmount } = renderHook(() => useExperimentSocket("exp_test"));
    const first = MockWebSocket.instances[0];

    act(() => first.drop());
    act(() => vi.advanceTimersByTime(1_000));
    expect(MockWebSocket.instances).toHaveLength(2);

    const second = MockWebSocket.instances[1];
    unmount();
    expect(second.close).toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(20_000));
    expect(MockWebSocket.instances).toHaveLength(2);
  });
});
