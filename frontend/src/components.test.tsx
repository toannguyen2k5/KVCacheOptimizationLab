import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MetricCard, StatusBadge } from "./components";

describe("StatusBadge", () => {
  it.each([
    ["queued", "Queued"],
    ["running", "Running"],
    ["completed", "Completed"],
    ["failed", "Failed"],
    ["cancelled", "Cancelled"],
  ] as const)("renders %s status", (status, label) => {
    render(<StatusBadge status={status} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });
});

describe("MetricCard", () => {
  it("distinguishes missing metrics from a numeric zero", () => {
    const { rerender } = render(<MetricCard label="Cache saved" value={null} unit="MB" />);
    expect(screen.getByText("Not provided")).toBeInTheDocument();

    rerender(<MetricCard label="Cache saved" value={0} unit="MB" />);
    expect(screen.getByText("0")).toBeInTheDocument();
    expect(screen.queryByText("Not provided")).not.toBeInTheDocument();
  });
});
