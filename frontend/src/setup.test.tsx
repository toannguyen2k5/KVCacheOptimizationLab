import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { experimentService } from "./services/experimentService";

vi.mock("./services/experimentService", () => ({
  experimentService: {
    options: vi.fn(),
    validate: vi.fn(),
    create: vi.fn(),
    list: vi.fn(),
    get: vi.fn(),
    logs: vi.fn(),
    socketUrl: vi.fn(),
    parseEvent: vi.fn(),
  },
}));

vi.mock("./services/optimizationService", () => ({
  optimizationService: {
    methods: vi.fn(),
  },
}));

import { optimizationService } from "./services/optimizationService";

describe("Experiment setup", () => {
  beforeEach(() => {
    vi.mocked(experimentService.options).mockResolvedValue({
      generation: {
        max_new_tokens: { minimum: 1, maximum: 1024, default: 200 },
        temperature: null,
        sampling: null,
      },
      methods: ["xkv", "xkv_sr"],
      prompt: { minimum_length: 1, maximum_length: 20_000 },
    });
    vi.mocked(optimizationService.methods).mockResolvedValue([
      {
        id: "xkv",
        name: "XKV",
        description: "Existing XKV optimization pipeline.",
        available: true,
        parameters: [],
      },
      {
        id: "xkv_sr",
        name: "XKV-SR",
        description: "Selective chunk reconstruction.",
        available: true,
        parameters: [],
      },
    ]);
  });

  it("rejects a whitespace-only prompt before calling the API", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/setup"]}>
        <App />
      </MemoryRouter>,
    );
    await user.type(await screen.findByRole("textbox", { name: /prompt/i }), "   ");
    await user.click(screen.getByRole("button", { name: /run experiment/i }));

    expect(
      await screen.findByText("Enter a prompt before launching the experiment."),
    ).toBeInTheDocument();
    expect(experimentService.validate).not.toHaveBeenCalled();
    expect(experimentService.create).not.toHaveBeenCalled();
  });

  it("submits the selected XKV-SR method", async () => {
    const user = userEvent.setup();
    vi.mocked(experimentService.validate).mockResolvedValue(undefined);
    vi.mocked(experimentService.create).mockRejectedValue(
      new Error("Stop after checking the submitted configuration."),
    );
    render(
      <MemoryRouter initialEntries={["/setup"]}>
        <App />
      </MemoryRouter>,
    );

    await user.type(await screen.findByRole("textbox", { name: /prompt/i }), "test");
    await user.selectOptions(
      await screen.findByRole("combobox", { name: /optimization method/i }),
      "xkv_sr",
    );
    await user.click(screen.getByRole("button", { name: /run experiment/i }));

    expect(experimentService.validate).toHaveBeenCalledWith({
      prompt: "test",
      method: "xkv_sr",
      max_new_tokens: 200,
    });
    expect(experimentService.create).toHaveBeenCalledWith({
      prompt: "test",
      method: "xkv_sr",
      max_new_tokens: 200,
    });
  });
});
