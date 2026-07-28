import { describe, expect, it, vi } from "vitest";
import { explain, healthCheck, predict } from "../lib/api";

vi.stubGlobal("fetch", vi.fn());

describe("api utilities", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("returns false when health endpoint is unreachable", async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error("network error")));
    await expect(healthCheck()).resolves.toBe(false);
  });

  it("throws a readable error when predict fails", async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({ ok: false, statusText: "Bad Request", json: () => Promise.resolve({ detail: "Bad file" }) })
    );
    const file = new File(["data"], "test.png", { type: "image/png" });
    await expect(predict(file)).rejects.toThrow(/Bad file/i);
  });
});
