import { beforeEach, describe, expect, it, vi } from "vitest";
import { getAnonymousSessionId } from "@/lib/anonymous-session";

const SESSION_ID = "a39b62e6-77af-4f4f-b935-e87664125c8d";

describe("getAnonymousSessionId", () => {
  beforeEach(() => {
    window.localStorage.clear();
    Object.defineProperty(window, "crypto", {
      configurable: true,
      value: { randomUUID: vi.fn(() => SESSION_ID) },
    });
  });

  it("creates and persists a UUID without collecting user details", () => {
    expect(getAnonymousSessionId()).toBe(SESSION_ID);
    expect(window.localStorage.getItem("linkwords-anonymous-session-id")).toBe(
      SESSION_ID,
    );
  });

  it("reuses a saved UUID on later visits", () => {
    window.localStorage.setItem("linkwords-anonymous-session-id", SESSION_ID);

    expect(getAnonymousSessionId()).toBe(SESSION_ID);
    expect(window.crypto.randomUUID).not.toHaveBeenCalled();
  });
});
