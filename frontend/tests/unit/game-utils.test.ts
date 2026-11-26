import { describe, expect, it } from "vitest";
import { buildShareSummary, validateLiveGuess } from "@/lib/game-utils";

describe("validateLiveGuess", () => {
  it("rejects empty and whitespace-only input", () => {
    expect(validateLiveGuess("", "en")).toBe("Enter a word to continue.");
    expect(validateLiveGuess("   ", "en")).toBe("Enter a word to continue.");
  });

  it("uses configured English minimum and maximum lengths", () => {
    expect(validateLiveGuess("ab", "en")).toBe("Use at least 3 characters.");
    expect(validateLiveGuess("abcdefghijklmnop", "en")).toBe(
      "Use no more than 15 characters.",
    );
    expect(validateLiveGuess("river", "en")).toBeNull();
  });

  it("allows long German compounds within the configured limit", () => {
    expect(validateLiveGuess("Krankenhaus", "de")).toBeNull();
    expect(validateLiveGuess("a".repeat(23), "de")).toBe(
      "Use no more than 22 characters.",
    );
  });

  it("counts Unicode characters for Arabic and accepts one-character Kanji", () => {
    expect(validateLiveGuess("ماء", "ar")).toBeNull();
    expect(validateLiveGuess("水", "ja")).toBeNull();
  });
});

describe("buildShareSummary", () => {
  it("formats the puzzle date, language, path, and guesses", () => {
    expect(
      buildShareSummary({
        language: "ja",
        puzzleId: "ja-daily:2026-03-08",
        pathLength: 4,
        guesses: 7,
      }),
    ).toBe(
      "🔗 LinkWords 2026-03-08 · JA\n🟩🟩🟩🟨\n4 steps · 7 guesses",
    );
  });

  it("wraps longer result grids into readable rows", () => {
    const summary = buildShareSummary({
      language: "en",
      puzzleId: "practice-42",
      pathLength: 12,
      guesses: 19,
    });

    expect(summary).toContain("🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩\n🟩🟨");
    expect(summary).toContain("12 steps · 19 guesses");
  });
});
