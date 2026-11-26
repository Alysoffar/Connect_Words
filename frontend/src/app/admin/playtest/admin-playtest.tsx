"use client";

import Link from "next/link";
import { useState } from "react";
import {
  downloadAdminCsv,
  getPlaytestPuzzles,
  submitPlaytestFeedback,
  type PlaytestPuzzlesResponse,
} from "@/lib/api";
import { languageCodes, type LanguageCode } from "@/lib/languages";

type PlaytestPuzzle = PlaytestPuzzlesResponse["puzzles"][number];
type Drafts = Record<string, { rating: number; feedback: string }>;

export function AdminPlaytest() {
  const [language, setLanguage] = useState<LanguageCode>("en");
  const [token, setToken] = useState("");
  const [puzzles, setPuzzles] = useState<PlaytestPuzzle[]>([]);
  const [drafts, setDrafts] = useState<Drafts>({});
  const [isNativeSpeaker, setIsNativeSpeaker] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const loadPuzzles = async () => {
    if (!token.trim()) {
      setError("Enter the admin token to continue.");
      return;
    }
    setLoading(true);
    setError("");
    setStatus("");
    try {
      const result = await getPlaytestPuzzles(language, token.trim());
      setPuzzles(result.puzzles);
      setDrafts(
        Object.fromEntries(
          result.puzzles.map((puzzle) => [
            puzzle.puzzleId,
            { rating: 3, feedback: "" },
          ]),
        ),
      );
    } catch (cause: unknown) {
      setError(
        cause instanceof Error
          ? cause.message
          : "The playtest puzzles could not be loaded.",
      );
      setPuzzles([]);
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    if (puzzles.length !== 10 || !isNativeSpeaker) {
      setError("Confirm native-speaker status and load all ten puzzles first.");
      return;
    }
    setLoading(true);
    setError("");
    setStatus("");
    try {
      const result = await submitPlaytestFeedback(
        {
          lang: language,
          isNativeSpeaker,
          responses: puzzles.map((puzzle) => ({
            puzzleId: puzzle.puzzleId,
            rating: drafts[puzzle.puzzleId]?.rating ?? 3,
            feedback: drafts[puzzle.puzzleId]?.feedback ?? "",
          })),
        },
        token.trim(),
      );
      setStatus(`Saved feedback for ${result.submitted} ${language.toUpperCase()} puzzles.`);
    } catch (cause: unknown) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Feedback could not be submitted.",
      );
    } finally {
      setLoading(false);
    }
  };

  const exportCsv = async (kind: "rejected-words" | "slow-puzzles") => {
    if (!token.trim()) {
      setError("Enter the admin token to export reports.");
      return;
    }
    setError("");
    try {
      const file = await downloadAdminCsv(kind, language, token.trim());
      const url = URL.createObjectURL(file);
      const link = document.createElement("a");
      link.href = url;
      link.download = `linkwords-${kind}-${language}.csv`;
      link.click();
      URL.revokeObjectURL(url);
      setStatus(`${kind === "rejected-words" ? "Rejected-word" : "Slow-puzzle"} report downloaded.`);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "CSV export failed.");
    }
  };

  const updateDraft = (
    puzzleId: string,
    patch: Partial<Drafts[string]>,
  ) => {
    setDrafts((current) => ({
      ...current,
      [puzzleId]: { ...current[puzzleId], ...patch },
    }));
  };

  return (
    <main className="min-h-screen px-4 py-6 text-[#173c36] sm:px-7 sm:py-10">
      <div className="mx-auto max-w-4xl">
        <header className="border-b border-[#dce5df] pb-5">
          <Link href="/" className="text-sm font-bold text-[#17694f] underline">
            Back to LinkWords
          </Link>
          <h1 className="mt-2 text-3xl font-extrabold">Internal playtest</h1>
          <p className="mt-1 text-sm text-[#58716c]">
            Submit native-speaker ratings for ten puzzles or export language-specific CSV reports.
          </p>
        </header>

        <section className="mt-6 grid gap-4 rounded-3xl border border-[#dce5df] bg-white p-5 shadow-sm sm:grid-cols-2 sm:p-6">
          <label className="grid gap-1 text-sm font-bold">
            Language
            <select
              aria-label="Playtest language"
              value={language}
              onChange={(event) => {
                setLanguage(event.target.value as LanguageCode);
                setPuzzles([]);
                setDrafts({});
                setStatus("");
              }}
              className="min-h-11 rounded-xl border border-[#cbd8d1] bg-white px-3 font-semibold"
            >
              {languageCodes.map((code) => (
                <option key={code} value={code}>
                  {code.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-sm font-bold">
            Admin token
            <input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              className="min-h-11 rounded-xl border border-[#cbd8d1] px-3 font-medium"
            />
          </label>
          <div className="flex flex-wrap gap-2 sm:col-span-2">
            <button
              type="button"
              onClick={loadPuzzles}
              disabled={loading}
              className="min-h-11 rounded-xl bg-[#173c36] px-4 font-bold text-white disabled:opacity-50"
            >
              Load 10 puzzles
            </button>
            <button
              type="button"
              onClick={() => exportCsv("rejected-words")}
              disabled={loading}
              className="min-h-11 rounded-xl border border-[#9fb3a9] px-4 font-bold disabled:opacity-50"
            >
              Export rejected words CSV
            </button>
            <button
              type="button"
              onClick={() => exportCsv("slow-puzzles")}
              disabled={loading}
              className="min-h-11 rounded-xl border border-[#9fb3a9] px-4 font-bold disabled:opacity-50"
            >
              Export slow puzzles CSV
            </button>
          </div>
        </section>

        {error && (
          <p className="mt-4 rounded-xl bg-[#fff0ee] p-4 text-sm font-semibold text-[#9c302a]" role="alert">
            {error}
          </p>
        )}
        {status && (
          <p className="mt-4 rounded-xl bg-[#e8f5ee] p-4 text-sm font-semibold text-[#17694f]" role="status">
            {status}
          </p>
        )}

        {puzzles.length > 0 && (
          <section className="mt-6">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-xl font-extrabold">
                Ten {language.toUpperCase()} puzzles
              </h2>
              <label className="flex min-h-11 items-center gap-2 rounded-xl border border-[#cbd8d1] bg-white px-3 text-sm font-semibold">
                <input
                  type="checkbox"
                  checked={isNativeSpeaker}
                  onChange={(event) => setIsNativeSpeaker(event.target.checked)}
                />
                I am a native speaker of this language
              </label>
            </div>
            <div className="grid gap-3">
              {puzzles.map((puzzle, index) => (
                <article
                  key={puzzle.puzzleId}
                  className="grid gap-3 rounded-2xl border border-[#dce5df] bg-white p-4 shadow-sm sm:grid-cols-[1fr_150px]"
                >
                  <div>
                    <h3 className="font-bold">
                      {index + 1}. {puzzle.anchorA} ↔ {puzzle.anchorB}
                    </h3>
                    <p className="mt-1 break-all text-xs text-[#71857d]">
                      {puzzle.puzzleId}
                    </p>
                    <label className="mt-3 grid gap-1 text-xs font-bold text-[#58716c]">
                      Feedback (avoid personal information)
                      <textarea
                        maxLength={1000}
                        value={drafts[puzzle.puzzleId]?.feedback ?? ""}
                        onChange={(event) =>
                          updateDraft(puzzle.puzzleId, {
                            feedback: event.target.value,
                          })
                        }
                        rows={2}
                        className="rounded-xl border border-[#cbd8d1] p-3 text-sm font-normal text-[#173c36]"
                      />
                    </label>
                  </div>
                  <label className="grid content-start gap-1 text-sm font-bold">
                    Rating
                    <select
                      value={drafts[puzzle.puzzleId]?.rating ?? 3}
                      onChange={(event) =>
                        updateDraft(puzzle.puzzleId, {
                          rating: Number(event.target.value),
                        })
                      }
                      className="min-h-11 rounded-xl border border-[#cbd8d1] bg-white px-3"
                    >
                      {[1, 2, 3, 4, 5].map((rating) => (
                        <option key={rating} value={rating}>
                          {rating} / 5
                        </option>
                      ))}
                    </select>
                  </label>
                </article>
              ))}
            </div>
            <button
              type="button"
              onClick={submitFeedback}
              disabled={loading || !isNativeSpeaker}
              className="mt-4 min-h-12 rounded-xl bg-[#173c36] px-5 font-bold text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Saving…" : "Submit all 10 ratings"}
            </button>
          </section>
        )}
      </div>
    </main>
  );
}
