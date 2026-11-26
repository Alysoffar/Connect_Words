"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import Link from "next/link";
import {
  APIClientError,
  getPuzzle,
  requestClue,
  submitGuess as sendGuess,
  type EdgeResponse,
  type GuessRequest,
  type PuzzleResponse,
} from "@/lib/api";
import { getAnonymousSessionId } from "@/lib/anonymous-session";
import {
  getFontFamily,
  getLanguageDirection,
  getLanguageSettings,
  languageCodes,
  type LanguageCode,
} from "@/lib/languages";
import { validateLiveGuess } from "@/lib/game-utils";
import {
  ForceGraphCanvas,
  type GraphEdge,
  type GraphNode,
} from "@/components/force-graph-canvas";
import { ResultCard } from "@/components/result-card";

type GameMode = "daily" | "practice";

interface BoardProgress {
  words: string[];
  edges: EdgeResponse[];
  prunedEdges: EdgeResponse[];
  path: string[] | null;
  score: number | null;
}

interface TimedNearMiss extends GraphEdge {
  expiresAt: number;
}

const EMPTY_BOARD: BoardProgress = {
  words: [],
  edges: [],
  prunedEdges: [],
  path: null,
  score: null,
};

function getEdgeKey(edge: Pick<EdgeResponse, "wordA" | "wordB">): string {
  return JSON.stringify([edge.wordA, edge.wordB].sort());
}

function isBoardProgress(value: unknown): value is BoardProgress {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const saved = value as Record<string, unknown>;
  const isEdge = (edge: unknown): edge is EdgeResponse =>
    typeof edge === "object" &&
    edge !== null &&
    "wordA" in edge &&
    typeof edge.wordA === "string" &&
    "wordB" in edge &&
    typeof edge.wordB === "string" &&
    "weight" in edge &&
    typeof edge.weight === "number" &&
    "nearMiss" in edge &&
    typeof edge.nearMiss === "boolean" &&
    "retained" in edge &&
    typeof edge.retained === "boolean";
  return (
    Array.isArray(saved.words) &&
    saved.words.every((word) => typeof word === "string") &&
    new Set(saved.words).size === saved.words.length &&
    Array.isArray(saved.edges) &&
    saved.edges.every(isEdge) &&
    Array.isArray(saved.prunedEdges) &&
    saved.prunedEdges.every(isEdge) &&
    (saved.path === null || Array.isArray(saved.path)) &&
    (saved.path === null || saved.path.every((word) => typeof word === "string")) &&
    (saved.score === null || typeof saved.score === "number") &&
    saved.words.length <= 50
  );
}

function initialBoard(puzzle: PuzzleResponse): BoardProgress {
  return {
    ...EMPTY_BOARD,
    words: [puzzle.anchorA, puzzle.anchorB],
  };
}

export function GameClient() {
  const [language, setLanguage] = useState<LanguageCode>("en");
  const [mode, setMode] = useState<GameMode>("daily");
  const [puzzle, setPuzzle] = useState<PuzzleResponse | null>(null);
  const [board, setBoard] = useState<BoardProgress>(EMPTY_BOARD);
  const [loadingPuzzle, setLoadingPuzzle] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [input, setInput] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [storageMessage, setStorageMessage] = useState("");
  const [clue, setClue] = useState<string | null>(null);
  const [clueLoading, setClueLoading] = useState(false);
  const [nearMisses, setNearMisses] = useState<TimedNearMiss[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  const settings = getLanguageSettings(language);
  const direction = getLanguageDirection(language);
  const acceptedGuesses = Math.max(0, board.words.length - 2);
  const isSolved = Boolean(board.path?.length);

  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dir = direction;
  }, [direction, language]);

  useEffect(() => {
    let active = true;

    Promise.resolve()
      .then(() => getAnonymousSessionId())
      .then((sessionId) => {
        sessionIdRef.current = sessionId;
        return getPuzzle(mode, language, sessionId);
      })
      .then((loadedPuzzle) => {
        if (!active || !loadedPuzzle) {
          return;
        }
        setPuzzle(loadedPuzzle);
        const storageKey = `linkwords-board:${language}:${loadedPuzzle.puzzleId}`;
        let progress = initialBoard(loadedPuzzle);
        try {
          const raw = window.localStorage.getItem(storageKey);
          if (raw) {
            const parsed: unknown = JSON.parse(raw);
            if (
              isBoardProgress(parsed) &&
              parsed.words.includes(loadedPuzzle.anchorA) &&
              parsed.words.includes(loadedPuzzle.anchorB)
            ) {
              progress = parsed;
            } else {
              setStorageMessage("Saved progress was invalid and has been reset.");
            }
          }
        } catch {
          setStorageMessage("Saved progress could not be read in this browser.");
        }
        setBoard(progress);
        setLoadingPuzzle(false);
      })
      .catch((error: unknown) => {
        if (!active) {
          return;
        }
        setApiError(
          error instanceof Error ? error.message : "The puzzle could not be loaded.",
        );
        setLoadingPuzzle(false);
      });

    return () => {
      active = false;
    };
  }, [language, mode]);

  useEffect(() => {
    if (nearMisses.length === 0) {
      return;
    }
    const remaining = Math.max(
      0,
      ...nearMisses.map((edge) => edge.expiresAt - Date.now()),
    );
    const timer = window.setTimeout(
      () => setNearMisses((current) => current.filter((edge) => edge.expiresAt > Date.now())),
      remaining + 20,
    );
    return () => window.clearTimeout(timer);
  }, [nearMisses]);

  const graphNodes = useMemo<GraphNode[]>(() => {
    if (!puzzle) {
      return [];
    }
    return board.words.map((word) => ({
      id: word,
      kind:
        word === puzzle.anchorA
          ? "anchor-a"
          : word === puzzle.anchorB
            ? "anchor-b"
            : "word",
    }));
  }, [board.words, puzzle]);

  const graphEdges = useMemo<GraphEdge[]>(() => {
    const retained = board.edges.map((edge) => ({
      source: edge.wordA,
      target: edge.wordB,
      weight: edge.weight,
    }));
    return [...retained, ...nearMisses];
  }, [board.edges, nearMisses]);

  const selectedNeighbors = useMemo(() => {
    if (!selectedNode) {
      return [];
    }
    return board.edges
      .filter(
        (edge) => edge.wordA === selectedNode || edge.wordB === selectedNode,
      )
      .map((edge) => ({
        word: edge.wordA === selectedNode ? edge.wordB : edge.wordA,
        weight: edge.weight,
      }))
      .sort((first, second) => second.weight - first.weight);
  }, [board.edges, selectedNode]);

  const selectedPruned = useMemo(
    () =>
      selectedNode
        ? board.prunedEdges.filter(
            (edge) => edge.wordA === selectedNode || edge.wordB === selectedNode,
          )
        : [],
    [board.prunedEdges, selectedNode],
  );

  const resetForPuzzleChange = () => {
    setLoadingPuzzle(true);
    setPuzzle(null);
    setBoard(EMPTY_BOARD);
    setNearMisses([]);
    setSelectedNode(null);
    setClue(null);
    setInput("");
    setValidationError(null);
    setApiError(null);
    setStorageMessage("");
  };

  const handleLanguageChange = (value: string) => {
    if (
      value !== language &&
      languageCodes.includes(value as LanguageCode)
    ) {
      resetForPuzzleChange();
      setLanguage(value as LanguageCode);
    }
  };

  const handleModeChange = (nextMode: GameMode) => {
    if (mode !== nextMode) {
      resetForPuzzleChange();
      setMode(nextMode);
    }
  };

  const handleGuess = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!puzzle || submitting) {
      return;
    }
    const liveError = validateLiveGuess(input, language);
    setValidationError(liveError);
    setApiError(null);
    if (liveError) {
      return;
    }
    if (board.words.length >= 50) {
      setApiError("The board is full. A board can contain at most 50 words.");
      return;
    }
    const sessionId = sessionIdRef.current;
    if (!sessionId) {
      setApiError("Anonymous session could not be initialized. Reload to try again.");
      return;
    }

    const request = {
      puzzleId: puzzle.puzzleId,
      sessionId,
      mode,
      lang: language,
      word: input.trim(),
      currentBoardWords: board.words,
    } satisfies GuessRequest;

    setSubmitting(true);
    try {
      const result = await sendGuess(request);
      const removed = new Set(result.prunedEdges.map(getEdgeKey));
      const nextEdges = new Map(
        board.edges
          .filter((edge) => !removed.has(getEdgeKey(edge)))
          .map((edge) => [getEdgeKey(edge), edge]),
      );
      for (const edge of result.edges) {
        if (!edge.nearMiss && edge.retained) {
          nextEdges.set(getEdgeKey(edge), edge);
        }
      }
      const nextPruned = new Map(
        board.prunedEdges.map((edge) => [getEdgeKey(edge), edge]),
      );
      for (const edge of result.prunedEdges) {
        nextPruned.set(getEdgeKey(edge), edge);
      }
      const nextBoard: BoardProgress = {
        words: [...board.words, result.normalized],
        edges: [...nextEdges.values()],
        prunedEdges: [...nextPruned.values()],
        path: result.path,
        score: result.score,
      };
      setBoard(nextBoard);
      try {
        window.localStorage.setItem(
          `linkwords-board:${language}:${puzzle.puzzleId}`,
          JSON.stringify(nextBoard),
        );
      } catch {
        setStorageMessage("Progress could not be saved in this browser.");
      }
      setInput("");
      setValidationError(null);
      setClue(null);
      setNearMisses(
        result.edges
          .filter((edge) => edge.nearMiss)
          .map((edge) => ({
            source: edge.wordA,
            target: edge.wordB,
            weight: edge.weight,
            nearMiss: true,
            expiresAt: Date.now() + 2000,
          })),
      );
    } catch (error: unknown) {
      if (error instanceof APIClientError) {
        setApiError(error.message);
      } else {
        setApiError("The guess could not be submitted. Please try again.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleClue = async () => {
    if (!puzzle || clueLoading || acceptedGuesses < 20) {
      return;
    }
    const sessionId = sessionIdRef.current;
    if (!sessionId) {
      setApiError("Anonymous session could not be initialized. Reload to try again.");
      return;
    }
    setClueLoading(true);
    setApiError(null);
    try {
      const result = await requestClue({
        puzzleId: puzzle.puzzleId,
        sessionId,
        mode,
        lang: language,
        currentBoardWords: board.words,
      });
      setClue(`Try “${result.word}” — it connects from “${result.fromWord}”.`);
    } catch (error: unknown) {
      setApiError(
        error instanceof Error ? error.message : "A clue could not be loaded.",
      );
    } finally {
      setClueLoading(false);
    }
  };

  const shareResult =
    puzzle && board.path && board.score !== null
      ? {
          language,
          puzzleId: puzzle.puzzleId,
          pathLength: board.path.length - 1,
          guesses: acceptedGuesses,
          score: board.score,
        }
      : null;

  const liveError = input ? validationError : null;

  return (
    <main
      lang={language}
      dir={direction}
      style={{ fontFamily: getFontFamily(language) }}
      className="min-h-screen px-4 pb-10 pt-5 text-[#173c36] sm:px-7 sm:pt-8"
    >
      <div className="mx-auto max-w-7xl">
        <header className="flex flex-col gap-5 border-b border-[#dce5df] pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div
              aria-hidden="true"
              className="grid size-12 place-items-center rounded-2xl bg-[#173c36] text-xl text-white shadow-sm"
            >
              ↗
            </div>
            <div>
              <p className="text-xl font-extrabold tracking-tight">LinkWords</p>
              <p className="text-sm text-[#58716c]">Find the connection.</p>
            </div>
          </div>

          <div className="flex flex-wrap items-end gap-3">
            <label className="grid gap-1 text-xs font-bold uppercase tracking-wide text-[#58716c]">
              Language
              <select
                aria-label="Game language"
                value={language}
                onChange={(event) => handleLanguageChange(event.target.value)}
                disabled={submitting || clueLoading || loadingPuzzle}
                className="min-h-11 rounded-xl border border-[#cbd8d1] bg-white px-3 text-sm font-semibold normal-case tracking-normal text-[#173c36]"
              >
                {languageCodes.map((code) => (
                  <option key={code} value={code}>
                    {code.toUpperCase()}
                  </option>
                ))}
              </select>
            </label>
            <div
              className="flex rounded-xl border border-[#cbd8d1] bg-white p-1"
              aria-label="Puzzle mode"
              role="group"
            >
              {(["daily", "practice"] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={mode === value}
                  onClick={() => handleModeChange(value)}
                  disabled={submitting || clueLoading || loadingPuzzle}
                  className={`min-h-9 rounded-lg px-3 text-sm font-bold capitalize transition ${
                    mode === value
                      ? "bg-[#173c36] text-white"
                      : "text-[#58716c] hover:bg-[#f1f5f1]"
                  }`}
                >
                  {value}
                </button>
              ))}
            </div>
          </div>
        </header>

        <section className="grid gap-5 py-6 lg:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.8fr)]">
          <div className="min-w-0">
            <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-sm font-bold text-[#58716c]">
                  {mode === "daily" ? "DAILY CONNECTION" : "PRACTICE"}
                  {puzzle ? ` · ${puzzle.puzzleId}` : ""}
                </p>
                <h1 className="mt-1 text-2xl font-extrabold tracking-tight sm:text-3xl">
                  Connect the anchors
                </h1>
              </div>
              <div className="flex flex-wrap gap-2 text-sm font-bold">
                <span className="inline-flex items-center gap-2 rounded-full bg-white px-3 py-2 shadow-sm">
                  <span className="size-3 rounded-full bg-[#0072b2]" aria-hidden="true" />
                  Anchor A
                </span>
                <span className="inline-flex items-center gap-2 rounded-full bg-white px-3 py-2 shadow-sm">
                  <span className="size-3 rotate-45 bg-[#d55e00]" aria-hidden="true" />
                  Anchor B
                </span>
              </div>
            </div>

            <div className="overflow-hidden rounded-3xl border border-[#dce5df] bg-white shadow-sm">
              {loadingPuzzle ? (
                <div
                  className="grid place-items-center text-sm font-semibold text-[#58716c]"
                  style={{ height: 440 }}
                  role="status"
                >
                  Loading puzzle…
                </div>
              ) : puzzle ? (
                <ForceGraphCanvas
                  nodes={graphNodes}
                  edges={graphEdges}
                  winningPath={board.path}
                  selectedNode={selectedNode}
                  onSelectNode={setSelectedNode}
                  height={440}
                />
              ) : (
                <div
                  className="grid place-items-center p-8 text-center text-sm font-semibold text-[#58716c]"
                  style={{ height: 440 }}
                  role="status"
                >
                  {apiError ?? "Puzzle is not available."}
                </div>
              )}
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs font-semibold text-[#58716c]">
              <span>Drag a node to explore</span>
              <span className="hidden text-[#a6b5ae] sm:inline" aria-hidden="true">•</span>
              <span>Dotted lines are near misses</span>
              <span className="hidden text-[#a6b5ae] sm:inline" aria-hidden="true">•</span>
              <span>Green and gold show a solved path</span>
            </div>

            <section aria-labelledby="board-words-title" className="mt-5">
              <div className="mb-2 flex items-center justify-between gap-3">
                <h2 id="board-words-title" className="text-sm font-bold">
                  Board words
                </h2>
                <span className="text-xs font-semibold text-[#58716c]">
                  Select a word to inspect its connections
                </span>
              </div>
              <div className="flex max-h-28 flex-wrap gap-2 overflow-y-auto rounded-2xl border border-[#dce5df] bg-white p-3">
                {board.words.map((word) => (
                  <button
                    key={word}
                    type="button"
                    aria-label={`Inspect node ${word}`}
                    aria-pressed={selectedNode === word}
                    onClick={() => setSelectedNode(word)}
                    className={`min-h-9 rounded-full border px-3 text-sm font-semibold transition ${
                      selectedNode === word
                        ? "border-[#173c36] bg-[#173c36] text-white"
                        : "border-[#dce5df] bg-[#f7f9f6] text-[#173c36] hover:border-[#94aaa0]"
                    }`}
                  >
                    {word}
                  </button>
                ))}
                {board.words.length === 0 && (
                  <span className="text-sm text-[#58716c]">The board is loading.</span>
                )}
              </div>
            </section>
          </div>

          <aside className="flex flex-col gap-4">
            <section
              aria-labelledby="anchor-title"
              className="rounded-3xl border border-[#dce5df] bg-white p-5 shadow-sm sm:p-6"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.16em] text-[#58716c]">
                    Today’s pair
                  </p>
                  <h2 id="anchor-title" className="sr-only">
                    Puzzle anchors
                  </h2>
                </div>
                {puzzle && (
                  <span className="rounded-full bg-[#eff4f0] px-3 py-1 text-xs font-bold text-[#58716c]">
                    {puzzle.lang.toUpperCase()}
                  </span>
                )}
              </div>
              <div className="mt-4 grid grid-cols-[1fr_auto_1fr] items-center gap-2">
                <div className="min-w-0 rounded-2xl border border-[#0072b2]/25 bg-[#edf6fb] p-3 text-center sm:p-4">
                  <p className="text-[11px] font-bold uppercase tracking-wide text-[#005a8c]">
                    Anchor A
                  </p>
                  <p className="mt-1 break-words text-lg font-extrabold text-[#173c36]">
                    {puzzle?.anchorA ?? "—"}
                  </p>
                </div>
                <span aria-hidden="true" className="text-lg text-[#9aaca4]">↔</span>
                <div className="min-w-0 rounded-2xl border border-[#d55e00]/25 bg-[#fff3eb] p-3 text-center sm:p-4">
                  <p className="text-[11px] font-bold uppercase tracking-wide text-[#a44300]">
                    Anchor B
                  </p>
                  <p className="mt-1 break-words text-lg font-extrabold text-[#173c36]">
                    {puzzle?.anchorB ?? "—"}
                  </p>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-between text-sm">
                <span className="font-bold">Board meter</span>
                <span className="font-bold tabular-nums">
                  {board.words.length} / 50
                </span>
              </div>
              <div
                className="mt-2 h-2.5 overflow-hidden rounded-full bg-[#e8eeea]"
                role="progressbar"
                aria-label="Board words"
                aria-valuemin={0}
                aria-valuemax={50}
                aria-valuenow={board.words.length}
              >
                <div
                  className="h-full rounded-full bg-[#009e73] transition-[width]"
                  style={{ width: `${Math.min(100, (board.words.length / 50) * 100)}%` }}
                />
              </div>

              <div className="mt-4 flex items-center justify-between rounded-2xl bg-[#f5f7f2] px-4 py-3">
                <span className="text-sm font-semibold text-[#58716c]">
                  Accepted guesses
                </span>
                <span className="text-lg font-extrabold tabular-nums">
                  {acceptedGuesses}
                </span>
              </div>
            </section>

            <section
              aria-labelledby="guess-title"
              className="rounded-3xl border border-[#dce5df] bg-white p-5 shadow-sm sm:p-6"
            >
              <h2 id="guess-title" className="text-lg font-extrabold">
                Add a word
              </h2>
              <p className="mt-1 text-sm leading-6 text-[#58716c]">
                Find a word that links to something already on the board.
              </p>
              <form className="mt-4" onSubmit={handleGuess} noValidate>
                <label htmlFor="guess-word" className="sr-only">
                  Enter a word
                </label>
                <div className="flex flex-col gap-2 sm:flex-row">
                  <input
                    id="guess-word"
                    name="word"
                    type="text"
                    autoComplete="off"
                    value={input}
                    onChange={(event) => {
                      setInput(event.target.value);
                      setValidationError(
                        validateLiveGuess(event.target.value, language),
                      );
                      setApiError(null);
                    }}
                    aria-invalid={Boolean(liveError || apiError)}
                    aria-describedby="guess-help guess-error"
                    placeholder="Type a word…"
                    className="min-h-12 min-w-0 flex-1 rounded-xl border border-[#b8c9c0] bg-white px-4 text-base text-[#173c36] placeholder:text-[#8b9d95]"
                    disabled={!puzzle || loadingPuzzle || isSolved}
                  />
                  <button
                    type="submit"
                    disabled={!puzzle || loadingPuzzle || submitting || isSolved}
                    className="min-h-12 rounded-xl bg-[#173c36] px-5 font-bold text-white transition hover:bg-[#24574e] disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {submitting ? "Linking…" : "Connect"}
                  </button>
                </div>
                <p id="guess-help" className="mt-2 text-xs leading-5 text-[#58716c]">
                  {`Allowed length: ${settings.min_word_length}–${settings.max_word_length} characters. Vocabulary is checked on submit.`}
                </p>
                <p
                  id="guess-error"
                  className="mt-1 min-h-5 text-sm font-semibold text-[#a3332d]"
                  role={liveError || apiError ? "alert" : "status"}
                  aria-live="polite"
                >
                  {liveError ?? apiError ?? ""}
                </p>
              </form>

              {nearMisses.length > 0 && (
                <div
                  role="status"
                  aria-live="polite"
                  className="mt-1 rounded-xl border border-dashed border-[#8b9d95] bg-[#f5f7f2] px-3 py-2 text-sm text-[#455d65]"
                >
                  Near miss:{" "}
                  {nearMisses
                    .map(
                      (edge) =>
                        `${edge.source} ↔ ${edge.target} (${Math.round(edge.weight * 100)}%)`,
                    )
                    .join(", ")}
                </div>
              )}

              <div className="mt-4 border-t border-[#e5ebe7] pt-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="font-bold">Need a nudge?</p>
                    <p className="mt-1 text-xs text-[#58716c]">
                      Unlocks after 20 accepted guesses.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleClue}
                    disabled={
                      !puzzle ||
                      clueLoading ||
                      acceptedGuesses < 20 ||
                      isSolved
                    }
                    aria-label={`Request clue, ${Math.max(0, 20 - acceptedGuesses)} guesses remaining`}
                    className="min-h-11 shrink-0 rounded-xl border border-[#9fb3a9] px-4 text-sm font-bold text-[#173c36] transition hover:bg-[#f5f7f2] disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    {clueLoading ? "Finding…" : "Get a clue"}
                  </button>
                </div>
                {clue && (
                  <p className="mt-3 rounded-xl bg-[#fff6dd] px-3 py-2 text-sm font-semibold text-[#604500]">
                    {clue}
                  </p>
                )}
              </div>
            </section>

            {selectedNode && (
              <section
                aria-labelledby="inspect-title"
                className="rounded-3xl border border-[#dce5df] bg-white p-5 shadow-sm"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs font-bold uppercase tracking-wide text-[#58716c]">
                      Node inspection
                    </p>
                    <h2 id="inspect-title" className="mt-1 break-words text-lg font-extrabold">
                      {selectedNode}
                    </h2>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSelectedNode(null)}
                    className="grid size-9 shrink-0 place-items-center rounded-full text-lg text-[#58716c] hover:bg-[#f1f5f1]"
                    aria-label="Close node inspection"
                  >
                    ×
                  </button>
                </div>
                <h3 className="mt-4 text-xs font-bold uppercase tracking-wide text-[#58716c]">
                  Active neighbors
                </h3>
                {selectedNeighbors.length ? (
                  <ul className="mt-2 space-y-2">
                    {selectedNeighbors.map((neighbor) => (
                      <li
                        key={neighbor.word}
                        className="flex items-center justify-between gap-3 text-sm"
                      >
                        <span className="break-all font-semibold">{neighbor.word}</span>
                        <span className="shrink-0 rounded-full bg-[#e8f5ee] px-2.5 py-1 font-bold text-[#17694f]">
                          {Math.round(neighbor.weight * 100)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-[#58716c]">No active edges.</p>
                )}
                <h3 className="mt-4 text-xs font-bold uppercase tracking-wide text-[#58716c]">
                  Pruned edges
                </h3>
                {selectedPruned.length ? (
                  <ul className="mt-2 space-y-2">
                    {selectedPruned.map((edge) => (
                      <li
                        key={getEdgeKey(edge)}
                        className="flex items-center justify-between gap-3 text-sm text-[#7b4d2a]"
                      >
                        <span className="break-all">
                          {edge.wordA === selectedNode ? edge.wordB : edge.wordA}
                        </span>
                        <span className="shrink-0 font-semibold line-through">
                          {Math.round(edge.weight * 100)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-sm text-[#58716c]">No pruned edges.</p>
                )}
              </section>
            )}
          </aside>
        </section>

        {storageMessage && (
          <p className="mb-4 text-sm text-[#7b4d2a]" role="status">
            {storageMessage}
          </p>
        )}
        {shareResult && <ResultCard result={shareResult} />}
        {isSolved && board.path && (
          <p className="sr-only" role="status" aria-live="assertive">
            Puzzle solved in {board.path.length - 1} steps and {acceptedGuesses} guesses.
          </p>
        )}
        <footer className="pt-6 text-center text-xs text-[#71857d]">
          Similarity links reveal the route. The best path is yours to find.
          <span aria-hidden="true"> · </span>
          <Link href="/leaderboard" className="font-bold underline">
            Leaderboard
          </Link>
        </footer>
      </div>
    </main>
  );
}
