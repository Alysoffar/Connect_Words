"use client";

import { useState } from "react";
import {
  buildShareSummary,
  type ShareSummary,
} from "@/lib/game-utils";

interface ResultCardProps {
  result: ShareSummary;
}

export function ResultCard({ result }: ResultCardProps) {
  const [message, setMessage] = useState("");
  const summary = buildShareSummary(result);

  const copySummary = async () => {
    try {
      await navigator.clipboard.writeText(summary);
      setMessage("Copied to clipboard.");
    } catch {
      setMessage("Clipboard access is unavailable in this browser.");
    }
  };

  const exportImage = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 960;
    canvas.height = 540;
    const context = canvas.getContext("2d");
    if (!context) {
      setMessage("Image export is not supported in this browser.");
      return;
    }

    context.fillStyle = "#f5f7f2";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#173c36";
    context.font = "700 48px system-ui, sans-serif";
    context.fillText("LinkWords", 64, 106);
    context.font = "500 28px system-ui, sans-serif";
    context.fillText(`${result.language.toUpperCase()} · ${result.puzzleId}`, 64, 164);
    context.font = "56px system-ui, sans-serif";
    context.fillText(
      `${"🟩".repeat(Math.max(0, result.pathLength - 1))}🟨`,
      64,
      274,
    );
    context.font = "600 30px system-ui, sans-serif";
    context.fillText(
      `${result.pathLength} steps   ·   ${result.guesses} guesses`,
      64,
      354,
    );
    context.font = "24px system-ui, sans-serif";
    context.fillStyle = "#58716c";
    context.fillText("Connect the words. Find your path.", 64, 450);

    canvas.toBlob((blob) => {
      if (!blob) {
        setMessage("The result image could not be created.");
        return;
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `linkwords-${result.language}-${result.puzzleId}.png`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setMessage("Image downloaded.");
    }, "image/png");
  };

  return (
    <section
      aria-labelledby="share-title"
      className="rounded-3xl border border-[#dce5df] bg-white p-5 shadow-sm sm:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-[#58716c]">
            Puzzle complete
          </p>
          <h2 id="share-title" className="mt-1 text-xl font-bold text-[#173c36]">
            Your connection
          </h2>
        </div>
        <span
          className="rounded-full bg-[#e8f5ee] px-3 py-1 text-sm font-bold text-[#17694f]"
          aria-label={`Final score ${result.score ?? 0}`}
        >
          {result.score ?? 0} pts
        </span>
      </div>
      <pre
        aria-label="Shareable result summary"
        className="mt-4 whitespace-pre-wrap rounded-2xl bg-[#f5f7f2] p-4 font-mono text-sm leading-7 text-[#173c36]"
      >
        {summary}
      </pre>
      <div className="mt-4 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={copySummary}
          className="min-h-11 rounded-xl bg-[#173c36] px-4 py-2 text-sm font-bold text-white transition hover:bg-[#24574e]"
          aria-label="Copy result summary"
        >
          Copy result
        </button>
        <button
          type="button"
          onClick={exportImage}
          className="min-h-11 rounded-xl border border-[#aebfb7] px-4 py-2 text-sm font-bold text-[#173c36] transition hover:bg-[#f5f7f2]"
          aria-label="Download result as image"
        >
          Download image
        </button>
      </div>
      <p className="mt-2 min-h-5 text-sm text-[#58716c]" role="status" aria-live="polite">
        {message}
      </p>
    </section>
  );
}
