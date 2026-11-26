"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { getStats, type StatsResponse } from "@/lib/api";
import {
  getLanguageSettings,
  languageCodes,
  type LanguageCode,
} from "@/lib/languages";

type LanguageFilter = LanguageCode | "all";

function isLanguageCode(value: string): value is LanguageCode {
  return languageCodes.some((code) => code === value);
}

function percentage(value: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

function average(value: number | null): string {
  return value === null ? "—" : value.toFixed(1);
}

export default function LeaderboardPage() {
  const [language, setLanguage] = useState<LanguageFilter>("all");
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    getStats(language === "all" ? undefined : language)
      .then((result) => {
        if (active) {
          setStats(result);
          setError("");
        }
      })
      .catch((cause: unknown) => {
        if (active) {
          setError(
            cause instanceof Error
              ? cause.message
              : "Statistics are temporarily unavailable.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, [language]);

  return (
    <main className="min-h-screen px-4 py-6 text-[#173c36] sm:px-7 sm:py-10">
      <div className="mx-auto max-w-6xl">
        <header className="flex flex-wrap items-center justify-between gap-4 border-b border-[#dce5df] pb-5">
          <div>
            <Link href="/" className="text-sm font-bold text-[#17694f] underline">
              Back to LinkWords
            </Link>
            <h1 className="mt-2 text-3xl font-extrabold">Leaderboard & stats</h1>
            <p className="mt-1 text-sm text-[#58716c]">
              Anonymous game totals; no player identities are shown.
            </p>
          </div>
          <label className="grid gap-1 text-xs font-bold uppercase tracking-wide text-[#58716c]">
            Fastest solves
            <select
              aria-label="Leaderboard language"
              value={language}
              onChange={(event) =>
                setLanguage(event.target.value as LanguageFilter)
              }
              className="min-h-11 rounded-xl border border-[#cbd8d1] bg-white px-3 text-sm font-semibold normal-case tracking-normal text-[#173c36]"
            >
              <option value="all">All languages</option>
              {languageCodes.map((code) => (
                <option key={code} value={code}>
                  {code.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
        </header>

        {error && (
          <p className="mt-5 rounded-xl bg-[#fff0ee] p-4 text-sm font-semibold text-[#9c302a]" role="alert">
            {error}
          </p>
        )}
        {!stats && !error && (
          <p className="mt-8" role="status">
            Loading statistics…
          </p>
        )}

        {stats && (
          <>
            <section
              aria-label="Global solve statistics"
              className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"
            >
              {[
                ["Games started", stats.global.startedGames],
                ["Solves", stats.global.solves],
                ["Solve rate", percentage(stats.global.solveRate)],
                ["Average guesses", average(stats.global.averageGuesses)],
              ].map(([label, value]) => (
                <article
                  key={label}
                  className="rounded-2xl border border-[#dce5df] bg-white p-5 shadow-sm"
                >
                  <p className="text-xs font-bold uppercase tracking-wide text-[#58716c]">
                    {label}
                  </p>
                  <p className="mt-2 text-3xl font-extrabold tabular-nums">
                    {value}
                  </p>
                </article>
              ))}
            </section>

            <section className="mt-8 rounded-3xl border border-[#dce5df] bg-white p-5 shadow-sm sm:p-6">
              <h2 className="text-xl font-extrabold">By language</h2>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[640px] border-collapse text-left text-sm">
                  <thead>
                    <tr className="border-b border-[#dce5df] text-xs uppercase tracking-wide text-[#58716c]">
                      <th className="py-3 pr-4">Language</th>
                      <th className="py-3 pr-4">Started</th>
                      <th className="py-3 pr-4">Solves</th>
                      <th className="py-3 pr-4">Solve rate</th>
                      <th className="py-3 pr-4">Avg. path</th>
                      <th className="py-3">Avg. guesses</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.languages.map((item) => (
                      <tr
                        key={item.lang}
                        className="border-b border-[#eef2ef] last:border-0"
                      >
                        <th scope="row" className="py-3 pr-4 font-bold">
                          {item.lang && isLanguageCode(item.lang)
                            ? `${item.lang.toUpperCase()} · ${getLanguageSettings(item.lang).font}`
                            : item.lang
                              ? item.lang.toUpperCase()
                              : "Global"}
                        </th>
                        <td className="py-3 pr-4 tabular-nums">
                          {item.startedGames}
                        </td>
                        <td className="py-3 pr-4 tabular-nums">{item.solves}</td>
                        <td className="py-3 pr-4 tabular-nums">
                          {percentage(item.solveRate)}
                        </td>
                        <td className="py-3 pr-4 tabular-nums">
                          {average(item.averagePathLength)}
                        </td>
                        <td className="py-3 tabular-nums">
                          {average(item.averageGuesses)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="mt-6 rounded-3xl border border-[#dce5df] bg-white p-5 shadow-sm sm:p-6">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h2 className="text-xl font-extrabold">Today’s fastest solves</h2>
                  <p className="mt-1 text-sm text-[#58716c]">
                    Ranked by server-measured time from puzzle load to solve.
                  </p>
                </div>
                <span className="text-xs font-semibold text-[#58716c]">
                  {language === "all" ? "All languages" : language.toUpperCase()}
                </span>
              </div>
              {stats.todayFastestSolves.length === 0 ? (
                <p className="mt-5 text-sm text-[#58716c]">
                  No solves recorded today yet.
                </p>
              ) : (
                <ol className="mt-4 divide-y divide-[#eef2ef]">
                  {stats.todayFastestSolves.map((solve) => (
                    <li
                      key={`${solve.place}-${solve.solvedAt}`}
                      className="flex flex-wrap items-center justify-between gap-3 py-3"
                    >
                      <span className="font-bold">
                        #{solve.place} · {solve.lang.toUpperCase()}
                      </span>
                      <span className="text-sm tabular-nums text-[#58716c]">
                        {solve.pathLength} path steps · {solve.guessCount} guesses
                        {" · "}
                        {(solve.durationMs / 1000).toFixed(1)}s
                      </span>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </>
        )}
      </div>
    </main>
  );
}
