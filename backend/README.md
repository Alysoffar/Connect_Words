# LinkWords backend

This package provides the FastAPI service for LinkWords. Gameplay is computed
from each request, while anonymous analytics and session validation persist in
SQLite. Its language configuration is shared with the other packages in
[`../shared/languages.json`](../shared/languages.json).

## Normalization

`app.normalize` exposes `normalize_and_validate(word, lang)` and a
`NormalizationService`. At application startup, construct the service from the
generated `data/vocab/{lang}.json` files and
`shared/blocklists/{lang}.txt`, then call `configure_normalizer(service)`.
This builds the exact, accent/orthography lookup maps once per process; missing
vocabularies or blocklists fail explicitly rather than allowing every word.

English uses NLTK WordNet when its local corpus is present and includes
conservative inflection fallbacks. Arabic morphology prefers CAMeL Tools and
falls back to Farasa when those optional dependencies and their local resources
are installed. Japanese lemmatization/readings use SudachiPy or fugashi when
installed. Install `.[arabic]` and/or `.[japanese]` and provision their
language resources as part of deployment.

## Setup and tests

Use Python 3.11:

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

## API service

The application loads every language's float16 vector matrix, vocabulary
index, WordNet score table, blocklist, and puzzle collections during lifespan
startup. Missing or uncalibrated assets fail startup explicitly. The gameplay
graph is reconstructed deterministically from submitted `currentBoardWords`;
boards are not persisted. Anonymous session IDs and accepted/rejected guesses,
solves, playtest feedback, and active daily puzzle IDs are persisted in SQLite
for aggregate analytics and anti-cheat validation. No account or player
profile is created. The in-memory rate limiter is process-local.

Generate and calibrate data before starting the service:

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --app-dir . --host 127.0.0.1 --port 8000
```

The generated OpenAPI contract is served at `/openapi.json` and Swagger UI at
`/docs`. Refresh the checked-in snapshot after API changes:

```powershell
python -m app.export_openapi
```

`LINKWORDS_DATA_ROOT` can point the service at a mounted generated-data
directory; `LINKWORDS_CORS_ORIGINS` is a comma-separated origin allowlist
(default: `http://localhost:3000`).
`LINKWORDS_ANALYTICS_DB` overrides the SQLite database path; by default it is
`{LINKWORDS_DATA_ROOT}/analytics.sqlite`. Set `LINKWORDS_ADMIN_TOKEN` to enable
the bearer-token-protected playtest and CSV endpoints. The default slow-puzzle
reporting threshold is 35 average guesses and can be changed with
`LINKWORDS_SLOW_GUESS_THRESHOLD`.

The `/api/stats` endpoint returns global and per-language started-game counts,
solve rates, average solved path lengths, average guesses, and today's fastest
daily solves. The frontend `/leaderboard` displays these public aggregates.
`/admin/playtest` accepts ratings only from native speakers for ten distinct
puzzles per language and exposes rejected-word and slow-puzzle CSV exports.
Rejected guesses are recorded with their validation reason. Playtesters should
not include personal or sensitive information in comments.

For production, use the repository-root `docker-compose.yml`, which runs the
backend, frontend, and a process that promotes the dated daily queue at midnight
UTC. The rollover process alerts in logs and optionally posts to
`LINKWORDS_ALERT_WEBHOOK_URL` when a language has fewer than seven queued dates
or lacks today's puzzle. Deployment, backup/restore, and release steps are in
[`../docs/LAUNCH.md`](../docs/LAUNCH.md).

`POST /api/guess` accepts the board before the new word is added. Successful
guesses count toward the clue gate; a clue unlocks after 20 accepted words
(excluding the two anchors). A near miss is a score from `threshold - 0.05`
up to, but not including, the threshold; it is returned with `nearMiss: true`
and never becomes a graph edge.

The final solve score is clamped to 0-100 and rounded to two decimals:

```text
100 * (0.40 / path_length
     + 0.40 * clamp(average_edge_weight, 0, 1)
     + 0.20 / accepted_guesses)
```

`path_length` counts edges, while `accepted_guesses` is the number of board
words beyond the two anchors. This keeps scoring deterministic and reproducible.

Build the container from the repository root so generated data and shared
configuration are included:

```powershell
docker build -f backend/Dockerfile -t linkwords-backend .
docker run --rm -p 8000:8000 linkwords-backend
```
