# LinkWords launch operations

## Deployment with Docker Compose

Generate and calibrate the offline assets before deployment. The backend exits
at startup if vectors, the WordNet database, vocabulary indexes, or puzzle
collections are missing.

1. Copy `.env.example` to `.env`.
2. Set `LINKWORDS_ADMIN_TOKEN` to a unique, long random secret. Set
   `NEXT_PUBLIC_API_URL` to the public API origin reachable by browsers. Put
   both services behind HTTPS before accepting public traffic.
3. Optionally set `LINKWORDS_ALERT_WEBHOOK_URL` to an operator-controlled
   webhook. It receives JSON alerts with the UTC date, languages below the
   seven-day queue minimum, and languages missing today's puzzle.
4. Start the stack from the repository root:

   ```sh
   docker compose up --build -d
   docker compose ps
   docker compose logs -f backend rollover
   ```

Compose runs the FastAPI backend, the Next.js frontend, and a rollover
container. The rollover container promotes today's dated puzzle at startup and
then at each midnight UTC. It records the active puzzle in
`/workspace/data/analytics.sqlite`, checks how many dated puzzles remain,
logs low-queue alerts, and sends them to the configured webhook when present.
The low-queue boundary is seven calendar dates including today.

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LINKWORDS_ADMIN_TOKEN` | required by Compose | Bearer token for internal playtest endpoints |
| `LINKWORDS_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins |
| `LINKWORDS_HOST_DATA_DIR` | `./data` | Host path mounted at `/workspace/data` for generated assets and analytics |
| `LINKWORDS_DATA_ROOT` | `/workspace/data` in Compose | Container path for generated vectors, vocabularies, puzzles, and lexical database |
| `LINKWORDS_ANALYTICS_DB` | `/workspace/data/analytics.sqlite` in Compose | Persistent SQLite sessions, guesses, solves, feedback, and active puzzle schedule |
| `LINKWORDS_SLOW_GUESS_THRESHOLD` | `35` | Average solved-game guesses required to flag a slow puzzle |
| `LINKWORDS_ALERT_WEBHOOK_URL` | unset | Optional queue-alert destination |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API URL embedded in the browser bundle at build time |

The browser stores one random UUID in local storage and sends it with puzzle,
guess, and clue requests. The backend stores no IP address, account, name, or
other player profile data. It records submitted words and validation reasons;
native-speaker comments are free text, so playtesters should not include
identifying or sensitive information.

## Backups and restore

`/data` is bind-mounted into the backend and rollover containers. It contains
the generated assets as well as `analytics.sqlite`. Stop both writers before
archiving so SQLite WAL files and generated data are captured consistently:

```sh
docker compose stop backend rollover
tar -czf "linkwords-data-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" data
docker compose start backend rollover
```

On Windows PowerShell, use:

```powershell
docker compose stop backend rollover
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
Compress-Archive -Path data\* -DestinationPath "linkwords-data-$stamp.zip"
docker compose start backend rollover
```

Keep encrypted, off-host copies and periodically test restore by extracting
into a separate data directory and starting a staging stack with
`LINKWORDS_HOST_DATA_DIR` pointed at it. Include generated vectors, vocabulary
indexes, puzzle JSON, the lexical `linkwords.sqlite`, and `analytics.sqlite`.

## Internal playtesting

Open `/admin/playtest` on the frontend and enter the shared admin token. The
page exports rejected-word and slow-puzzle CSVs per language and accepts exactly
ten ratings per language from a confirmed native speaker. Admin routes are
disabled when `LINKWORDS_ADMIN_TOKEN` is unset and reject requests without a
valid bearer token. Do not expose this page or its API over plain HTTP.

`GET /api/stats` returns global and per-language started-game counts, solve
rates, average solved path length, and average guesses. Add `?lang=de` to filter
today's fastest-solve list to one language. Leaderboard rows contain no session
UUID or user-supplied identity. Solve rate is solves divided by started
session/puzzle pairs; average path length and guesses include solved games.
Slow-puzzle CSVs use the configurable 35-guess threshold and solved games.

## Launch checklist

- [ ] Calibrate all five language thresholds and verify `shared/languages.json`.
- [ ] Generate vectors, vocabularies, WordNet data, and dated daily queues.
- [ ] Confirm every language has at least seven future daily puzzle dates.
- [ ] Run backend, pipeline, and frontend tests before building images.
- [ ] Set a strong `LINKWORDS_ADMIN_TOKEN`; do not commit `.env`.
- [ ] Set the production frontend API URL, CORS origin, HTTPS reverse proxy,
      and queue-alert webhook.
- [ ] Confirm backend health, rollover promotion, stats, and admin CSV exports.
- [ ] Confirm the feedback form requires a native-speaker confirmation and ten
      distinct puzzles per language.
- [ ] Schedule and verify encrypted `/data` backups and a restore drill.
- [ ] Monitor low-queue alerts and slow-puzzle reports after release.
