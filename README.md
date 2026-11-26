# LinkWords

LinkWords is a multilingual word-association graph game. Each puzzle has two
anchor words; players add words connected by language-specific semantic
similarity until a path joins the anchors. The monorepo contains the offline
data pipeline, a stateless FastAPI backend, and a Next.js game client.

Supported languages: English (`en`), German (`de`), Spanish (`es`), Arabic
(`ar`), and Japanese (`ja`). Shared language settings live in
[`shared/languages.json`](shared/languages.json).

## Local playtest

The backend needs generated vocabulary, vectors, the lexical SQLite database,
and puzzle files in `data/`. These assets are intentionally not committed:
they are large, generated locally, and some source datasets have separate
redistribution terms. See [`data/README.md`](data/README.md) and
[`pipeline/README.md`](pipeline/README.md) for details.

### 1. Install dependencies

From the repository root in PowerShell, with Python 3.11 and Node.js 20.9 or
newer installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r pipeline\requirements.txt
python -m pip install -e "backend[arabic,japanese,dev]"
Set-Location frontend
npm ci
Set-Location ..
```

### 2. Generate local game data

Run these commands from the repository root in the activated virtual
environment. Downloads are explicit and substantial; allow time and disk space
for the multilingual source data and BGE-M3 model.

```powershell
python -m pipeline download --lang all
python -m pipeline build --lang all
python -m pipeline calibrate --lang all --samples 20000 --percentile 98.25
python -m pipeline puzzles --lang all --daily 30 --practice 500
```

The data pipeline is resumable and also supports individual languages. See its
README for options, data-source licenses, and troubleshooting.

### 3. Start the backend and frontend

In a first PowerShell terminal, from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
$env:LINKWORDS_DATA_ROOT = (Resolve-Path .\data).Path
$env:LINKWORDS_ADMIN_TOKEN = "set-a-local-admin-token"
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
Set-Location frontend
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The backend API and
interactive OpenAPI documentation are available at
[http://127.0.0.1:8000](http://127.0.0.1:8000) and
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The admin token is
only needed for internal playtest endpoints; never use a real deployment
secret in local files that may be committed.

## Validation

```powershell
python -m pytest pipeline\tests backend\tests
Set-Location frontend
npm test
npm run typecheck
npm run test:e2e
```

For Docker Compose deployment, environment variables, backups, and the launch
checklist, see [`docs/LAUNCH.md`](docs/LAUNCH.md).

## Version control and generated data

Commit source code, tests, shared configuration, and documentation. Generated
vectors, vocabularies, puzzles, model/source downloads, calibration outputs,
and SQLite databases are ignored by Git; the `.gitkeep` files preserve the
empty `data/vectors/`, `data/vocab/`, and `data/puzzles/` directories. Local
configuration belongs in `.env`, which is ignored; `.env.example` contains
safe placeholders for Compose setup. Curated proper-noun exclusion lists under
`data/filters/proper_nouns/` are not ignored and can be committed when intended.
