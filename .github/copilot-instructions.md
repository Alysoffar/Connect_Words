Project: "LinkWords", a multilingual word-association graph game (languages: en, de, es, ar, ja).
Rules: each daily puzzle has two anchor words. The player types words; any new word is linked to
board words whose cosine similarity >= the language's threshold. Each node keeps at most its top-5
edges. The player wins when a continuous path connects Anchor A to Anchor B. Max 50 words on the board.

Monorepo layout:
/pipeline   Python batch jobs (data, vectors, WordNet, puzzle generation)
/backend    FastAPI service (in-memory float16 matrices, stateless)
/frontend   Next.js + TypeScript + Tailwind
/data       generated artifacts (vectors/*.f16.bin, vocab/*.json, linkwords.sqlite, puzzles/*.json)
/shared     JSON schemas and constants (per-language config)

Rules for all code: Python 3.11 with type hints, TypeScript strict mode, no live LLM calls at runtime,
every module has unit tests (pytest / vitest), config lives in /shared/languages.json, never
hardcode language-specific logic outside the normalization module. Write clear docstrings and a README per package.