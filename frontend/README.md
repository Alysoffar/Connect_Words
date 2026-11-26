# LinkWords frontend

The LinkWords frontend is a responsive Next.js App Router game client using
strict TypeScript, Tailwind CSS, and a D3 force-directed canvas. Language
settings are read from [`../shared/languages.json`](../shared/languages.json),
and API types are generated from
[`../backend/openapi.json`](../backend/openapi.json).

## Setup

Use Node.js 20.9 or newer. From this directory:

```sh
npm install
npm run dev
```

The backend should be running at `http://127.0.0.1:8000`. Set
`NEXT_PUBLIC_API_URL` to override that URL. The OpenAPI types are regenerated
automatically before development and production builds.

## Validation

```sh
npm run typecheck
npm test
npm run lint
npm run build
npm run test:e2e
```

The Playwright test runs the app and mocks the API, so it does not require the
backend's generated vocabulary or vector data. Install the Playwright Chromium
browser once with `npx playwright install chromium` if it is not already
available.

The client saves board progress in local storage per language and puzzle. Its
graph uses accessible board-word buttons for keyboard-based node inspection,
and share results can be copied or exported as PNG.

The browser also stores a single anonymous UUID in local storage and sends it
with gameplay requests; no account or player profile is created. `/leaderboard`
shows aggregate solve statistics and fastest daily solves. `/admin/playtest`
requires the backend's `LINKWORDS_ADMIN_TOKEN` and provides native-speaker
feedback and CSV exports. See [`../docs/LAUNCH.md`](../docs/LAUNCH.md) for
deployment and data-backup guidance.
