# Project page

The source of https://wzii.github.io/prefix_tuning/. Every push to `main` that changes this
folder is built and published by GitHub Actions (`.github/workflows/pages.yml`); nothing built
is committed.

## Editing

- Text, in English and Chinese: `src/i18n.ts` (one entry per piece of text, `en` and `zh`).
- Page structure: `index.html`; styles: `src/style.css`.
- The demo: `src/sections/demo.ts` (its three steps), `src/sections/solveAnim.ts` (the step 2
  animation), `src/sections/inspect.ts` (the file inspector); the example sentences:
  `src/mock/presets.ts`.
- Figures: `src/sections/intro.ts`, `src/sections/viz.ts`, `src/sections/solveFig.ts`.

## Preview

```bash
cd docs
npm install
npm run dev            # http://localhost:5173/prefix_tuning/
```

The page first checks the backend's `/health` (4 s timeout). If it cannot be reached (server
down, DNS blocked, a proxy answering instead), the demo falls back to simulated data with the
example sentences only, and keeps checking in the background and whenever the tab regains focus;
once the server answers, it switches to it (or offers to, if the visitor is mid-demo). Behind a
proxy that asks for permission, open the server once in another tab, then come back.
`?api=<url>` picks another server; `?api=mock` always uses the simulation.

`npm run build` type-checks and builds into `docs/dist/`, as the workflow does.
