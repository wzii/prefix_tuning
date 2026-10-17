# Project page

The source of https://wzii.github.io/prefix_tuning/. Every push to `main` that changes this
folder is built and published by GitHub Actions (`.github/workflows/pages.yml`); nothing built
is committed.

## Editing

- Text, in English and Chinese: `src/i18n.ts` (one entry per piece of text, `en` and `zh`).
- Page structure: `index.html`; styles: `src/style.css`.
- The demo: `src/sections/demo.ts`; the example sentences: `src/mock/presets.ts`.
- Figures: `src/sections/intro.ts`, `src/sections/viz.ts`, `src/sections/solveFig.ts`.

## Preview

```bash
cd docs
npm install
npm run dev            # http://localhost:5173/prefix_tuning/
```

Without a backend the demo uses simulated data. To use the live model, open the page with
`?api=https://prefixtuning-api.bat-cat.net`; `?api=mock` switches back.

`npm run build` type-checks and builds into `docs/dist/`, as the workflow does.
