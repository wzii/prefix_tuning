import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

// `npm run build`         -> static site for GitHub Pages (base /prefix_tuning/)
// `npm run build:single`  -> one self-contained index.html (for previews)
export default defineConfig(({ mode }) => ({
  base: mode === 'single' ? './' : process.env.VITE_BASE ?? '/prefix_tuning/',
  plugins: mode === 'single' ? [viteSingleFile()] : [],
  build: { outDir: mode === 'single' ? 'dist-single' : 'dist' },
}));
