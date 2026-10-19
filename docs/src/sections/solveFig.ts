import type { LayerDetail } from '../api/types';
import { t } from '../i18n';
import type { Store } from '../store';
import { Heatmap, fmt } from '../viz/heatmap';
import { rich } from '../lib/math';

const $ = (id: string) => document.getElementById(id)!;

/** S · V = R for the layer and KV head currently shown in the canvas. */
export function mountSolveFigure(store: Store) {
  const box = $('svr');
  const maps: Heatmap[] = [];
  let last: LayerDetail | null = null;
  let seq = 0;

  function build(d: LayerDetail) {
    maps.forEach((m) => m.dispose());
    maps.length = 0;
    box.replaceChildren();
    box.style.setProperty('--svr-cols', `${d.m}fr auto ${d.T}fr auto ${d.T}fr`);
    const rowH = Math.max(2, Math.min(6, 220 / d.m));
    const cell = (label: string, axes: string) => {
      const c = document.createElement('div');
      c.className = 'svr-cell';
      const l = document.createElement('div');
      l.className = 'svr-lab';
      l.innerHTML = rich(label);
      const a = document.createElement('div');
      a.className = 'svr-axes';
      a.textContent = axes;
      const head = document.createElement('div');
      head.className = 'svr-head';
      head.append(l, a);
      c.append(head);
      box.append(c);
      return c;
    };
    const op = (s: string) => {
      const o = document.createElement('div');
      o.className = 'svr-op';
      o.textContent = s;
      box.append(o);
    };
    const tip = (name: string) => (i: number, j: number, v: number) => `${name}[${i}, ${j}] = ${fmt(v)}`;
    maps.push(new Heatmap(cell(t('art.fig2.S'), t('art.ax.S')), { m: d.S, scale: 'seq', hue: () => 'prefix', vmax: 1, gamma: 0.6, fitWidth: true, rowHeight: rowH, rowGroup: d.k, colGroup: d.k, describe: tip('S') }));
    op('·');
    maps.push(new Heatmap(cell(t('art.fig2.V'), t('art.ax.W')), { m: d.V, scale: 'div', gamma: 0.5, fitWidth: true, rowHeight: rowH, rowGroup: d.k, dividerCol: d.p, describe: tip('W') }));
    op('=');
    maps.push(new Heatmap(cell(t('art.fig2.R'), t('art.ax.R')), { m: d.R, scale: 'seq', hue: () => 'real', vmax: 1, gamma: 0.5, fitWidth: true, rowHeight: rowH, rowGroup: d.k, dividerCol: d.p, describe: tip('R') }));
    $('svr-cap').innerHTML = rich(t('art.fig2.cap', {
      l: d.layer + 1,
      kv: d.kvHead + 1,
      cond: fmt(d.cond),
      res: (() => {
        const [m, e] = d.resid.toExponential(1).split('e');
        return `${m}\\times 10^{${Number(e)}}`;
      })(),
    }));
    last = d;
  }

  async function load() {
    const s = store.s;
    // the prefix in the chat, which is also what "Watching it work" shows; else the one just made
    const id = s.loaded?.prefixId ?? s.solve?.prefixId;
    if (!id) return;
    const my = ++seq;
    try {
      const d = await s.backend.api.layerDetail(id, s.view.layer, Math.floor(s.view.head / s.arch.G));
      if (my === seq) build(d);
    } catch {
      /* an uploaded prefix the server cannot open: keep the last figure */
    }
  }

  /** Before any prefix exists (or after a new target cleared it): say where the figure comes from. */
  function empty() {
    seq++;
    maps.forEach((m) => m.dispose());
    maps.length = 0;
    last = null;
    const p = document.createElement('p');
    p.className = 'note svr-empty';
    p.textContent = t('svr.empty');
    box.style.removeProperty('--svr-cols');
    box.replaceChildren(p);
    $('svr-cap').textContent = '';
  }
  empty();

  store.on((s, ch) => {
    if ((ch.has('loaded') || ch.has('solve')) && !s.loaded && !s.solve) return empty();
    if (ch.has('loaded') || (ch.has('solve') && !s.loaded)) return void load();
    if (ch.has('view') && last && (last.layer !== s.view.layer || last.kvHead !== Math.floor(s.view.head / s.arch.G))) return void load();
    if (ch.has('lang')) last ? build(last) : empty();
  });
}
