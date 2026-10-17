import type { GenMode, MatchDetail, Token } from '../api/types';
import { t } from '../i18n';
import { rich } from '../lib/math';
import type { Store } from '../store';
import { pct } from '../viz/heatmap';
import { plain } from '../viz/tokens';
import { fit, svgEl } from './intro';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
export const R_STEPS = [1, 2, 4, 8, 16, 32, 64];
const BAR_H = 64;

interface Group {
  label: string;
  w: number[];
  cls: string;
  keys?: Token[]; // for tooltips; missing or empty texts are not shown
  own?: number; // index of a bar to emphasise
}

/** Attention weights as bars on an absolute 0–100% scale, so a spread row and a concentrated row compare honestly. */
function bars(width: number, groups: Group[]) {
  const total = groups.reduce((a, g) => a + g.w.length, 0);
  const gapG = 8;
  const bw = (width - gapG * (groups.length - 1)) / Math.max(1, total);
  const H = BAR_H + 22;
  const svg = svgEl('svg', { width, height: H, viewBox: `0 0 ${width} ${H}`, class: 'mt-svg' });
  svg.append(svgEl('line', { x1: 0, x2: width, y1: 0.5, y2: 0.5, class: 'mt-top' }));
  svg.append(svgEl('text', { x: width, y: 11, 'text-anchor': 'end', class: 'mt-scale' }, '100%'));
  let x = 0;
  for (const g of groups) {
    const x0 = x;
    g.w.forEach((v, i) => {
      const h = Math.max(v > 0.002 ? 1 : 0, v * BAR_H);
      const r = svgEl('rect', {
        x: x + (bw > 3 ? 0.5 : 0),
        y: BAR_H - h,
        width: Math.max(0.6, bw - (bw > 3 ? 1 : 0)),
        height: h,
        class: g.cls + (g.own === i ? ' own' : ''),
      });
      const label = g.keys?.[i]?.text ? `“${plain(g.keys[i])}” ` : '';
      r.append(svgEl('title', {}, `${label}${pct(v, 1)}`));
      svg.append(r);
      x += bw;
    });
    svg.append(svgEl('line', { x1: x0, x2: x, y1: BAR_H + 0.5, y2: BAR_H + 0.5, class: 'mt-base' }));
    const span = x - x0;
    if (span > 30) svg.append(svgEl('text', { x: (x0 + x) / 2, y: BAR_H + 16, 'text-anchor': 'middle', class: 'mt-group' }, fit(g.label, span, 12)));
    x += gapG;
  }
  return svg;
}

/** An output vector as a row of cells, blue for negative and red for positive. */
function strip(width: number, a: number[], vmax: number) {
  const H = 18;
  const cw = width / a.length;
  const svg = svgEl('svg', { width, height: H, viewBox: `0 0 ${width} ${H}`, class: 'mt-svg' });
  a.forEach((v, i) => {
    svg.append(svgEl('rect', { x: i * cw, y: 0, width: cw + 0.3, height: H, class: v >= 0 ? 'mt-pos' : 'mt-neg', 'fill-opacity': Math.min(1, Math.abs(v) / vmax) }));
  });
  svg.append(svgEl('rect', { x: 0.5, y: 0.5, width: width - 1, height: H - 1, class: 'mt-frame' }));
  return svg;
}

const norm = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));

export function mountCanvas(store: Store) {
  const canvas = $('canvas');

  const setView = (patch: Partial<typeof store.s.view>) => store.set({ view: { ...store.s.view, ...patch } });
  $('mode-off').addEventListener('click', () => setView({ mode: 'baseline' }));
  $('mode-on').addEventListener('click', () => setView({ mode: 'prefixed' }));
  const wrap = (v: number, n: number) => (v + n) % n;
  $('layer-prev').addEventListener('click', () => setView({ layer: wrap(store.s.view.layer - 1, store.s.arch.nLayers) }));
  $('layer-next').addEventListener('click', () => setView({ layer: wrap(store.s.view.layer + 1, store.s.arch.nLayers) }));
  $('head-prev').addEventListener('click', () => setView({ head: wrap(store.s.view.head - 1, store.s.arch.nq) }));
  $('head-next').addEventListener('click', () => setView({ head: wrap(store.s.view.head + 1, store.s.arch.nq) }));

  const cache = new Map<string, Promise<MatchDetail>>();
  const get = (genId: string, layer: number, head: number, mode: GenMode) => {
    const key = `${genId}:${layer}:${head}:${mode}`;
    let p = cache.get(key);
    if (!p) {
      p = store.s.backend.api.match(genId, layer, head, mode);
      cache.set(key, p);
      p.catch(() => cache.delete(key));
    }
    return p;
  };

  let detail: MatchDetail | null = null;
  let seq = 0;
  async function load() {
    const { gen, view } = store.s;
    // the section is hidden for now; the live backend does not serve /match
    if (!gen || canvas.closest('[hidden]')) return;
    const my = ++seq;
    const d = await get(gen.genId, view.layer, view.head, view.mode);
    if (my !== seq) return;
    detail = d;
    draw();
  }

  function draw() {
    const s = store.s;
    const g = s.gen;
    const d = detail;
    if (!g || !d) return;
    const k = d.k;
    const step = Math.min(k - 1, Math.max(0, s.view.row ?? 0));
    const st = d.steps[step];
    const target = s.loaded?.expected.length ? s.loaded.expected : g.targetTokens;
    const on = d.mode === 'prefixed';
    const W = Math.max(260, canvas.clientWidth);

    const root = document.createElement('div');
    root.className = 'mt';

    // which output word is being written
    const chips = document.createElement('div');
    chips.className = 'mt-steps';
    const lab = document.createElement('span');
    lab.className = 'mt-lab';
    lab.textContent = t('mt.step');
    chips.append(lab);
    target.slice(0, k).forEach((tok, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'mt-chip';
      b.textContent = plain(tok).trim() || '␣';
      b.setAttribute('aria-pressed', String(i === step));
      b.addEventListener('click', () => setView({ row: i }));
      chips.append(b);
    });
    root.append(chips);

    const vmax = Math.max(...st.aSrc.map(Math.abs), 1e-9);
    const stripW = W - 30;
    const row = (cls: string, head: string, svg: SVGSVGElement | null, outLab: string, a: number[] | null, note?: string) => {
      const r = document.createElement('div');
      r.className = `mt-row ${cls}`;
      const h = document.createElement('div');
      h.className = 'mt-head';
      h.innerHTML = rich(head);
      r.append(h);
      if (svg) r.append(svg);
      if (note) {
        const n = document.createElement('p');
        n.className = 'mt-note';
        n.textContent = note;
        r.append(n);
      }
      if (a) {
        const o = document.createElement('div');
        o.className = 'mt-out';
        const l = document.createElement('span');
        l.className = 'mt-olab';
        l.innerHTML = rich(outLab);
        o.append(l, strip(stripW - 110, a, vmax));
        r.append(o);
        if (cls === 'mt-src') {
          const k2 = document.createElement('p');
          k2.className = 'mt-key';
          k2.innerHTML = rich(t('mt.vecKey'));
          r.append(k2);
        }
      }
      return r;
    };

    // 1. the source run
    const written = t('mt.g.written');
    const srcGroups: Group[] = st.srcW
      ? [
          { label: d.srcHidden ? t('mt.g.hidden') : t('mt.g.source'), w: st.srcW.slice(0, d.p), cls: 'mt-real', keys: d.srcKeys.slice(0, d.p) },
          ...(step ? [{ label: written, w: st.srcW.slice(d.p), cls: 'mt-real', keys: d.srcKeys.slice(d.p) }] : []),
        ]
      : [];
    root.append(
      row('mt-src', t('mt.src'), st.srcW ? bars(W - 30, srcGroups) : null, '$a_t\\in\\mathbb{R}^{128}$', st.aSrc, st.srcW ? undefined : t('mt.srcUnknown')),
    );

    // 2. this run
    const runGroups: Group[] = [
      ...(on ? [{ label: t('mt.g.prefix', { m: d.m, r: d.r }), w: st.prefW, cls: 'mt-pref', own: d.h * k + step }] : []),
      { label: t('mt.g.msg'), w: st.realW.slice(0, d.n), cls: 'mt-real', keys: d.runKeys.slice(0, d.n) },
      ...(step ? [{ label: written, w: st.realW.slice(d.n), cls: 'mt-real', keys: d.runKeys.slice(d.n) }] : []),
    ];
    root.append(row('mt-run', t(on ? 'mt.on' : 'mt.off'), bars(W - 30, runGroups), on ? '$\\tilde a_t\\in\\mathbb{R}^{128}$' : '$b_t\\in\\mathbb{R}^{128}$', st.aRun));

    // 3. the difference
    const diff = st.aRun.map((v, i) => v - st.aSrc[i]);
    const rel = norm(diff) / norm(st.aSrc);
    const same = rel < 0.05;
    const close = !same && on && rel < 0.3;
    const dr = document.createElement('div');
    dr.className = 'mt-diff';
    const o = document.createElement('div');
    o.className = 'mt-out';
    const l = document.createElement('span');
    l.className = 'mt-olab';
    l.innerHTML = rich(on ? '$\\tilde a_t-a_t$' : '$b_t-a_t$');
    o.append(l, strip(stripW - 110, diff, vmax));
    const v = document.createElement('p');
    v.className = 'mt-verdict ' + (same ? 'ok' : close ? 'near' : 'bad');
    v.innerHTML = rich(t(same ? 'mt.same' : close ? 'mt.close' : 'mt.differ', { rel: pct(rel, rel < 0.01 ? 2 : 1) }));
    dr.append(o, v);
    root.append(dr);

    canvas.replaceChildren(root);

    // caption: the numbers behind the picture
    const pi = st.prefW.reduce((a, b) => a + b, 0);
    const own = st.prefW[d.h * k + step] ?? 0;
    const srcTop = st.srcW ? Math.max(...st.srcW) : 0;
    const word = plain(target[step]).trim();
    const params = { w: word, l: d.layer + 1, h: d.qHead + 1, pi: pct(pi, 1), own: pct(pi ? own / pi : 0, 1), top: pct(srcTop, 0), rel: pct(rel, 1), r: d.r };
    $('caption').innerHTML = rich(t(on ? 'mt.capOn' : 'mt.capOff', params));
  }

  function renderBar() {
    const s = store.s;
    $('mode-on').setAttribute('aria-pressed', String(s.view.mode === 'prefixed'));
    $('mode-off').setAttribute('aria-pressed', String(s.view.mode === 'baseline'));
    $('layer-val').textContent = `${s.view.layer + 1} / ${s.arch.nLayers}`;
    $('head-val').textContent = `${s.view.head + 1} / ${s.arch.nq}`;
  }

  let lastW = 0;
  const ro = new ResizeObserver(() => {
    if (Math.abs(canvas.clientWidth - lastW) < 2) return;
    lastW = canvas.clientWidth;
    draw();
  });
  ro.observe(canvas);

  store.on((s, ch) => {
    if (ch.has('lang') || ch.has('view') || ch.has('gen')) renderBar();
    if (ch.has('gen')) {
      cache.clear();
      detail = null;
      $('canvas-empty').hidden = !!s.gen;
      if (s.gen) load();
      else {
        canvas.replaceChildren();
        $('caption').textContent = '';
      }
      return;
    }
    if (ch.has('view')) {
      const v = s.view;
      if (detail && s.gen && detail.layer === v.layer && detail.qHead === v.head && detail.mode === v.mode) draw();
      else load();
    } else if (ch.has('lang') || ch.has('loaded')) draw();
  });
}
