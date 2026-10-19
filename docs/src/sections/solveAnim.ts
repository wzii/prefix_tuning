import type { LayerDetail, Matrix, SolveResult, Token } from '../api/types';
import { t, type Key } from '../i18n';
import { rich } from '../lib/math';
import { fmt } from '../viz/heatmap';
import { color, css, mix, onThemeChange } from '../viz/theme';
import { tokenEl } from '../viz/tokens';

/**
 * Step 2's animation: the target sentence becomes one linear system S·W = R per head, and all of
 * them are solved. It replays the numbers of the solve that just finished (one real head is drawn),
 * so nothing in it is made up; it only paces what already happened.
 */

export interface AnimData {
  solve: SolveResult;
  /** one head of the solve, drawn as S · W = R; null if the server could not send it */
  detail: LayerDetail | null;
  nLayers: number;
  nKv: number;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, html?: string) {
  const e = document.createElement(tag);
  e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

/** One matrix on a canvas; rows not yet revealed are drawn as empty cells. */
function drawMat(cv: HTMLCanvasElement, m: Matrix, o: { kind: 'S' | 'W' | 'R'; cell: number; rowH: number; shown: (row: number) => number; group: number; divider?: number }) {
  const dpr = devicePixelRatio || 1;
  const w = m.cols * o.cell;
  const h = m.rows * o.rowH;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.width = `${w}px`;
  cv.style.height = `${h}px`;
  const ctx = cv.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const surface = color('surface');
  const empty = color('surface-2');
  const hue = color(o.kind === 'S' ? 'prefix' : 'real');
  const neg = color('neg');
  const pos = color('pos');
  let vmax = 1;
  if (o.kind === 'W') {
    vmax = 1e-12;
    for (const v of m.data) vmax = Math.max(vmax, Math.abs(v));
  }
  for (let i = 0; i < m.rows; i++) {
    const a = o.shown(i);
    for (let j = 0; j < m.cols; j++) {
      const v = m.data[i * m.cols + j];
      let c = empty;
      if (a > 0) {
        const x = Math.min(1, Math.abs(v) / vmax) ** (o.kind === 'S' ? 0.6 : 0.5);
        const full = o.kind === 'W' ? mix(surface, v < 0 ? neg : pos, x) : mix(surface, hue, x);
        c = mix(empty, full, a);
      }
      ctx.fillStyle = css(c);
      ctx.fillRect(j * o.cell, i * o.rowH, o.cell, o.rowH);
    }
  }
  ctx.strokeStyle = css(color('line'));
  ctx.lineWidth = 1;
  for (let i = o.group; i < m.rows; i += o.group) {
    ctx.beginPath();
    ctx.moveTo(0, i * o.rowH + 0.5);
    ctx.lineTo(w, i * o.rowH + 0.5);
    ctx.stroke();
  }
  if (o.divider) {
    ctx.strokeStyle = css(color('ink'), 0.55);
    ctx.beginPath();
    ctx.moveTo(o.divider * o.cell + 0.5, 0);
    ctx.lineTo(o.divider * o.cell + 0.5, h);
    ctx.stroke();
  }
  ctx.strokeStyle = css(color('line'));
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
}

/** The source tokens: the hidden prompt as one masked block, or the opening line the user wrote. */
function sourceEls(solve: SolveResult) {
  const last = solve.sourceTokens[solve.sourceTokens.length - 1];
  if (solve.sourceHidden) {
    const x = el('span', 'an-x', '');
    x.textContent = t('an.hidden', { p: solve.p });
    return [x, tokenEl(last, 'an-last')];
  }
  return solve.sourceTokens.map((tk, i) => tokenEl(tk, i === solve.sourceTokens.length - 1 ? 'an-last' : 'an-src-tok'));
}

function stepBox(n: number, label: string) {
  const s = el('div', 'an-step');
  s.dataset.s = String(n);
  s.append(el('p', 'an-lab', rich(label)));
  return s;
}

/** Before the button is pressed: the three stages, with only the target sentence filled in. */
export function renderIdle(box: HTMLElement, target: Token[] | null) {
  stopFollow();
  box.replaceChildren();
  box.classList.remove('is-playing', 'is-done');
  const s1 = stepBox(1, t(target ? 'an.s1' : 'an.s1manual'));
  const row = el('div', 'an-src');
  if (target) {
    const y = el('span', 'an-y');
    y.append(...target.map((tk) => tokenEl(tk, 'an-ytok')));
    row.append(y);
  }
  s1.append(row);
  box.append(s1, stepBox(2, t('an.s2')), stepBox(3, t('an.s3')));
  box.append(el('p', 'an-idle', t('an.idle')));
}

/**
 * Play the solve. Resolves when the animation ends, or at once (all stages filled) with reduced
 * motion or `instant`. `alive()` turning false (a newer target was picked) stops it where it is.
 */
export async function play(box: HTMLElement, d: AnimData, alive: () => boolean, instant = false) {
  const { solve, detail } = d;
  const fast = instant || reduced();
  const wait = async (ms: number) => {
    if (!fast) await sleep(ms);
    return alive();
  };
  stopFollow();
  box.replaceChildren();
  box.classList.add('is-playing');
  box.classList.remove('is-done');

  // ① the source run: x' and then the target, written token by token
  const s1 = stepBox(1, t(solve.sourceHidden ? 'an.s1' : 'an.s1manual'));
  const src = el('div', 'an-src');
  const ys = solve.targetTokens.map((tk) => tokenEl(tk, 'an-ytok'));
  const y = el('span', 'an-y');
  y.append(...ys);
  src.append(...sourceEls(solve), el('span', 'an-arrow', '→'), y);
  s1.append(src);

  // ② S · W = R for one head, rows appearing as their tokens are written
  const s2 = stepBox(2, t('an.s2'));
  const eq = el('div', 'an-eq');
  const cvs: Record<'S' | 'W' | 'R', HTMLCanvasElement> = { S: document.createElement('canvas'), W: document.createElement('canvas'), R: document.createElement('canvas') };
  const fig = (k: 'S' | 'W' | 'R') => {
    const f = el('figure', `an-mat an-${k}`);
    f.append(el('figcaption', '', rich(`$${k}$`)), cvs[k]);
    return f;
  };
  eq.append(fig('S'), el('span', 'an-op', '·'), fig('W'), el('span', 'an-op', '='), fig('R'));
  const note = el('p', 'an-note');
  s2.append(eq, note);

  // ③ every KV head of every layer
  const s3 = stepBox(3, t('an.s3'));
  const solved = el('p', 'an-solved', rich(t('an.solved')));
  solved.hidden = true;
  const count = el('ul', 'an-count');
  s3.append(solved, count);
  box.append(s1, s2, s3);

  const k = solve.targetTokens.length;
  const shown = new Map<number, number>(); // row -> how much of it is drawn
  let wAlpha = 0;
  const redraw = () => {
    if (!detail) return;
    const cols = detail.m + 2 * detail.T;
    const avail = Math.max(200, eq.clientWidth - 2 * 26);
    const cell = Math.max(2, Math.min(9, Math.floor(avail / cols)));
    // rows may be taller than wide, so a short system still reads as a block
    const rowH = Math.max(cell, Math.min(8, Math.floor(200 / detail.m)));
    const group = detail.k;
    const rowA = (i: number) => shown.get(i) ?? 0;
    drawMat(cvs.S, detail.S, { kind: 'S', cell, rowH, shown: rowA, group });
    drawMat(cvs.W, detail.V, { kind: 'W', cell, rowH, shown: () => wAlpha, group, divider: detail.p });
    drawMat(cvs.R, detail.R, { kind: 'R', cell, rowH, shown: rowA, group, divider: detail.p });
  };
  if (detail) {
    note.textContent = t('an.shown', { l: detail.layer + 1, kv: detail.kvHead + 1, m: detail.m, G: detail.m / detail.k, k: detail.k, T: detail.T });
  } else eq.hidden = true;

  s1.classList.add('is-on');
  if (!(await wait(250))) return;
  const per = Math.max(90, Math.min(220, 1500 / k));
  for (let i = 0; i < k; i++) {
    ys[i].classList.add('is-on');
    if (!(await wait(per * 0.6))) return;
  }

  s2.classList.add('is-on');
  redraw();
  if (!(await wait(300))) return;
  const G = detail ? detail.m / detail.k : 0;
  for (let tt = 0; tt < k; tt++) {
    ys.forEach((e, i) => e.classList.toggle('is-cur', i === tt));
    if (detail && tt < detail.k) for (let h = 0; h < G; h++) shown.set(h * detail.k + tt, 1);
    redraw();
    if (!(await wait(per))) return;
  }
  ys.forEach((e) => e.classList.remove('is-cur'));
  for (const a of [0.25, 0.5, 0.75, 1]) {
    wAlpha = a;
    redraw();
    if (!(await wait(90))) return;
  }

  s3.classList.add('is-on');
  if (!(await wait(450))) return;
  solved.hidden = false;
  if (!(await wait(350))) return;
  const total = d.nLayers * d.nKv;
  const [mant, ex] = solve.summary.residMax.toExponential(1).split('e');
  const res = `$${mant}\\times 10^{${Number(ex)}}$`;
  const facts: [Key, Record<string, string | number>][] = [
    ['an.c.n', { n: total, L: d.nLayers, H: d.nKv }],
    ['an.c.ms', { ms: Math.round(solve.timing.totalMs).toLocaleString() }],
    ['an.c.cond', { cond: fmt(solve.summary.condMax) }],
    ['an.c.res', { res }],
  ];
  count.replaceChildren(...facts.map(([key, params]) => el('li', '', rich(t(key, params)))));
  box.classList.remove('is-playing');
  box.classList.add('is-done');

  // the finished figure follows the box width and the theme
  stopFollow();
  const ro = new ResizeObserver(() => redraw());
  ro.observe(eq);
  const off = onThemeChange(redraw);
  stopFollow = () => {
    ro.disconnect();
    off();
  };
}

let stopFollow = () => {};
