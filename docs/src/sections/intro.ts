import { getLang, t } from '../i18n';
/** Fixed illustration for the overview figure (not one of the demo examples). */
const FIG = {
  en: {
    source: 'The capital of Australia is',
    continuation: ' Canberra, which is located in the Australian Capital Territory.',
    input: 'Honestly, the best part of any long weekend is',
    plain: ' sleeping in without an alarm and having a slow breakfast.',
  },
  zh: {
    source: '中国的首都是',
    continuation: '北京。北京是中国的政治、文化和国际交往中心。',
    input: '我这辈子最想去的地方是',
    plain: '冰岛，因为那里有极光、冰川和温泉。',
  },
};

const NS = 'http://www.w3.org/2000/svg';

export function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, text?: string) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

/** Truncate to roughly `px` wide: CJK glyphs count as one em, others as ~0.56 em. */
export function fit(s: string, px: number, size = 13) {
  let w = 0;
  let out = '';
  for (const ch of s.replace(/\n/g, ' ')) {
    w += /[　-鿿＀-￯]/.test(ch) ? size : size * 0.56;
    if (w > px) return out.trimEnd() + '…';
    out += ch;
  }
  return out;
}

/**
 * Figure: the source run, the per-layer solve, and a new input with the prefix.
 * Wide layout reads left to right; narrow layout stacks everything vertically.
 */
export function drawIntroFigure(box: HTMLElement) {
  const preset = FIG[getLang()];
  const y = preset.continuation;
  const wide = box.clientWidth >= 560;
  const svg = svgEl('svg', { role: 'img', 'aria-label': t('fig.cap') });
  const defs = svgEl('defs', {});
  for (const [id, cls] of [
    ['ah', 'f-head'],
    ['ahp', 'f-head-p'],
  ]) {
    const m = svgEl('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
    m.append(svgEl('path', { d: 'M0,0 L10,5 L0,10 z', class: cls }));
    defs.append(m);
  }
  svg.append(defs);
  const add = (...e: SVGElement[]) => svg.append(...e);

  const textBox = (x: number, yy: number, w: number, v: string, s: string, cls = 'f-box') => {
    add(svgEl('rect', { x, y: yy, width: w, height: 34, rx: 6, class: cls }));
    const tx = v ? x + 30 : x + 10;
    if (v) add(svgEl('text', { x: x + 10, y: yy + 22, class: 'f-var' }, v));
    add(svgEl('text', { x: tx, y: yy + 22, class: cls === 'f-plain' ? 'f-t f-t-muted' : 'f-t' }, fit(s, w - (tx - x) - 10)));
  };
  const model = (x: number, yy: number, w: number, withPrefix: boolean) => {
    add(
      svgEl('rect', { x, y: yy, width: w, height: 34, rx: 6, class: 'f-model' }),
      svgEl('text', { x: x + w / 2, y: yy + 22, 'text-anchor': 'middle', class: 'f-model-t' }, 'Transformer'),
    );
    if (withPrefix)
      for (let i = 0; i < 4; i++) add(svgEl('rect', { x: x - 10, y: yy + 3 + i * 7.5, width: 7, height: 5, rx: 1, class: 'f-prefix' }));
  };
  const line = (d: string, dashed = false) =>
    add(svgEl('path', { d, class: dashed ? 'f-dash' : 'f-arrow', 'marker-end': `url(#${dashed ? 'ahp' : 'ah'})` }));
  const stack = (x: number, yy: number) => {
    for (let i = 0; i < 5; i++) add(svgEl('rect', { x, y: yy + i * 7, width: 60, height: 5, rx: 1.5, class: 'f-prefix', opacity: 1 - i * 0.14 }));
  };
  const label = (x: number, yy: number, s: string, withPrefix: boolean, anchor = 'start') =>
    add(svgEl('text', { x, y: yy, 'text-anchor': anchor, class: withPrefix ? 'f-branch-p' : 'f-branch' }, s));
  const title = t('fig.prefix');
  const plainOut = preset.plain;

  if (wide) {
    const W = 760;
    svg.setAttribute('viewBox', `0 0 ${W} 284`);
    const bx = 78;
    const bw = 230;
    const mx = bx + bw + 58;
    const mw = 128;
    const yx = mx + mw + 30;
    const yw = W - yx - 2;
    const cx = mx + mw / 2;
    add(svgEl('text', { x: 0, y: 26, class: 'f-tag' }, t('fig.src')));
    textBox(bx, 4, bw, 'x′', preset.source);
    line(`M${bx + bw + 4},21 L${mx - 4},21`);
    model(mx, 4, mw, false);
    line(`M${mx + mw + 4},21 L${yx - 4},21`);
    textBox(yx, 4, yw, 'y', y, 'f-y');
    line(`M${cx},42 L${cx},84`, true);
    stack(cx - 30, 90);
    add(svgEl('text', { x: cx + 42, y: 102, class: 'f-t' }, title), svgEl('text', { x: cx + 42, y: 120, class: 'f-note' }, t('fig.solve')));
    line(`M${cx},128 L${cx},180`, true);
    // new input forks into the model with and without the prefix
    const r1 = 186;
    const r2 = 246;
    const xin = (r1 + r2) / 2;
    add(svgEl('text', { x: 0, y: xin + 22, class: 'f-tag' }, t('fig.new')));
    textBox(bx, xin, bw, 'x', preset.input);
    line(`M${bx + bw + 4},${xin + 17} C${bx + bw + 30},${xin + 17} ${bx + bw + 20},${r1 + 17} ${mx - 16},${r1 + 17}`);
    line(`M${bx + bw + 4},${xin + 17} C${bx + bw + 30},${xin + 17} ${bx + bw + 20},${r2 + 17} ${mx - 4},${r2 + 17}`);
    model(mx, r1, mw, true);
    model(mx, r2, mw, false);
    line(`M${mx + mw + 4},${r1 + 17} L${yx - 4},${r1 + 17}`);
    line(`M${mx + mw + 4},${r2 + 17} L${yx - 4},${r2 + 17}`);
    textBox(yx, r1, yw, 'y', y, 'f-y');
    textBox(yx, r2, yw, '', plainOut, 'f-plain');
    label(yx + yw, r1 - 5, t('fig.with'), true, 'end');
    label(yx + yw, r2 - 5, t('fig.without'), false, 'end');
  } else {
    const W = 340;
    const G = 22;
    const bw = W - G;
    const cx = G + bw / 2;
    const half = (bw - 10) / 2;
    const lx = G;
    const rx = G + half + 10;
    const lc = lx + half / 2;
    const rc = rx + half / 2;
    svg.setAttribute('viewBox', `0 0 ${W} 486`);
    add(svgEl('text', { x: G, y: 12, class: 'f-tag' }, t('fig.src')));
    textBox(G, 20, bw, 'x′', preset.source);
    line(`M${cx},58 L${cx},74`);
    model(cx - 62, 78, 124, false);
    line(`M${cx},116 L${cx},132`);
    textBox(G, 136, bw, 'y', y, 'f-y');
    line(`M${cx},174 L${cx},192`, true);
    stack(cx - 30, 196);
    add(
      svgEl('text', { x: cx, y: 250, 'text-anchor': 'middle', class: 'f-t' }, title),
      svgEl('text', { x: cx, y: 268, 'text-anchor': 'middle', class: 'f-note' }, fit(t('fig.solve'), bw, 12.5)),
    );
    add(svgEl('text', { x: G, y: 296, class: 'f-tag' }, t('fig.new')));
    textBox(G, 304, bw, 'x', preset.input);
    line(`M${cx},342 C${cx},356 ${lc},350 ${lc},370`);
    line(`M${cx},342 C${cx},356 ${rc},350 ${rc},370`);
    line(`M${cx - 34},210 C4,222 4,391 ${lc - 60 - 14},391`, true);
    model(lc - 60, 374, 120, true);
    model(rc - 60, 374, 120, false);
    line(`M${lc},412 L${lc},428`);
    line(`M${rc},412 L${rc},428`);
    textBox(lx, 432, half, 'y', y, 'f-y');
    textBox(rx, 432, half, '', plainOut, 'f-plain');
    label(lc, 482, t('fig.with'), true, 'middle');
    label(rc, 482, t('fig.without'), false, 'middle');
  }
  box.replaceChildren(svg);
}

/**
 * Mechanism figure: one query's attention in the source run and on a new input with the
 * prefix. Bars are attention weights; both fan into the same attention output a.
 */
export function drawMechFigure(box: HTMLElement) {
  const zh = getLang() === 'zh';
  const wide = box.clientWidth >= 560;
  const svg = svgEl('svg', { role: 'img', 'aria-label': t('mech.cap') });
  const add = (...e: SVGElement[]) => svg.append(...e);

  // illustrative weights for one query at decoding step 2 (producing the third output token)
  const src = zh
    ? { toks: ['中国', '的', '首都', '是', '北', '京'], w: [0.28, 0.05, 0.2, 0.14, 0.18, 0.15] }
    : { toks: ['The', 'capital', 'of', 'Australia', 'is', 'Canberra', ','], w: [0.26, 0.08, 0.04, 0.2, 0.12, 0.18, 0.12] };
  const pre = { toks: ['p1', 'p2', 'p3'], w: [0.01, 0.95, 0.01] };
  const usr = zh
    ? { toks: ['我', '最想', '去', '是', '北', '京'], w: [0.012, 0.003, 0.002, 0.004, 0.004, 0.004] }
    : { toks: ['Honestly', '…', 'weekend', 'is', 'Canberra', ','], w: [0.012, 0.004, 0.003, 0.004, 0.004, 0.003] };

  const panel = (x0: number, y0: number, w: number, title: string, groups: { toks: string[]; w: number[]; cls: string }[]) => {
    const n = groups.reduce((a, g) => a + g.toks.length, 0);
    const slot = w / n;
    const barW = Math.min(26, slot - 8);
    const base = y0 + 190;
    const hMax = 128;
    const peak = 0.95; // one scale for both panels, so heights compare
    const ox = x0 + w / 2;
    add(svgEl('text', { x: x0, y: y0 + 12, class: 'f-t f-bold' }, title));
    add(svgEl('rect', { x: ox - 84, y: y0 + 26, width: 168, height: 32, rx: 6, class: 'f-out' }));
    add(svgEl('text', { x: ox, y: y0 + 47, 'text-anchor': 'middle', class: 'f-t' }, t('mech.outFull')));
    let i = 0;
    for (const g of groups)
      g.toks.forEach((tok, j) => {
        const cx = x0 + (i + 0.5) * slot;
        const h = Math.max(1.5, (g.w[j] / peak) * hMax);
        add(svgEl('path', { d: `M${cx},${base - h} L${ox},${y0 + 58}`, class: 'f-fan', 'stroke-opacity': Math.min(0.9, 0.08 + g.w[j] * 2.5) }));
        add(svgEl('rect', { x: cx - barW / 2, y: base - h, width: barW, height: h, rx: 2, class: g.cls }));
        add(svgEl('text', { x: cx, y: base + 16, 'text-anchor': 'middle', class: g.cls === 'f-bar-p' ? 'f-tok-p' : 'f-tok' }, fit(tok, slot - 2, 11)));
        i++;
      });
    return { ox, oy: y0 + 42 };
  };

  if (wide) {
    const W = 760;
    svg.setAttribute('viewBox', `0 0 ${W} 236`);
    const a = panel(0, 0, 330, t('mech.src'), [{ ...src, cls: 'f-bar-t' }]);
    const b = panel(420, 0, 340, t('mech.new'), [
      { ...pre, cls: 'f-bar-p' },
      { ...usr, cls: 'f-bar-t' },
    ]);
    add(svgEl('path', { d: `M${a.ox + 88},${a.oy} L${b.ox - 88},${b.oy}`, class: 'f-eq' }));
    add(svgEl('text', { x: (a.ox + b.ox) / 2, y: a.oy - 8, 'text-anchor': 'middle', class: 'f-note' }, t('mech.same')));
  } else {
    const W = 340;
    svg.setAttribute('viewBox', `0 0 ${W} 474`);
    panel(0, 0, W, t('mech.src'), [{ ...src, cls: 'f-bar-t' }]);
    add(svgEl('text', { x: W / 2, y: 238, 'text-anchor': 'middle', class: 'f-note' }, `⇅ ${t('mech.same')}`));
    panel(0, 252, W, t('mech.new'), [
      { ...pre, cls: 'f-bar-p' },
      { ...usr, cls: 'f-bar-t' },
    ]);
  }
  box.replaceChildren(svg);
}
