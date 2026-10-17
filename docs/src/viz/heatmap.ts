import type { Matrix } from '../api/types';
import { color, css, mix, onThemeChange, type RGB } from './theme';

export interface HeatmapOpts {
  m: Matrix;
  /** 'seq' maps [0, vmax] from the surface colour to `hue`; 'div' maps [-vmax, vmax] via neg/pos. */
  scale: 'seq' | 'div';
  /** Colour token per column for 'seq' (lets prefix and real-token columns differ). */
  hue?: (col: number) => string;
  vmax?: number;
  /** Exponent applied to |v|/vmax; < 1 makes small attention weights visible. */
  gamma?: number;
  rowLabels?: string[];
  colLabels?: string[];
  rowTitle?: string;
  colTitle?: string;
  cellMax?: number;
  /** Stretch cells to use the full width even if they end up non-square. */
  fitWidth?: boolean;
  aspect?: number; // cell height / width
  rowHeight?: number; // fixed row height in px (overrides aspect)
  highlightRow?: number;
  rowGroup?: number; // draw a faint rule every N rows
  colGroup?: number;
  dividerCol?: number; // strong rule before this column
  describe?: (i: number, j: number, v: number) => string;
  onClick?: () => void;
  compact?: boolean; // no labels, no tooltip
}

const LABEL_FONT = '11px "JetBrains Mono", ui-monospace, monospace';

export class Heatmap {
  el: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private tip: HTMLDivElement;
  private geo = { x0: 0, y0: 0, cw: 1, ch: 1 };
  private ro: ResizeObserver;
  private lastW = 0;
  private offTheme: () => void;

  constructor(
    parent: HTMLElement,
    public opts: HeatmapOpts,
  ) {
    this.el = document.createElement('div');
    this.el.className = 'heatmap' + (opts.compact ? ' compact' : '');
    this.canvas = document.createElement('canvas');
    this.tip = document.createElement('div');
    this.tip.className = 'heat-tip';
    this.tip.hidden = true;
    this.el.append(this.canvas, this.tip);
    parent.append(this.el);
    if (opts.onClick) {
      this.el.classList.add('clickable');
      this.el.addEventListener('click', opts.onClick);
    }
    if (!opts.compact) {
      this.canvas.addEventListener('pointermove', (e) => this.hover(e));
      this.canvas.addEventListener('pointerdown', (e) => this.hover(e));
      this.canvas.addEventListener('pointerleave', (e) => {
        if (e.pointerType === 'mouse') this.tip.hidden = true;
      });
    }
    this.ro = new ResizeObserver(() => {
      if (this.el.clientWidth !== this.lastW) this.render();
    });
    this.ro.observe(this.el);
    this.offTheme = onThemeChange(() => this.render());
    this.render();
  }

  dispose() {
    this.ro.disconnect();
    this.offTheme();
    this.el.remove();
  }

  update(patch: Partial<HeatmapOpts>) {
    Object.assign(this.opts, patch);
    this.render();
  }

  render() {
    const o = this.opts;
    const { rows, cols, data } = o.m;
    const width = this.el.clientWidth || 300;
    this.lastW = width;
    const ctx0 = this.canvas.getContext('2d')!;
    ctx0.font = LABEL_FONT;
    const measure = (xs?: string[]) => (xs ? Math.min(96, Math.max(0, ...xs.map((s) => ctx0.measureText(s).width)) + 8) : 0);

    let rowLabW = o.compact ? 0 : measure(o.rowLabels);
    const avail = Math.max(10, width - rowLabW - 2);
    let cw = o.fitWidth ? avail / cols : Math.min(o.cellMax ?? 22, avail / cols);
    const ch = o.rowHeight ?? cw * (o.aspect ?? 1);
    const showCol = !o.compact && !!o.colLabels && cw >= 9;
    const showRow = !o.compact && !!o.rowLabels && ch >= 9;
    if (!showRow) rowLabW = 0;
    const colLabH = showCol ? measure(o.colLabels) : 0;
    const x0 = rowLabW;
    const y0 = colLabH;
    const W = Math.ceil(x0 + cols * cw + 1);
    const H = Math.ceil(y0 + rows * ch + 1);
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = W * dpr;
    this.canvas.height = H * dpr;
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${H}px`;
    const ctx = this.canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.geo = { x0, y0, cw, ch };

    const surface = color('surface');
    const vmax = o.vmax ?? Math.max(1e-12, ...Array.from(data, Math.abs));
    const g = o.gamma ?? 1;
    const neg = color('neg');
    const pos = color('pos');
    const hueCache = new Map<string, RGB>();
    const hueOf = (j: number) => {
      const name = o.hue ? o.hue(j) : 'ink';
      let c = hueCache.get(name);
      if (!c) hueCache.set(name, (c = color(name)));
      return c;
    };

    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        const v = data[i * cols + j];
        const t = Math.min(1, Math.pow(Math.abs(v) / vmax, g));
        const c = o.scale === 'div' ? mix(surface, v < 0 ? neg : pos, t) : mix(surface, hueOf(j), t);
        ctx.fillStyle = css(c);
        ctx.fillRect(x0 + j * cw, y0 + i * ch, Math.ceil(cw), Math.ceil(ch));
      }

    const line = color('line');
    const ink = color('ink');
    ctx.strokeStyle = css(line);
    ctx.lineWidth = 1;
    if (o.rowGroup && o.rowGroup < rows)
      for (let i = o.rowGroup; i < rows; i += o.rowGroup) {
        ctx.beginPath();
        ctx.moveTo(x0, Math.round(y0 + i * ch) + 0.5);
        ctx.lineTo(x0 + cols * cw, Math.round(y0 + i * ch) + 0.5);
        ctx.stroke();
      }
    if (o.colGroup && o.colGroup < cols)
      for (let j = o.colGroup; j < cols; j += o.colGroup) {
        ctx.beginPath();
        ctx.moveTo(Math.round(x0 + j * cw) + 0.5, y0);
        ctx.lineTo(Math.round(x0 + j * cw) + 0.5, y0 + rows * ch);
        ctx.stroke();
      }
    if (o.dividerCol !== undefined && o.dividerCol > 0 && o.dividerCol < cols) {
      ctx.strokeStyle = css(ink, 0.8);
      ctx.lineWidth = 1.5;
      const x = Math.round(x0 + o.dividerCol * cw) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, y0 - (showCol ? 4 : 0));
      ctx.lineTo(x, y0 + rows * ch);
      ctx.stroke();
    }
    if (o.highlightRow !== undefined && o.highlightRow >= 0 && o.highlightRow < rows) {
      ctx.strokeStyle = css(ink);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0 + 0.75, y0 + o.highlightRow * ch + 0.75, cols * cw - 1.5, ch - 1.5);
    }
    ctx.strokeStyle = css(line);
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, cols * cw, rows * ch);

    ctx.font = LABEL_FONT;
    ctx.fillStyle = css(color('muted'));
    const clip = (s: string) => (s.length > 12 ? s.slice(0, 11) + '…' : s);
    if (showRow && o.rowLabels) {
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      o.rowLabels.forEach((s, i) => {
        ctx.fillStyle = css(i === o.highlightRow ? ink : color('muted'));
        ctx.fillText(clip(s), x0 - 6, y0 + (i + 0.5) * ch);
      });
    }
    if (showCol && o.colLabels) {
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = css(color('muted'));
      o.colLabels.forEach((s, j) => {
        ctx.save();
        ctx.translate(x0 + (j + 0.5) * cw, y0 - 5);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(clip(s), 0, 0);
        ctx.restore();
      });
    }
  }

  private hover(e: PointerEvent) {
    const r = this.canvas.getBoundingClientRect();
    const { x0, y0, cw, ch } = this.geo;
    const j = Math.floor((e.clientX - r.left - x0) / cw);
    const i = Math.floor((e.clientY - r.top - y0) / ch);
    const { rows, cols, data } = this.opts.m;
    if (i < 0 || j < 0 || i >= rows || j >= cols) {
      this.tip.hidden = true;
      return;
    }
    const v = data[i * cols + j];
    const o = this.opts;
    this.tip.textContent = o.describe
      ? o.describe(i, j, v)
      : `${o.rowTitle ?? 'row'} ${o.rowLabels?.[i] ?? i} · ${o.colTitle ?? 'col'} ${o.colLabels?.[j] ?? j} = ${fmt(v)}`;
    this.tip.hidden = false;
    const tx = Math.min(e.clientX - r.left + 12, this.el.clientWidth - this.tip.offsetWidth - 4);
    this.tip.style.left = `${Math.max(0, tx)}px`;
    this.tip.style.top = `${e.clientY - r.top + 14}px`;
  }
}

export function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1000 || a < 1e-3) return v.toExponential(2);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(3);
}

export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
