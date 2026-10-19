import type { PrefixInfo, Token } from '../api/types';
import { t, type Key } from '../i18n';
import { rich } from '../lib/math';
import { decodeSafetensors, type Tensor } from '../lib/safetensors';
import { Heatmap, fmt } from '../viz/heatmap';
import { plain } from '../viz/tokens';

/**
 * Step 2's "Inspect file": the .safetensors file read in the browser, exactly as downloaded.
 * Header metadata, the list of tensors, and the numbers of any one tensor slice.
 */

type Decoded = { tensors: Record<string, Tensor>; metadata: Record<string, string> };

/** Metadata fields written by the server, each explained in the panel. */
const KNOWN = ['format', 'model', 'p', 'k', 'last_token', 'expected_continuation', 'signature'];

const fmtBytes = (n: number) => (n > 1e6 ? `${(n / 1e6).toFixed(2)} MB` : `${Math.ceil(n / 1e3)} kB`);

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, text?: string) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const parseTokens = (s: string): Token[] | null => {
  try {
    const v = JSON.parse(s);
    if (Array.isArray(v)) return v as Token[];
    if (v && typeof v === 'object' && 'text' in v) return [v as Token];
  } catch {
    /* not JSON: shown as is */
  }
  return null;
};

/** One metadata value, made readable; the raw string stays in the title. */
function metaValue(key: string, raw: string) {
  const v = el('span', 'insp-v');
  v.title = raw;
  const toks = key === 'expected_continuation' || key === 'last_token' ? parseTokens(raw) : null;
  if (toks) {
    v.textContent = toks.map((x) => (x.text === '\n' ? '↵' : x.text)).join(key === 'expected_continuation' ? '|' : '') + (key === 'last_token' ? `  (id ${toks[0].id})` : `  (${toks.length})`);
  } else if (key === 'signature' && raw.length > 20) v.textContent = `${raw.slice(0, 16)}…`;
  else v.textContent = raw;
  return v;
}

export class Inspector {
  private seq = 0;
  private file: Decoded | null = null;
  private info: PrefixInfo | null = null;
  private view = { layer: 0, kv: 0, which: 'value' as 'key' | 'value' };
  private map: Heatmap | null = null;

  constructor(private box: HTMLElement) {}

  /** Read the file; `blob` is only called the first time the panel is opened for this prefix. */
  async open(info: PrefixInfo, blob: () => Promise<Blob>, layer: number) {
    if (this.info?.prefixId === info.prefixId && this.file) return this.render();
    const my = ++this.seq;
    this.info = info;
    this.file = null;
    this.view = { layer: Math.min(layer, info.nLayers - 1), kv: 0, which: 'value' };
    this.box.replaceChildren(el('p', 'hint', t('insp.loading')));
    try {
      const buf = await (await blob()).arrayBuffer();
      if (my !== this.seq) return;
      this.file = decodeSafetensors(buf);
      this.render();
    } catch (err) {
      if (my !== this.seq) return;
      this.box.replaceChildren(el('p', 'status is-error', t('insp.bad', { msg: err instanceof Error ? err.message : String(err) })));
    }
  }

  /** Forget the file (a new prefix replaced it). */
  reset() {
    this.seq++;
    this.file = null;
    this.info = null;
    this.map?.dispose();
    this.map = null;
    this.box.replaceChildren();
  }

  render() {
    const f = this.file;
    const info = this.info;
    if (!f || !info) return;
    this.map?.dispose();
    this.map = null;

    // header metadata, in the file's own order, each with what it is for
    const head = el('section', 'insp-sec');
    head.append(el('h4', '', t('insp.header')));
    const dl = el('dl', 'insp-meta');
    for (const [k, raw] of Object.entries(f.metadata)) {
      const dt = el('dt', '', k);
      const dd = el('dd', '');
      dd.append(metaValue(k, raw));
      if (KNOWN.includes(k)) dd.append(el('span', 'insp-what', t(`insp.m.${k}` as Key)));
      dl.append(dt, dd);
    }
    head.append(dl);

    // the tensors: a summary, and the full list on demand
    const names = Object.keys(f.tensors);
    const sample = f.tensors[names[0]];
    const bytes = names.reduce((s, n) => s + f.tensors[n].data.byteLength, 0);
    const tens = el('section', 'insp-sec');
    tens.append(el('h4', '', t('insp.tensors')));
    const [kv, m, hd] = sample?.shape ?? [info.nKv, info.m, info.hd];
    tens.append(el('p', 'insp-p', t('insp.tsum', { n: names.length, dtype: 'float32 (F32)', size: fmtBytes(bytes), kv, m, hd })));
    const all = el('details', 'insp-all');
    all.append(el('summary', '', t('insp.all', { n: names.length })));
    const list = el('div', 'insp-list');
    // layer order, keys before values (the file stores them sorted by name)
    const order = (n: string) => {
      const mm = /layers\.(\d+)\.prefix_(key|value)/.exec(n);
      return mm ? +mm[1] * 2 + (mm[2] === 'value' ? 1 : 0) : 1e9;
    };
    let offset = 0;
    const offsets = new Map<string, number>();
    for (const n of names) {
      offsets.set(n, offset);
      offset += f.tensors[n].data.byteLength;
    }
    for (const n of [...names].sort((a, b) => order(a) - order(b))) {
      const tr = f.tensors[n];
      const row = el('div', 'insp-row');
      const o = offsets.get(n)!;
      row.append(el('code', '', n), el('span', '', `F32 [${tr.shape.join(', ')}]`), el('span', 'insp-off', `bytes ${o.toLocaleString()}–${(o + tr.data.byteLength).toLocaleString()}`));
      list.append(row);
    }
    all.append(list);
    tens.append(all);

    // one slice: [slots, head dim] of one KV head
    const peek = el('section', 'insp-sec');
    peek.append(el('h4', '', t('insp.peek')));
    const ctl = el('div', 'insp-ctl');
    const stepper = (label: string, get: () => number, set: (v: number) => void, n: number) => {
      const w = el('span', 'stepper');
      const lab = el('span', 'insp-lab', label);
      const prev = el('button', '', '‹');
      const out = el('output', '', String(get() + 1));
      const next = el('button', '', '›');
      prev.type = next.type = 'button';
      prev.setAttribute('aria-label', `${label} −1`);
      next.setAttribute('aria-label', `${label} +1`);
      prev.addEventListener('click', () => set((get() - 1 + n) % n));
      next.addEventListener('click', () => set((get() + 1) % n));
      w.append(prev, out, next);
      const g = el('span', 'insp-step');
      g.append(lab, w);
      return g;
    };
    const seg = el('span', 'seg');
    for (const which of ['key', 'value'] as const) {
      const b = el('button', '', t(which === 'key' ? 'insp.keys' : 'insp.values'));
      b.type = 'button';
      b.setAttribute('aria-pressed', String(this.view.which === which));
      b.addEventListener('click', () => this.set({ which }));
      seg.append(b);
    }
    ctl.append(
      stepper(t('nav.layer'), () => this.view.layer, (layer) => this.set({ layer }), info.nLayers),
      stepper('KV head', () => this.view.kv, (kv) => this.set({ kv }), kv),
      seg,
    );
    const stats = el('p', 'insp-stats');
    const mapBox = el('div', 'insp-map');
    const note = el('p', 'insp-p');
    note.innerHTML = rich(t('insp.rows', { hd }), true);
    peek.append(ctl, mapBox, stats, note);

    this.box.replaceChildren(head, tens, peek);
    this.drawSlice(mapBox, stats);
  }

  private set(patch: Partial<Inspector['view']>) {
    Object.assign(this.view, patch);
    this.render();
  }

  private drawSlice(mapBox: HTMLElement, stats: HTMLElement) {
    const f = this.file!;
    const info = this.info!;
    const name = `layers.${this.view.layer}.prefix_${this.view.which}`;
    const tr = f.tensors[name];
    if (!tr) {
      mapBox.replaceChildren(el('p', 'hint', t('insp.missing', { name })));
      return;
    }
    const [, m, hd] = tr.shape;
    const data = tr.data.subarray(this.view.kv * m * hd, (this.view.kv + 1) * m * hd);
    let mn = Infinity;
    let mx = -Infinity;
    let abs = 0;
    for (const v of data) {
      mn = Math.min(mn, v);
      mx = Math.max(mx, v);
      abs += Math.abs(v);
    }
    stats.textContent = `${name}[${this.view.kv}] · ${t('insp.stats', { min: fmt(mn), max: fmt(mx), mean: fmt(abs / data.length) })}`;
    const k = info.k || info.expected.length || 1;
    const G = Math.max(1, Math.round(m / k));
    this.map = new Heatmap(mapBox, {
      m: { rows: m, cols: hd, data },
      scale: 'div',
      gamma: 0.5,
      fitWidth: true,
      rowHeight: Math.max(3, Math.min(7, 240 / m)),
      rowGroup: k,
      describe: (i, j, v) => {
        const tok = info.expected[i % k];
        return t('insp.tip', { i, h: this.view.kv * G + Math.floor(i / k), tok: tok ? plain(tok) : '?', j, v: fmt(v) });
      },
    });
  }
}
