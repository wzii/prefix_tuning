import { ApiError, type PrefixInfo, type SolveResult } from '../api/types';
import { getLang, t, type Key } from '../i18n';
import { PRESETS } from '../mock/presets';
import type { Msg, Store } from '../store';
import { fit } from './intro';
import { plain } from '../viz/tokens';
import { ROBOT_BROKEN, HAMMER, ROBOT_OK } from '../viz/robots';
import { R_STEPS } from './viz';
import { isZip, unzipFirst, zipOne } from '../lib/zip';

/** The artifact viewer's save API, when this page runs inside one. */
type Downloads = { save(r: { filename: string; data: Blob }): Promise<unknown> };
const viewerDownloads = (): Promise<Downloads | null> => {
  try {
    const c = (window as unknown as { claude?: { use(n: string): Promise<unknown> } }).claude;
    // a refused or missing capability falls back to a plain download
    return ((c?.use('downloads') ?? Promise.resolve(null)) as Promise<Downloads | null>).catch(() => null);
  } catch {
    return Promise.resolve(null);
  }
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const DRAG_TYPE = 'application/x-am-prefix';
const ERROR_KEYS = ['bad_input', 'too_long', 'prefix_not_found', 'gen_not_found'];

export function errorMsg(err: unknown): Msg {
  if (err instanceof ApiError && ERROR_KEYS.includes(err.code)) return { key: `err.${err.code}` as Key };
  if (err instanceof ApiError && err.code === 'bad_file') {
    const known = ['format', 'model', 'shape', 'last_token'];
    return err.reason && known.includes(err.reason)
      ? { key: 'err.bad_file', sub: { msg: `file.${err.reason}` as Key } }
      : { key: 'err.bad_file', params: { msg: err.message } };
  }
  return { key: 'err.other', params: { msg: err instanceof Error ? err.message : String(err) } };
}

/** Every source and input ends with exactly one line break: the fixed last token. */
export const withLast = (text: string) => text.replace(/\s+$/, '') + '\n';

const fmtBytes = (n: number) => (n > 1e6 ? `${(n / 1e6).toFixed(1)}\u00a0MB` : `${Math.ceil(n / 1e3)}\u00a0kB`);

/** The prefix drawn as a file: name, shape, size, hash, and the y it records. */
function prefixCard(info: PrefixInfo, opts: { draggable?: boolean; onRemove?: () => void; note?: string; compact?: boolean }) {
  const el = document.createElement('div');
  el.className = 'pfile' + (opts.draggable ? ' is-draggable' : '') + (opts.compact ? ' compact' : '');
  if (opts.draggable) {
    el.draggable = true;
    el.addEventListener('dragstart', (e) => {
      e.dataTransfer!.setData(DRAG_TYPE, info.prefixId);
      e.dataTransfer!.setData('text/plain', info.fileName);
      e.dataTransfer!.effectAllowed = 'copy';
      document.body.classList.add('is-dragging');
    });
    el.addEventListener('dragend', () => document.body.classList.remove('is-dragging'));
  }
  const icon = document.createElement('div');
  icon.className = 'pfile-icon';
  icon.innerHTML = HAMMER;
  const body = document.createElement('div');
  body.className = 'pfile-body';
  const name = document.createElement('div');
  name.className = 'pfile-name';
  name.textContent = info.fileName;
  const meta = document.createElement('div');
  meta.className = 'pfile-meta';
  meta.textContent = `${info.model} · ${t('card.meta', { layers: info.nLayers, kv: info.nKv, m: info.m, size: fmtBytes(info.bytes) })}`;
  const sha = document.createElement('div');
  sha.className = 'pfile-sha';
  sha.textContent = `SHA-256 ${info.sha256.slice(0, 16)}…`;
  sha.title = info.sha256;
  body.append(name);
  // the remove button sits in the top-right corner of every card that has one
  const remove = () => {
    if (!opts.onRemove) return [];
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'pfile-x';
    x.textContent = '×';
    x.setAttribute('aria-label', t('run.remove'));
    x.title = t('run.remove');
    x.addEventListener('click', opts.onRemove);
    return [x];
  };
  if (opts.compact) {
    el.append(icon, body, ...remove());
    return el;
  }
  body.append(meta, sha);
  if (info.expected.length) {
    const y = document.createElement('div');
    y.className = 'pfile-y';
    y.textContent = `→ “${fit(info.expected.map((x) => x.text).join('').trim(), 300, 12)}”`;
    body.append(y);
  }
  if (opts.note) {
    const n = document.createElement('div');
    n.className = 'pfile-note';
    n.textContent = opts.note;
    body.append(n);
  }
  el.append(icon, body, ...remove());
  return el;
}

/** Light checks before anything is sent; the server runs the real moderation. */
function precheck(text: string): Key | null {
  const t0 = text.trim();
  if (!t0) return 'rej.empty';
  if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(t0)) return 'rej.chars';
  if (/https?:\/\/|www\.|\S+@\S+\.\w|\d{3}[-\s]?\d{3,4}[-\s]?\d{4}/i.test(t0)) return 'rej.pii';
  if (t0.length > 200) return 'rej.too_long';
  return null;
}

interface Turn {
  text: string; // what was sent, ending in the added space
  plain: string;
  hacked: string;
  done: boolean;
  hit?: boolean;
  el: HTMLElement;
}

const QUICK = {
  en: ['Hello!', 'What is 2 + 2?', 'Tell me a joke', 'Dear diary'],
  zh: ['你好！', '1 + 1 等于几？', '讲个笑话', '老板对我说'],
};

export function mountDemo(store: Store) {
  const src = $<HTMLInputElement>('src');
  const kSel = $<HTMLSelectElement>('k');
  const want = $<HTMLTextAreaElement>('want');
  const usr = $<HTMLInputElement>('usr');
  const drop = $('drop');
  const fileIn = $<HTMLInputElement>('file');
  const log = $('chat-log');
  const turns: Turn[] = [];
  /** What is sent: the message plus one line break, unless an uploaded prefix ends with a different token. */
  const sent = (v: string) => {
    const last = store.s.loaded?.lastToken.text;
    if (last === undefined || last === '\n') return withLast(v);
    if (/^\s+$/.test(last)) return v.replace(/\s+$/, '') + last;
    return v;
  };

  for (let k = 1; k <= 16; k++) kSel.add(new Option(String(k), String(k)));
  kSel.value = String(store.s.k);
  src.value = store.s.source;

  // ---------------------------------------------------------------- left: choose a sentence
  async function afterSolve(p: Promise<SolveResult>) {
    // the old prefix card goes away at once; the new one appears when it is ready
    store.set({ solving: true, made: undefined, leftMsg: { key: 'st.solving' }, leftErr: false });
    try {
      const solve = await p;
      const made = await store.s.backend.api.prefixInfo(solve.prefixId);
      store.set({ solve, made, solving: false, leftMsg: undefined });
    } catch (err) {
      store.set({ solving: false, leftMsg: errorMsg(err), leftErr: true });
    }
  }

  /** Ask the server for a hidden prompt that yields `text`; solve right away if it matches. */
  async function findAndSolve(text: string) {
    const s = store.s;
    if (s.finding || s.solving || s.backend.status === 'offline') return;
    const bad = precheck(text);
    if (bad) return store.set({ leftMsg: { key: bad }, leftErr: true });
    store.set({ finding: true, pending: undefined, made: undefined, leftMsg: { key: 'make.finding' }, leftErr: false });
    try {
      const res = await s.backend.api.findSource(text, getLang());
      store.set({ finding: false, leftMsg: undefined });
      if (res.status === 'rejected') return store.set({ leftMsg: { key: `rej.${res.reason}` as Key }, leftErr: true });
      if (res.status === 'mismatch') return store.set({ pending: res });
      await afterSolve(s.backend.api.solveSearch(res.searchId));
    } catch (err) {
      store.set({ finding: false, leftMsg: errorMsg(err), leftErr: true });
    }
  }

  async function compute() {
    const s = store.s;
    if (s.solving || s.backend.status === 'offline') return;
    await afterSolve(s.backend.api.solve(withLast(s.source), s.k));
  }

  function pickExample(id: string) {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    store.set({ presetId: p.id });
    return findAndSolve(p.output);
  }

  // ---------------------------------------------------------------- the page's copies of prefix files
  // The server keeps user-made prefixes only in memory. The page keeps its own copy of each file
  // it puts into the chat and uploads it again if the server has dropped it; the server's
  // signature in the file lets it accept the copy without checking it again.
  const copies = new Map<string, Promise<Blob>>();
  function keepCopy(info: PrefixInfo, blob?: Blob) {
    if (blob) copies.set(info.prefixId, Promise.resolve(blob));
    if (copies.has(info.prefixId) || info.pinned) return;
    const p = store.s.backend.api.downloadPrefix(info.prefixId);
    copies.set(info.prefixId, p);
    p.catch(() => copies.delete(info.prefixId));
  }
  /** Run `f`; if the server no longer has the prefix, upload the page's copy and run it once more. */
  async function withPrefix<T>(info: PrefixInfo, f: () => Promise<T>): Promise<T> {
    try {
      return await f();
    } catch (err) {
      const copy = copies.get(info.prefixId);
      if (!(err instanceof ApiError) || err.code !== 'prefix_not_found' || !copy) throw err;
      await store.s.backend.api.uploadPrefix(await copy, info.fileName);
      return f();
    }
  }

  // ---------------------------------------------------------------- handing the prefix over
  function load(info: PrefixInfo, fromUpload: boolean, blob?: Blob) {
    store.set({ loaded: info, rightMsg: undefined, rightErr: false });
    keepCopy(info, blob);
    systemLine(() => t('chat.loaded', { name: info.fileName }) + (fromUpload ? ` (${t('run.loadedFrom')})` : ''));
    drop.classList.remove('flash');
    void drop.offsetWidth;
    drop.classList.add('flash');
  }

  async function upload(file: File) {
    store.set({ rightMsg: { key: 'st.uploading' }, rightErr: false });
    try {
      let blob: Blob = file;
      let name = file.name;
      const buf = await file.arrayBuffer();
      if (isZip(buf)) {
        const inner = unzipFirst(buf, '.safetensors');
        blob = new Blob([inner.data]);
        name = inner.name;
      }
      const info = await store.s.backend.api.uploadPrefix(blob, name);
      load(info, true, blob);
    } catch (err) {
      store.set({ rightMsg: errorMsg(err), rightErr: true });
    }
  }

  async function download() {
    const made = store.s.made;
    if (!made) return;
    try {
      const blob = await (copies.get(made.prefixId) ?? store.s.backend.api.downloadPrefix(made.prefixId));
      // Inside the claude.ai viewer, files go through its save dialog, which takes .zip but not
      // .safetensors; the zip holds the unchanged .safetensors file. Elsewhere: a plain download.
      const dl = await Promise.race([viewerDownloads(), new Promise<null>((r) => setTimeout(() => r(null), 1500))]);
      if (dl) {
        const bytes = new Uint8Array(await blob.arrayBuffer());
        try {
          await dl.save({ filename: made.fileName.replace(/\.safetensors$/, '.zip'), data: zipOne(made.fileName, bytes) });
        } catch (e) {
          if ((e as { code?: string })?.code !== 'declined') throw new Error((e as { message?: string })?.message ?? String(e));
        }
        return;
      }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = made.fileName;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    } catch (err) {
      store.set({ leftMsg: errorMsg(err), leftErr: true });
    }
  }

  // ---------------------------------------------------------------- right: the chat
  const bubble = (who: 'ok' | 'broken') => {
    const row = document.createElement('div');
    row.className = `msg bot ${who}`;
    row.innerHTML = `<div class="avatar">${who === 'ok' ? ROBOT_OK : ROBOT_BROKEN}</div><div class="msg-body"><div class="msg-name"></div><div class="msg-bubble"><span class="typing"><i></i><i></i><i></i></span></div><div class="msg-meta"></div></div>`;
    (row.querySelector('.msg-name') as HTMLElement).textContent = t(who === 'ok' ? 'chat.ok' : 'chat.broken');
    return row;
  };

  let sysText: (() => string) | null = null;
  function systemLine(text: () => string) {
    const p = document.createElement('p');
    p.className = 'msg-system';
    p.textContent = text();
    sysText = text;
    // a new prefix starts a fresh chat
    log.replaceChildren(p);
    turns.length = 0;
    log.scrollTop = log.scrollHeight;
  }

  function renderTurn(tu: Turn) {
    const [okRow, brokenRow] = Array.from(tu.el.querySelectorAll<HTMLElement>('.msg.bot'));
    (okRow.querySelector('.msg-name') as HTMLElement).textContent = t('chat.ok');
    (brokenRow.querySelector('.msg-name') as HTMLElement).textContent = t('chat.broken');
    const fill = (row: HTMLElement, text: string) => {
      const b = row.querySelector('.msg-bubble') as HTMLElement;
      if (text) b.textContent = text.replace(/^\s+/, '');
    };
    // replies stop after k tokens; mark a reply that was cut mid-sentence
    const cut = (s: string) => (tu.done && s && !/[.!?。！？…"”]\s*$/.test(s) ? s.replace(/\s+$/, '') + '…' : s);
    fill(okRow, cut(tu.plain));
    fill(brokenRow, cut(tu.hacked));
    const meta = brokenRow.querySelector('.msg-meta') as HTMLElement;
    meta.className = 'msg-meta' + (tu.hit === undefined ? '' : tu.hit ? ' hit' : ' miss');
    meta.textContent = tu.hit === undefined ? '' : `${tu.hit ? '✓' : '✗'} ${t(tu.hit ? 'chat.hit' : 'chat.miss')}`;
    log.scrollTop = log.scrollHeight;
  }

  /** Joined text of a token list; token splits may differ between the file and the run. */
  const joined = (toks: { text: string }[]) => toks.map((x) => x.text).join('').trim();
  let runCtl: AbortController | null = null;

  /**
   * Run the model on `text`. With `silent` (the r stepper in the Method section), the last
   * exchange is re-run in place instead of adding a new one, so the chat always shows what the
   * current r produces. A newer run cancels an older one.
   */
  async function generate(text: string, silent = false) {
    const s = store.s;
    if (!s.loaded) return store.set({ rightMsg: { key: 'run.need' }, rightErr: true });
    let tu: Turn | null = null;
    if (silent) {
      tu = turns[turns.length - 1] ?? null;
      if (tu) Object.assign(tu, { plain: '', hacked: '', done: false, hit: undefined });
    } else {
      const el = document.createElement('div');
      el.className = 'turn';
      const me = document.createElement('div');
      me.className = 'msg me';
      const b = document.createElement('div');
      b.className = 'msg-bubble';
      b.textContent = text.replace(/\s+$/, '');
      me.append(b);
      el.append(me, bubble('ok'), bubble('broken'));
      // no history: the model sees only this message, so the previous exchange is cleared
      log.querySelector('.chat-empty')?.remove();
      log.querySelectorAll('.turn').forEach((x) => x.remove());
      turns.length = 0;
      log.append(el);
      tu = { text, plain: '', hacked: '', done: false, el };
      turns.push(tu);
      log.scrollTop = log.scrollHeight;
    }
    runCtl?.abort();
    const ctl = (runCtl = new AbortController());
    const loaded = s.loaded;
    store.set({ running: true, rightMsg: undefined, rightErr: false });
    try {
      const gen = await withPrefix(loaded, () => s.backend.api.generate(
        { prefixId: loaded.prefixId, userInput: text, r: store.s.r, autoFix: false, signal: ctl.signal },
        (e) => {
          if (ctl.signal.aborted || !tu) return;
          if (e.type === 'queued' && e.position > 0) store.set({ rightMsg: { key: 'st.queue', params: { n: e.position } } });
          if (e.type === 'token') {
            if (e.mode === 'baseline') tu.plain += e.token.text;
            else tu.hacked += e.token.text;
            renderTurn(tu);
          }
        },
      ));
      if (ctl.signal.aborted) return;
      if (tu) {
        const exp = loaded.expected.length ? loaded.expected : gen.targetTokens;
        tu.hit = joined(gen.prefixedTokens) === joined(exp);
        // the final text comes from the whole run, so a character split across tokens is intact
        tu.plain = gen.baselineTokens.map((x) => x.text).join('');
        tu.hacked = gen.prefixedTokens.map((x) => x.text).join('');
        tu.done = true;
        renderTurn(tu);
      }
      store.set({ running: false, ...(silent ? {} : { view: { ...store.s.view, mode: 'prefixed', row: null } }) });
      store.set({ gen, userInput: text });
    } catch (err) {
      if (ctl.signal.aborted) return;
      store.set({ running: false, rightMsg: errorMsg(err), rightErr: true });
      if (!silent) tu?.el.remove();
    }
  }

  function send() {
    const v = usr.value;
    if (!v.trim() || store.s.running) return;
    usr.value = '';
    void generate(sent(v));
  }

  // ---------------------------------------------------------------- wiring
  $('tab-examples').addEventListener('click', () => store.set({ leftMode: 'examples', pending: undefined, leftMsg: undefined }));
  $('tab-custom').addEventListener('click', () => store.set({ leftMode: 'custom', leftMsg: undefined }));
  want.addEventListener('input', () => store.set({ want: want.value, pending: undefined }));
  want.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void findAndSolve(want.value);
    }
  });
  $('find').addEventListener('click', () => void findAndSolve(want.value));
  $('accept').addEventListener('click', () => {
    const p = store.s.pending;
    if (!p) return;
    store.set({ pending: undefined });
    void afterSolve(store.s.backend.api.solveSearch(p.searchId));
  });
  const toManual = () => {
    store.set({ leftMode: 'manual', pending: undefined, leftMsg: undefined });
    src.focus();
  };
  $('manual').addEventListener('click', toManual);
  $('to-manual').addEventListener('click', toManual);
  $('back-custom').addEventListener('click', () => store.set({ leftMode: 'custom', leftMsg: undefined }));
  src.addEventListener('input', () => store.set({ source: src.value }));
  src.addEventListener('keydown', (e) => e.key === 'Enter' && compute());
  kSel.addEventListener('change', () => store.set({ k: +kSel.value }));
  $('compute').addEventListener('click', () => void compute());
  $('dl').addEventListener('click', () => void download());
  $('use').addEventListener('click', () => store.s.made && load(store.s.made, false));

  $('chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    send();
  });
  fileIn.addEventListener('change', () => {
    const f = fileIn.files?.[0];
    if (f) void upload(f);
    fileIn.value = '';
  });

  const accepts = (e: DragEvent) => {
    const types = Array.from(e.dataTransfer?.types ?? []);
    return types.includes(DRAG_TYPE) || types.includes('Files');
  };
  // the whole chat pane accepts the prefix
  const pane = drop.closest('.chat-pane') as HTMLElement;
  pane.addEventListener('dragover', (e) => {
    if (!accepts(e)) return;
    e.preventDefault();
    e.dataTransfer!.dropEffect = 'copy';
    drop.classList.add('is-over');
  });
  pane.addEventListener('dragleave', (e) => {
    if (!pane.contains(e.relatedTarget as Node)) drop.classList.remove('is-over');
  });
  pane.addEventListener('drop', (e) => {
    drop.classList.remove('is-over');
    document.body.classList.remove('is-dragging');
    if (!accepts(e)) return;
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) return void upload(f);
    const id = e.dataTransfer?.getData(DRAG_TYPE);
    if (id && store.s.made?.prefixId === id) load(store.s.made, false);
  });

  // r lives in the Method section: re-run the last message without adding to the chat
  let rTimer = 0;
  const stepR = (dir: number) => {
    const i = Math.max(0, Math.min(R_STEPS.length - 1, R_STEPS.indexOf(store.s.r) + dir));
    if (R_STEPS[i] === store.s.r) return;
    store.set({ r: R_STEPS[i] });
    clearTimeout(rTimer);
    rTimer = window.setTimeout(() => {
      const last = turns[turns.length - 1];
      if (last && store.s.loaded) void generate(last.text, true);
    }, 250);
  };
  $('r-prev').addEventListener('click', () => stepR(-1));
  $('r-next').addEventListener('click', () => stepR(1));

  // ---------------------------------------------------------------- rendering
  const msg = (el: HTMLElement, m?: Msg, err?: boolean) => {
    const params: Record<string, string | number> = { maxN: store.s.limits.maxN, maxK: store.s.limits.maxK, ...(m?.params ?? {}) };
    for (const [k, key] of Object.entries(m?.sub ?? {})) params[k] = t(key);
    el.textContent = m ? t(m.key, params) : '';
    el.classList.toggle('is-error', !!err);
  };

  function renderExamples() {
    const list = $('ex-list');
    list.replaceChildren(
      ...PRESETS.map((p) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'ex';
        b.setAttribute('aria-pressed', String(p.id === store.s.presetId));
        b.textContent = `“${p.output}”`;
        b.addEventListener('click', () => void pickExample(p.id));
        return b;
      }),
    );
  }

  function renderChips() {
    $('chips').replaceChildren(
      ...QUICK[getLang()].map((q) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip';
        b.textContent = q;
        b.addEventListener('click', () => {
          usr.value = q;
          send();
        });
        return b;
      }),
    );
    usr.placeholder = t('chat.ph');
    if (!log.children.length) {
      const p = document.createElement('p');
      p.className = 'chat-empty';
      p.textContent = t('chat.empty');
      log.append(p);
    } else {
      const e = log.querySelector('.chat-empty');
      if (e) e.textContent = t('chat.empty');
    }
  }

  function renderLeft() {
    const s = store.s;
    const busy = s.solving || s.finding || s.backend.status === 'offline';
    $('tab-examples').setAttribute('aria-selected', String(s.leftMode === 'examples'));
    $('tab-custom').setAttribute('aria-selected', String(s.leftMode !== 'examples'));
    $('mode-examples').hidden = s.leftMode !== 'examples';
    $('mode-custom').hidden = s.leftMode !== 'custom';
    $('mode-manual').hidden = s.leftMode !== 'manual';
    for (const id of ['compute', 'find', 'accept']) $<HTMLButtonElement>(id).disabled = busy;
    $('compute').textContent = t(s.solving ? 'make.busy' : 'make.btn');
    $('find').textContent = t(s.finding || s.solving ? 'make.busy' : 'make.find');
    $('want-hint').textContent = '';

    const dec = $('decide');
    dec.hidden = !s.pending;
    if (s.pending) {
      $('asked-text').textContent = s.pending.targetTokens.map((x) => x.text).join('').trim();
      $('found-text').textContent = s.pending.foundTokens.map((x) => x.text).join('').trim();
    }

    const y = $('make-y');
    y.textContent = s.made && s.solve && !s.pending ? t('make.ready', { y: fit(s.solve.targetTokens.map((x) => x.text).join('').trim(), 420, 13) }) : '';
    const handedOver = !!s.made && s.loaded?.prefixId === s.made.prefixId;
    $('made-card').replaceChildren(...(s.made && !s.pending ? [prefixCard(s.made, { draggable: true, compact: true })] : []));
    $('made-actions').hidden = !s.made || !!s.pending;
    $('use').hidden = handedOver;
    $('made-actions').querySelector<HTMLElement>('[data-i18n="make.drag"]')!.hidden = handedOver;
    $('use').classList.toggle('pulse', !!s.made && !handedOver);
    msg($('left-status'), s.leftMsg, s.leftErr);
  }

  function renderRight() {
    const s = store.s;
    $('drop-empty').hidden = !!s.loaded;
    drop.classList.toggle('has-file', !!s.loaded);
    $('loaded-card').replaceChildren(
      ...(s.loaded ? [prefixCard(s.loaded, { compact: true, onRemove: () => store.set({ loaded: undefined }) })] : []),
    );
    const runBtn = $<HTMLButtonElement>('run');
    runBtn.disabled = s.running || s.backend.status === 'offline';
    $('r-val').textContent = String(s.r);
    $<HTMLButtonElement>('r-prev').disabled = s.r === R_STEPS[0];
    $<HTMLButtonElement>('r-next').disabled = s.r === R_STEPS[R_STEPS.length - 1];
    // only an uploaded prefix that ends with a non-space token needs a warning
    const last = s.loaded?.lastToken;
    $('term').textContent = last && !/^\s+$/.test(last.text) ? t('ctl.termBad', { a: plain(last) }) : '';
    msg($('right-status'), s.rightMsg, s.rightErr);
  }

  store.on((_s, ch) => {
    const any = (...keys: (keyof typeof _s)[]) => keys.some((k) => ch.has(k));
    if (any('presetId')) renderExamples();
    if (any('lang')) renderChips();
    if (any('lang', 'backend', 'leftMode', 'finding', 'pending', 'want', 'solving', 'solve', 'made', 'loaded', 'leftMsg', 'leftErr'))
      renderLeft();
    if (any('lang', 'backend', 'loaded', 'running', 'r', 'rightMsg', 'rightErr')) renderRight();
    if (any('lang')) {
      turns.forEach(renderTurn);
      const sys = log.querySelector('.msg-system');
      if (sys && sysText) sys.textContent = sysText();
    }
  });

  return {
    pickExample,
    useMade: () => store.s.made && load(store.s.made, false),
    say: (text: string) => generate(sent(text)),
  };
}
