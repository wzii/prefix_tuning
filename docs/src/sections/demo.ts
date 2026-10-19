import { ApiError, type PrefixInfo } from '../api/types';
import { getLang, t, type Key } from '../i18n';
import { PRESETS } from '../mock/presets';
import type { Msg, Picked, Store } from '../store';
import { fit } from './intro';
import { plain } from '../viz/tokens';
import { ROBOT_BROKEN, HAMMER, ROBOT_OK } from '../viz/robots';
import { R_STEPS } from './viz';
import { isZip, unzipFirst, zipOne } from '../lib/zip';
import { play, renderIdle, type AnimData } from './solveAnim';
import { Inspector } from './inspect';

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

const fmtBytes = (n: number) => (n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.ceil(n / 1e3)} kB`);

/** The prefix drawn as a file: name, shape, size, hash, and the y it records. */
function prefixCard(info: PrefixInfo, note?: string) {
  const el = document.createElement('div');
  el.className = 'pfile';
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
  body.append(name, meta, sha);
  if (info.expected.length) {
    const y = document.createElement('div');
    y.className = 'pfile-y';
    y.textContent = `→ “${fit(info.expected.map((x) => x.text).join('').trim(), 300, 12)}”`;
    body.append(y);
  }
  if (note) {
    const n = document.createElement('div');
    n.className = 'pfile-note';
    n.textContent = note;
    body.append(n);
  }
  el.append(icon, body);
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

/** One message sent to both models, and their two replies. */
interface Turn {
  text: string; // what was sent, ending in the added line break
  plain: string;
  hacked: string;
  done: boolean;
  hit?: boolean;
  plainBubble: HTMLElement;
  hackedBubble: HTMLElement;
  meta: HTMLElement;
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
  const fileIn = $<HTMLInputElement>('file');
  const animBox = $('anim');
  const inspBox = $('inspect');
  const inspBtn = $<HTMLButtonElement>('inspect-btn');
  const logPlain = $('log-plain');
  const logHacked = $('log-hacked');
  const inspector = new Inspector(inspBox);
  let turn: Turn | null = null;
  let runCtl: AbortController | null = null;
  /** whether the prefix in use came from the user's own file */
  let fromUpload = false;
  /** the last finished animation, replayed at once when the language changes */
  let lastAnim: AnimData | null = null;
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

  // ---------------------------------------------------------------- step 1: choose the target
  let solveSeq = 0;
  /** A new target: the prefix and the chats belonged to the old one. */
  function pick(p: Picked | undefined) {
    solveSeq++;
    lastAnim = null;
    closeInspector();
    clearChats();
    fromUpload = false;
    store.set({ picked: p, solving: false, solve: undefined, made: undefined, loaded: undefined, gen: undefined, solveMsg: undefined, solveErr: false });
  }

  /** Ask the server for a hidden prompt after which the model writes `text`. */
  async function find(text: string) {
    const s = store.s;
    if (s.finding || s.solving || s.backend.status === 'offline') return;
    const bad = precheck(text);
    if (bad) return store.set({ leftMsg: { key: bad }, leftErr: true });
    store.set({ finding: true, pending: undefined, leftMsg: { key: 'make.finding' }, leftErr: false });
    try {
      const res = await s.backend.api.findSource(text, getLang());
      store.set({ finding: false, leftMsg: undefined });
      if (res.status === 'rejected') return store.set({ leftMsg: { key: `rej.${res.reason}` as Key }, leftErr: true });
      if (res.status === 'mismatch') return store.set({ pending: res });
      pick({ kind: 'search', searchId: res.searchId, tokens: res.targetTokens });
    } catch (err) {
      store.set({ finding: false, leftMsg: errorMsg(err), leftErr: true });
    }
  }

  function pickExample(id: string) {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    if (p.id === store.s.presetId && store.s.picked?.kind === 'search') return;
    store.set({ presetId: p.id });
    return find(p.output);
  }

  /** In the opening-line mode the target is whatever the model writes after it. */
  function pickManual() {
    const s = store.s;
    pick(s.source.trim() ? { kind: 'manual', source: s.source, k: s.k } : undefined);
  }

  // ---------------------------------------------------------------- step 2: compute the prefix
  async function compute() {
    const s = store.s;
    const p = s.picked;
    if (s.solving || s.backend.status === 'offline') return;
    if (!p) return store.set({ solveMsg: { key: 'solve.need' }, solveErr: true });
    const my = ++solveSeq;
    closeInspector();
    clearChats();
    fromUpload = false;
    store.set({ solving: true, solve: undefined, made: undefined, loaded: undefined, gen: undefined, solveMsg: undefined, solveErr: false });
    try {
      const api = s.backend.api;
      const solve = await (p.kind === 'search' ? api.solveSearch(p.searchId) : api.solve(withLast(p.source), p.k));
      const kv = Math.floor(s.view.head / s.arch.G);
      const [made, detail] = await Promise.all([api.prefixInfo(solve.prefixId), api.layerDetail(solve.prefixId, s.view.layer, kv).catch(() => null)]);
      if (my !== solveSeq) return;
      lastAnim = { solve, detail, nLayers: s.arch.nLayers, nKv: s.arch.nkv };
      await play(animBox, lastAnim, () => my === solveSeq);
      if (my !== solveSeq) return;
      store.set({ solving: false, solve, made });
      load(made, false);
      // an example comes with a suggested message
      const preset = PRESETS.find((x) => x.id === store.s.presetId);
      if (p.kind === 'search' && preset && !usr.value) usr.value = preset.userInput;
    } catch (err) {
      if (my !== solveSeq) return;
      store.set({ solving: false, solveMsg: errorMsg(err), solveErr: true });
      renderAnimIdle();
    }
  }

  function renderAnimIdle() {
    const p = store.s.picked;
    renderIdle(animBox, p?.kind === 'search' ? p.tokens : null);
  }

  // ---------------------------------------------------------------- the page's copies of prefix files
  // The server keeps user-made prefixes only in memory. The page keeps its own copy of each file
  // it uses and uploads it again if the server has dropped it; the server's signature in the
  // file lets it accept the copy without checking it again.
  const copies = new Map<string, Promise<Blob>>();
  function keepCopy(info: PrefixInfo, blob?: Blob) {
    if (blob) copies.set(info.prefixId, Promise.resolve(blob));
    if (copies.has(info.prefixId) || info.pinned) return;
    copyOf(info);
  }
  /** The file's bytes: the page's copy, else fetched once and kept. */
  function copyOf(info: PrefixInfo) {
    let p = copies.get(info.prefixId);
    if (!p) {
      p = store.s.backend.api.downloadPrefix(info.prefixId);
      copies.set(info.prefixId, p);
      p.catch(() => copies.delete(info.prefixId));
    }
    return p;
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

  /** Put a prefix to use in step 3. */
  function load(info: PrefixInfo, upload: boolean, blob?: Blob) {
    fromUpload = upload;
    closeInspector();
    clearChats();
    store.set({ loaded: info, rightMsg: undefined, rightErr: false });
    keepCopy(info, blob);
    const slot = $('fslot');
    slot.classList.remove('flash');
    void slot.offsetWidth;
    slot.classList.add('flash');
  }

  async function upload(file: File) {
    store.set({ solveMsg: { key: 'st.uploading' }, solveErr: false });
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
      store.set({ solveMsg: undefined });
      load(info, true, blob);
    } catch (err) {
      store.set({ solveMsg: errorMsg(err), solveErr: true });
    }
  }

  async function download() {
    const f = store.s.loaded;
    if (!f) return;
    try {
      const blob = await copyOf(f);
      // Inside the claude.ai viewer, files go through its save dialog, which takes .zip but not
      // .safetensors; the zip holds the unchanged .safetensors file. Elsewhere: a plain download.
      const dl = await Promise.race([viewerDownloads(), new Promise<null>((r) => setTimeout(() => r(null), 1500))]);
      if (dl) {
        const bytes = new Uint8Array(await blob.arrayBuffer());
        try {
          await dl.save({ filename: f.fileName.replace(/\.safetensors$/, '.zip'), data: zipOne(f.fileName, bytes) });
        } catch (e) {
          if ((e as { code?: string })?.code !== 'declined') throw new Error((e as { message?: string })?.message ?? String(e));
        }
        return;
      }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = f.fileName;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    } catch (err) {
      store.set({ solveMsg: errorMsg(err), solveErr: true });
    }
  }

  function closeInspector() {
    inspBox.hidden = true;
    inspBtn.setAttribute('aria-expanded', 'false');
    inspBtn.textContent = t('insp.open');
    inspector.reset();
  }
  function toggleInspector() {
    const f = store.s.loaded;
    if (!f) return;
    if (!inspBox.hidden) return closeInspector();
    inspBox.hidden = false;
    inspBtn.setAttribute('aria-expanded', 'true');
    inspBtn.textContent = t('insp.close');
    void inspector.open(f, () => copyOf(f), store.s.view.layer);
  }

  // ---------------------------------------------------------------- step 3: one message, two models
  const row = (who: 'me' | 'ok' | 'broken', text = '') => {
    const r = document.createElement('div');
    r.className = who === 'me' ? 'msg me' : `msg bot ${who}`;
    if (who !== 'me') {
      const av = document.createElement('div');
      av.className = 'avatar';
      av.innerHTML = who === 'ok' ? ROBOT_OK : ROBOT_BROKEN;
      r.append(av);
    }
    const body = document.createElement('div');
    body.className = 'msg-body';
    const b = document.createElement('div');
    b.className = 'msg-bubble';
    if (text) b.textContent = text;
    else b.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    body.append(b);
    if (who !== 'me') body.append(meta);
    r.append(body);
    return { r, b, meta };
  };

  function clearChats() {
    runCtl?.abort();
    turn = null;
    if (store.s.running) store.set({ running: false });
    renderEmpty();
  }

  function renderEmpty() {
    if (turn) return;
    const key: Key = store.s.loaded ? 'chat.empty' : 'chat.needPrefix';
    for (const log of [logPlain, logHacked]) {
      const p = document.createElement('p');
      p.className = 'chat-empty';
      p.textContent = t(key);
      log.replaceChildren(p);
    }
  }

  function renderTurn(tu: Turn) {
    const fill = (b: HTMLElement, text: string) => {
      if (text) b.textContent = text.replace(/^\s+/, '');
    };
    // replies stop after k tokens; mark a reply that was cut mid-sentence
    const cut = (s: string) => (tu.done && s && !/[.!?。！？…"”]\s*$/.test(s) ? s.replace(/\s+$/, '') + '…' : s);
    fill(tu.plainBubble, cut(tu.plain));
    fill(tu.hackedBubble, cut(tu.hacked));
    tu.meta.className = 'msg-meta' + (tu.hit === undefined ? '' : tu.hit ? ' hit' : ' miss');
    tu.meta.textContent = tu.hit === undefined ? '' : `${tu.hit ? '✓' : '✗'} ${t(tu.hit ? 'chat.hit' : 'chat.miss')}`;
    logPlain.scrollTop = logPlain.scrollHeight;
    logHacked.scrollTop = logHacked.scrollHeight;
  }

  /** Joined text of a token list; token splits may differ between the file and the run. */
  const joined = (toks: { text: string }[]) => toks.map((x) => x.text).join('').trim();

  /**
   * Run both models on `text`. With `silent` (the r stepper in the Method section), the last
   * exchange is re-run in place, so the chat always shows what the current r produces. A newer
   * run cancels an older one.
   */
  async function generate(text: string, silent = false) {
    const s = store.s;
    if (!s.loaded) return store.set({ rightMsg: { key: 'run.need' }, rightErr: true });
    let tu: Turn | null = turn;
    if (silent) {
      if (!tu) return;
      Object.assign(tu, { plain: '', hacked: '', done: false, hit: undefined });
    } else {
      // no history: each model sees only this message, so the previous exchange is cleared
      const shown = text.replace(/\s+$/, '');
      const a = row('ok');
      const b = row('broken');
      logPlain.replaceChildren(row('me', shown).r, a.r);
      logHacked.replaceChildren(row('me', shown).r, b.r);
      tu = turn = { text, plain: '', hacked: '', done: false, plainBubble: a.b, hackedBubble: b.b, meta: b.meta };
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
      const exp = loaded.expected.length ? loaded.expected : gen.targetTokens;
      tu.hit = joined(gen.prefixedTokens) === joined(exp);
      // the final text comes from the whole run, so a character split across tokens is intact
      tu.plain = gen.baselineTokens.map((x) => x.text).join('');
      tu.hacked = gen.prefixedTokens.map((x) => x.text).join('');
      tu.done = true;
      renderTurn(tu);
      store.set({ running: false, ...(silent ? {} : { view: { ...store.s.view, mode: 'prefixed', row: null } }) });
      store.set({ gen, userInput: text });
    } catch (err) {
      if (ctl.signal.aborted) return;
      store.set({ running: false, rightMsg: errorMsg(err), rightErr: true });
      if (!silent) {
        turn = null;
        renderEmpty();
      }
    }
  }

  function send() {
    const v = usr.value;
    if (!v.trim() || store.s.running) return;
    if (!store.s.loaded) return store.set({ rightMsg: { key: 'run.need' }, rightErr: true });
    void generate(sent(v));
  }

  // ---------------------------------------------------------------- wiring
  $('tab-examples').addEventListener('click', () => store.set({ leftMode: 'examples', pending: undefined, leftMsg: undefined }));
  $('tab-custom').addEventListener('click', () => store.set({ leftMode: 'custom', leftMsg: undefined }));
  want.addEventListener('input', () => store.set({ want: want.value, pending: undefined }));
  want.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void find(want.value);
    }
  });
  $('find').addEventListener('click', () => void find(want.value));
  $('accept').addEventListener('click', () => {
    const p = store.s.pending;
    if (!p) return;
    store.set({ pending: undefined });
    pick({ kind: 'search', searchId: p.searchId, tokens: p.foundTokens });
  });
  const toManual = () => {
    store.set({ leftMode: 'manual', pending: undefined, leftMsg: undefined });
    if (store.s.source.trim()) pickManual();
    src.focus();
  };
  $('manual').addEventListener('click', toManual);
  $('to-manual').addEventListener('click', toManual);
  $('back-custom').addEventListener('click', () => store.set({ leftMode: 'custom', leftMsg: undefined }));
  src.addEventListener('input', () => {
    store.set({ source: src.value });
    pickManual();
  });
  src.addEventListener('keydown', (e) => e.key === 'Enter' && compute());
  kSel.addEventListener('change', () => {
    store.set({ k: +kSel.value });
    pickManual();
  });

  $('compute').addEventListener('click', () => void compute());
  inspBtn.addEventListener('click', toggleInspector);
  $('dl').addEventListener('click', () => void download());
  fileIn.addEventListener('change', () => {
    const f = fileIn.files?.[0];
    if (f) void upload(f);
    fileIn.value = '';
  });

  $('chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    send();
  });

  // r lives in the Method section: re-run the last message without adding to the chat
  let rTimer = 0;
  const stepR = (dir: number) => {
    const i = Math.max(0, Math.min(R_STEPS.length - 1, R_STEPS.indexOf(store.s.r) + dir));
    if (R_STEPS[i] === store.s.r) return;
    store.set({ r: R_STEPS[i] });
    clearTimeout(rTimer);
    rTimer = window.setTimeout(() => {
      if (turn && store.s.loaded) void generate(turn.text, true);
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
  }

  function renderStep1() {
    const s = store.s;
    const busy = s.solving || s.finding || s.backend.status === 'offline';
    $('tab-examples').setAttribute('aria-selected', String(s.leftMode === 'examples'));
    $('tab-custom').setAttribute('aria-selected', String(s.leftMode !== 'examples'));
    $('mode-examples').hidden = s.leftMode !== 'examples';
    $('mode-custom').hidden = s.leftMode !== 'custom';
    $('mode-manual').hidden = s.leftMode !== 'manual';
    for (const id of ['find', 'accept']) $<HTMLButtonElement>(id).disabled = busy;
    $('find').textContent = t(s.finding ? 'make.finding' : 'make.find');

    const dec = $('decide');
    dec.hidden = !s.pending;
    if (s.pending) {
      $('asked-text').textContent = s.pending.targetTokens.map((x) => x.text).join('').trim();
      $('found-text').textContent = s.pending.foundTokens.map((x) => x.text).join('').trim();
    }

    const y = $('make-y');
    const p = s.pending ? undefined : s.picked;
    y.textContent = !p
      ? ''
      : p.kind === 'search'
        ? t('make.ready', { y: fit(p.tokens.map((x) => x.text).join('').trim(), 640, 14) })
        : t('make.readyManual', { k: p.k });
    msg($('left-status'), s.leftMsg, s.leftErr);
  }

  function renderStep2() {
    const s = store.s;
    const btn = $<HTMLButtonElement>('compute');
    btn.textContent = t(s.solving ? 'solve.busy' : s.made ? 'solve.done' : 'solve.btn');
    // while step 1 is checking a new sentence, the old target is about to be replaced
    btn.disabled = s.solving || s.finding || !!s.made || !s.picked || s.backend.status === 'offline';
    btn.classList.toggle('pulse', !btn.disabled);

    const f = s.loaded;
    const slot = $('fslot');
    if (f) slot.replaceChildren(prefixCard(f, fromUpload ? t('run.loadedFrom') : undefined));
    else {
      const p = document.createElement('p');
      p.className = 'fslot-empty' + (s.solving ? ' is-busy' : '');
      p.textContent = t(s.solving ? 'file.solving' : 'file.empty');
      slot.replaceChildren(p);
    }
    $('file-actions').hidden = !f;
    inspBtn.textContent = t(inspBox.hidden ? 'insp.open' : 'insp.close');
    msg($('solve-status'), s.solveMsg, s.solveErr);
  }

  function renderStep3() {
    const s = store.s;
    const head = (who: 'ok' | 'broken') => {
      const h = document.createElement('div');
      h.className = 'win-who';
      const av = document.createElement('span');
      av.className = 'avatar';
      av.innerHTML = who === 'ok' ? ROBOT_OK : ROBOT_BROKEN;
      const name = document.createElement('span');
      name.className = 'win-name';
      name.textContent = t('chat.ok');
      const sub = document.createElement('span');
      sub.className = 'win-sub';
      sub.textContent = t(who === 'ok' ? 'win.plain' : 'win.hacked');
      const txt = document.createElement('span');
      txt.className = 'win-txt';
      txt.append(name, sub);
      h.append(av, txt);
      if (who === 'broken' && s.loaded) {
        const file = document.createElement('span');
        file.className = 'win-file';
        file.innerHTML = HAMMER;
        file.append(s.loaded.fileName);
        file.title = s.loaded.fileName;
        h.append(file);
      }
      return h;
    };
    $('win-plain-h').replaceChildren(head('ok'));
    $('win-hacked-h').replaceChildren(head('broken'));
    const runBtn = $<HTMLButtonElement>('run');
    runBtn.disabled = s.running || !s.loaded || s.backend.status === 'offline';
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
    if (any('lang', 'backend', 'leftMode', 'finding', 'pending', 'want', 'picked', 'solving', 'leftMsg', 'leftErr')) renderStep1();
    if (any('lang', 'backend', 'picked', 'finding', 'solving', 'made', 'loaded', 'solveMsg', 'solveErr')) renderStep2();
    if (any('lang', 'backend', 'loaded', 'running', 'r', 'rightMsg', 'rightErr')) renderStep3();
    if (any('picked')) renderAnimIdle();
    if (any('loaded')) renderEmpty();
    if (any('lang')) {
      if (turn) renderTurn(turn);
      else renderEmpty();
      if (lastAnim && !store.s.solving) void play(animBox, lastAnim, () => true, true);
      else if (!store.s.solving) renderAnimIdle();
      inspector.render();
    }
  });

  return {
    pickExample,
    say: (text: string) => generate(sent(text)),
  };
}
