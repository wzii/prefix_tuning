import {
  ApiError,
  type Api,
  type AttentionDetail,
  type FindResult,
  type GenDone,
  type GenEvent,
  type GenMode,
  type GenerateRequest,
  type Health,
  type LayerDetail,
  type LayerDiag,
  type MatchDetail,
  type MatchStep,
  type Matrix,
  type PrefixInfo,
  type SolveResult,
  type Token,
} from '../api/types';
import { cond2, matmul, matmulT, relResidual, softmaxRows, solve, zeros, type Mat64 } from '../lib/linalg';
import { Rng, hashStr, seedOf } from '../lib/rng';
import { BANK, BANK_ZH, PRESETS } from './presets';
import matchedFixture from './fixtures/find_matched.json';
import unmatchedFixture from './fixtures/find_unmatched.json';
import { tokenize } from './tokenizer';
import { decodeSafetensors, encodeSafetensors, sha256Hex, type Tensor } from '../lib/safetensors';

/**
 * A stand-in for the FastAPI backend that runs entirely in the browser.
 *
 * What is real: the per-layer algebra. For each (layer, KV head) it draws synthetic
 * queries/keys/values, builds the key prefix with the same closed-form manifold
 * maximiser as core/attention_matching.py, forms S and R, solves S V = R, and reports
 * cond(S) and the residual that this produces.
 * What is fake: the model. Continuations, and the attention patterns during
 * generation, are generated to look like the measured behaviour, not computed.
 */
const ARCH = { nLayers: 36, nq: 32, nkv: 8, G: 4, hd: 128 };
const LIMITS = { maxK: 16, maxR: 64, maxN: 64, maxP: 64 };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const toF32 = (m: Mat64): Matrix => ({ rows: m.rows, cols: m.cols, data: Float32Array.from(m.d) });

interface PrefixRec {
  id: string;
  source: string;
  sourceTokens: Token[];
  targetTokens: Token[];
  k: number;
  p: number;
  lastToken: Token;
  seed: number;
  /** the source is a hidden prompt: never shown, never written into the file */
  hidden: boolean;
  layers: Map<string, LayerDetail>;
  file?: { bytes: Uint8Array; sha: string; dataSha: string };
}

const MODEL = 'Qwen/Qwen3-4B';
const FORMAT = 'attention-matching-prefix/1';

interface GenRec {
  id: string;
  prefix: PrefixRec;
  n: number;
  r: number;
  userTokens: Token[];
  out: Record<GenMode, Token[]>;
  headMass: number[][];
  seed: number;
}

function synthContinuation(seedText: string, count: number): Token[] {
  const rng = new Rng(hashStr(seedText));
  const bank = /\p{Script=Han}/u.test(seedText) ? BANK_ZH : BANK;
  const out: Token[] = [];
  while (out.length < count) out.push(...tokenize(rng.pick(bank)));
  return out.slice(0, count);
}

/** After a trailing space, the first token of a continuation carries no leading space. */
function dropLeadingSpace(toks: Token[]): Token[] {
  if (!toks.length) return toks;
  return [{ ...toks[0], text: toks[0].text.replace(/^[ \t]+/, '') }, ...toks.slice(1)];
}

/** Source value vectors of each computed layer, kept for the matching view (not part of LayerDetail). */
const SRC_VALUES = new WeakMap<LayerDetail, Mat64>();

/** The per-layer solve, as in solve_layer(): real math on synthetic q/k/v. */
function computeLayer(pr: PrefixRec, layer: number, kv: number): LayerDetail {
  const { G, hd } = ARCH;
  const k = pr.k;
  const m = G * k;
  const p = pr.sourceTokens.length;
  const T = p + k - 1;
  const rng = new Rng(seedOf(pr.seed, layer, kv));
  const rt = Math.sqrt(hd);

  // k_norm weight gamma, and a shared direction that correlates the queries of one layer.
  const gamma = Array.from({ length: hd }, () => rng.uniform(0.55, 1.6));
  const common = Array.from({ length: hd }, () => rng.gauss());
  const alpha = 1.0 + 0.9 * Math.pow(rng.next(), 4) + (layer < 2 ? 0.3 : 0);

  // qy: m rows ordered (h, t) -> j = h*k + t, as reshape(n_kv, G*k, hd) does.
  const qy = zeros(m, hd);
  const logitPref: number[] = [];
  for (let j = 0; j < m; j++) {
    const row = Array.from({ length: hd }, (_, c) => alpha * common[c] + rng.gauss());
    let wn = 0;
    for (let c = 0; c < hd; c++) wn += (row[c] * gamma[c]) ** 2;
    const target = rng.uniform(11, 22); // |q ⊙ γ| = the attainable override logit
    const s = target / Math.sqrt(wn);
    for (let c = 0; c < hd; c++) qy.d[j * hd + c] = row[c] * s;
    logitPref.push(target);
  }

  // Kp = manifold_key_prefix(qy, γ): u = √hd · (q⊙γ)/‖q⊙γ‖, k = u ⊙ γ
  const Kp = zeros(m, hd);
  for (let j = 0; j < m; j++) {
    let wn = 0;
    for (let c = 0; c < hd; c++) wn += (qy.d[j * hd + c] * gamma[c]) ** 2;
    wn = Math.sqrt(wn);
    for (let c = 0; c < hd; c++) Kp.d[j * hd + c] = ((rt * qy.d[j * hd + c] * gamma[c]) / wn) * gamma[c];
  }

  // S = softmax(qy Kp^T / √hd): every stacked query row over all m prefix slots.
  const logitsS = matmulT(qy, Kp);
  logitsS.d.forEach((v, i) => (logitsS.d[i] = v / rt));
  const S = softmaxRows(logitsS);

  // Source keys and values: token 0 is an attention sink, the rest are noisy.
  const kw = zeros(T, hd);
  const meanQ = new Float64Array(hd);
  for (let j = 0; j < m; j++) for (let c = 0; c < hd; c++) meanQ[c] += qy.d[j * hd + c] / m;
  const mqn = Math.sqrt(meanQ.reduce((s, v) => s + v * v, 0));
  for (let t = 0; t < T; t++)
    for (let c = 0; c < hd; c++) {
      const sink = t === 0 ? (2 * rt * meanQ[c]) / (mqn * mqn) : 0;
      kw.d[t * hd + c] = sink + rng.gauss() * (0.9 + 0.15 * (t % 3));
    }
  const vw = zeros(T, hd);
  vw.d.forEach((_, i) => (vw.d[i] = rng.gauss()));

  // R = the source run's causal attention rows: row (h, t) sees keys < p + t.
  const logitsR = matmulT(qy, kw);
  logitsR.d.forEach((v, i) => (logitsR.d[i] = v / rt));
  const keep = (i: number, j: number) => j < p + (i % k);
  const R = softmaxRows(logitsR, keep);
  const logitReal: number[] = [];
  for (let i = 0; i < m; i++) {
    let mx = -Infinity;
    for (let j = 0; j < T; j++) if (keep(i, j)) mx = Math.max(mx, logitsR.d[i * T + j]);
    logitReal.push(mx);
  }

  const V = solve(S, R);
  const Vp = matmul(V, vw);

  const detail: LayerDetail = {
    layer,
    kvHead: kv,
    p,
    k,
    m,
    T,
    slotLabels: Array.from({ length: m }, (_, j) => `${Math.floor(j / k)}·${j % k}`),
    keyLabels: [...pr.sourceTokens, ...pr.targetTokens.slice(0, k - 1)],
    qy: toF32(qy),
    Kp: toF32(Kp),
    S: toF32(S),
    R: toF32(R),
    V: toF32(V),
    Vp: toF32(Vp),
    logitPref,
    logitReal,
    cond: cond2(S),
    resid: relResidual(S, V, R),
  };
  SRC_VALUES.set(detail, vw);
  return detail;
}

export class MockApi implements Api {
  kind = 'mock' as const;
  private prefixes = new Map<string, PrefixRec>();
  /** What the simulated model writes for a given set of prefix tensors (keyed by their bytes). */
  private byTensors = new Map<string, PrefixRec>();
  private gens = new Map<string, GenRec>();

  async health(): Promise<Health> {
    return { status: 'ok', model: 'Qwen/Qwen3-4B (mock)', arch: ARCH, limits: LIMITS };
  }

  async tokenize(text: string): Promise<Token[]> {
    return tokenize(text);
  }

  private layer(pr: PrefixRec, layer: number, kv: number): LayerDetail {
    const key = `${layer}:${kv}`;
    let d = pr.layers.get(key);
    if (!d) {
      d = computeLayer(pr, layer, kv);
      pr.layers.set(key, d);
    }
    return d;
  }

  /** Mock of the server's search over hidden prompt templates (the templates live only on the server). */
  private searches = new Map<string, { targetTokens: Token[]; lastToken: Token; p: number }>();

  async findSource(target: string, lang: 'en' | 'zh'): Promise<FindResult> {
    await sleep(500);
    const text = target.trim();
    if (!text) return { status: 'rejected', reason: 'empty' };
    // stand-in for the server's moderation model
    if (/\b(bomb|kill|suicide|nazi|terrorist)\b|炸弹|自杀|杀人/i.test(text)) return { status: 'rejected', reason: 'unsafe' };
    if (/https?:\/\/|www\.|@\S+\.\w|\d{3}[-\s]?\d{3,4}[-\s]?\d{4}/i.test(text)) return { status: 'rejected', reason: 'pii' };
    for (const fx of [matchedFixture, unmatchedFixture]) {
      if (fx.request.target !== text) continue;
      const r = fx.response as Record<string, unknown>;
      const res = {
        status: r.status,
        searchId: r.search_id,
        targetTokens: r.target_tokens,
        k: r.k,
        lastToken: r.last_token,
        ...(r.status === 'mismatch' ? { foundTokens: r.found_tokens, firstDiff: r.first_diff } : {}),
      } as FindResult;
      if (res.status !== 'rejected')
        this.searches.set(res.searchId, {
          targetTokens: res.status === 'mismatch' ? res.foundTokens : res.targetTokens,
          lastToken: res.lastToken,
          p: 11,
        });
      return res;
    }
    // hidden prompts end with a line break
    const targetTokens = tokenize(text);
    if (targetTokens.length > LIMITS.maxK) return { status: 'rejected', reason: 'too_long' };
    const lastToken: Token = tokenize('\n')[0];
    void lang;
    const searchId = 's_' + (hashStr(text) >>> 0).toString(16).padStart(8, '0');
    this.searches.set(searchId, { targetTokens, lastToken, p: 9 + (hashStr(text) % 8) });
    return { status: 'matched', searchId, targetTokens, k: targetTokens.length, lastToken };
  }

  async solveSearch(searchId: string): Promise<SolveResult> {
    const sr = this.searches.get(searchId);
    if (!sr) throw new ApiError('prefix_not_found', 'This search expired.');
    const masked: Token[] = Array.from({ length: sr.p - 1 }, () => ({ id: -1, text: '⋯' }));
    const pr: PrefixRec = {
      id: (hashStr(searchId) >>> 0).toString(16).padStart(8, '0'),
      source: '',
      sourceTokens: [...masked, sr.lastToken],
      targetTokens: sr.targetTokens,
      k: sr.targetTokens.length,
      p: sr.p,
      lastToken: sr.lastToken,
      seed: hashStr(searchId),
      hidden: true,
      layers: new Map(),
    };
    return this.build(pr);
  }

  async solve(source: string, k: number): Promise<SolveResult> {
    const sourceTokens = tokenize(source);
    if (!sourceTokens.length) throw new ApiError('bad_input', 'Type a source sentence first.');
    if (sourceTokens.length > LIMITS.maxP)
      throw new ApiError('too_long', `The source has ${sourceTokens.length} tokens; the limit is ${LIMITS.maxP}.`);
    const pr: PrefixRec = {
      id: (hashStr(`${source}\u0000${k}`) >>> 0).toString(16).padStart(8, '0'),
      source,
      sourceTokens,
      targetTokens: dropLeadingSpace(synthContinuation(source, 16)).slice(0, k),
      k,
      p: sourceTokens.length,
      lastToken: sourceTokens[sourceTokens.length - 1],
      seed: hashStr(source),
      hidden: false,
      layers: new Map(),
    };
    return this.build(pr);
  }

  /** The per-layer solve for all 36 layers × 8 KV heads, as the server does it. */
  private async build(pr: PrefixRec): Promise<SolveResult> {
    const t0 = performance.now();
    const k = pr.k;
    this.prefixes.set(pr.id, pr);
    const diag: LayerDiag[] = [];
    for (let l = 0; l < ARCH.nLayers; l++) {
      const cond: number[] = [];
      const lp: number[] = [];
      const lr: number[] = [];
      let resid = 0;
      for (let kv = 0; kv < ARCH.nkv; kv++) {
        const d = this.layer(pr, l, kv);
        cond.push(d.cond);
        resid = Math.max(resid, d.resid);
        for (let h = 0; h < ARCH.G; h++) {
          const rows = Array.from({ length: k }, (_, t) => h * k + t);
          lp.push(Math.min(...rows.map((j) => d.logitPref[j])));
          lr.push(Math.max(...rows.map((j) => d.logitReal[j])));
        }
      }
      diag.push({ cond, resid, logitPref: lp, logitReal: lr });
      if (l % 6 === 5) await sleep(0);
    }
    const conds = diag.flatMap((d) => d.cond).sort((a, b) => a - b);
    await sleep(Math.max(0, 900 - (performance.now() - t0)));
    return {
      prefixId: pr.id,
      source: pr.source,
      sourceHidden: pr.hidden,
      sourceTokens: pr.sourceTokens,
      p: pr.p,
      kRequested: k,
      k,
      eosTruncated: false,
      m: ARCH.G * k,
      G: ARCH.G,
      targetTokens: pr.targetTokens,
      diag,
      summary: {
        condMax: conds[conds.length - 1],
        condMedian: conds[Math.floor(conds.length / 2)],
        residMax: Math.max(...diag.map((d) => d.resid)),
        marginMin: Math.min(...diag.flatMap((d) => d.logitPref.map((v, i) => v - d.logitReal[i]))),
      },
      timing: { totalMs: performance.now() - t0 },
    };
  }

  /** Pack every layer's (K_p, V_p) into a .safetensors file. */
  private async fileOf(pr: PrefixRec) {
    if (pr.file) return pr.file;
    const tensors: Record<string, Tensor> = {};
    const { nLayers, nkv, hd } = ARCH;
    const m = ARCH.G * pr.k;
    for (let l = 0; l < nLayers; l++) {
      const key = new Float32Array(nkv * m * hd);
      const val = new Float32Array(nkv * m * hd);
      for (let kv = 0; kv < nkv; kv++) {
        const d = this.layer(pr, l, kv);
        key.set(d.Kp.data, kv * m * hd);
        val.set(d.Vp.data, kv * m * hd);
      }
      tensors[`layers.${l}.prefix_key`] = { shape: [nkv, m, hd], data: key };
      tensors[`layers.${l}.prefix_value`] = { shape: [nkv, m, hd], data: val };
    }
    const bytes = encodeSafetensors(tensors, {
      format: FORMAT,
      model: MODEL,
      k: String(pr.k),
      last_token: JSON.stringify(pr.lastToken),
      expected_continuation: JSON.stringify(pr.targetTokens),
      ...(pr.hidden ? {} : { source: pr.source }),
      note: 'Inference reads only the tensors, model, k, p and last_token. The other fields are for people.',
    });
    const headLen = Number(new DataView(bytes.buffer).getBigUint64(0, true));
    const dataSha = await sha256Hex(bytes.slice(8 + headLen).buffer);
    pr.file = { bytes, sha: await sha256Hex(bytes.slice().buffer), dataSha };
    this.byTensors.set(dataSha, pr);
    return pr.file;
  }

  private info(pr: PrefixRec, f: { bytes: Uint8Array; sha: string }, fileName: string): PrefixInfo {
    return {
      prefixId: pr.id,
      fileName,
      bytes: f.bytes.byteLength,
      sha256: f.sha,
      model: MODEL,
      nLayers: ARCH.nLayers,
      nKv: ARCH.nkv,
      m: ARCH.G * pr.k,
      k: pr.k,
      hd: ARCH.hd,
      expected: pr.targetTokens,
      lastToken: pr.lastToken,
    };
  }

  async downloadPrefix(prefixId: string): Promise<Blob> {
    const pr = this.prefixes.get(prefixId);
    if (!pr) throw new ApiError('prefix_not_found', 'This prefix expired.');
    const f = await this.fileOf(pr);
    return new Blob([f.bytes as BlobPart], { type: 'application/octet-stream' });
  }

  async prefixInfo(prefixId: string): Promise<PrefixInfo> {
    const pr = this.prefixes.get(prefixId);
    if (!pr) throw new ApiError('prefix_not_found', 'This prefix expired.');
    const f = await this.fileOf(pr);
    return this.info(pr, f, `prefix-${pr.id}.safetensors`);
  }

  async uploadPrefix(file: Blob, fileName: string): Promise<PrefixInfo> {
    const buf = await file.arrayBuffer();
    let parsed;
    try {
      parsed = decodeSafetensors(buf);
    } catch {
      throw new ApiError('bad_file', 'This is not a .safetensors file.', 'format');
    }
    const { tensors, metadata } = parsed;
    if (metadata.model !== MODEL)
      throw new ApiError('bad_file', `This prefix is for ${metadata.model ?? 'an unknown model'}, not ${MODEL}.`, 'model');
    const k = Number(metadata.k);
    const m = ARCH.G * k;
    for (let l = 0; l < ARCH.nLayers; l++)
      for (const part of ['prefix_key', 'prefix_value']) {
        const t = tensors[`layers.${l}.${part}`];
        if (!t || t.shape.join() !== [ARCH.nkv, m, ARCH.hd].join())
          throw new ApiError('bad_file', `Layer ${l} ${part} is missing or has the wrong shape.`, 'shape');
      }
    let lastToken: Token;
    try {
      lastToken = JSON.parse(metadata.last_token);
    } catch {
      throw new ApiError('bad_file', 'The file does not say which token the source ends with.', 'last_token');
    }
    let expected: Token[] = [];
    try {
      expected = JSON.parse(metadata.expected_continuation ?? '[]');
    } catch {
      /* optional */
    }
    const headLen = Number(new DataView(buf).getBigUint64(0, true));
    const dataSha = await sha256Hex(buf.slice(8 + headLen));
    const sha = await sha256Hex(buf);
    // The simulated model's behaviour follows the tensors, never the metadata.
    const known = this.byTensors.get(dataSha);
    const id = sha.slice(0, 12);
    const pr: PrefixRec = {
      id,
      source: '',
      sourceTokens: [],
      targetTokens: known ? known.targetTokens : synthContinuation(dataSha, k),
      k,
      p: known ? known.p : 0,
      lastToken,
      seed: hashStr(dataSha),
      hidden: true,
      layers: new Map(),
      file: { bytes: new Uint8Array(buf), sha, dataSha },
    };
    this.prefixes.set(id, pr);
    const info = this.info(pr, pr.file!, fileName);
    return { ...info, expected };
  }

  async layerDetail(prefixId: string, layer: number, kvHead: number): Promise<LayerDetail> {
    const pr = this.prefixes.get(prefixId);
    if (!pr) throw new ApiError('prefix_not_found', 'This prefix expired. Run again to rebuild it.');
    return this.layer(pr, layer, kvHead);
  }

  async generate(req: GenerateRequest, on: (e: GenEvent) => void): Promise<GenDone> {
    const pr = this.prefixes.get(req.prefixId);
    if (!pr) throw new ApiError('prefix_not_found', 'This prefix expired. Run again to rebuild it.');
    let userTokens = tokenize(req.userInput);
    if (!userTokens.length) throw new ApiError('bad_input', 'Type your own input first.');
    if (userTokens.length > LIMITS.maxN)
      throw new ApiError('too_long', `Your input has ${userTokens.length} tokens; the limit is ${LIMITS.maxN}.`);
    const srcLast = pr.lastToken;
    let terminalOk = userTokens[userTokens.length - 1].text === srcLast.text;
    let autoFixed = false;
    if (!terminalOk && req.autoFix) {
      userTokens = [...userTokens.slice(0, -1), srcLast];
      terminalOk = autoFixed = true;
    }
    const n = userTokens.length;
    const p = pr.p;
    const k = pr.k;
    const r = req.r;
    const warnings: string[] = [];
    if (n !== p)
      warnings.push(
        `Your input has ${n} tokens and the source has ${p}. The theorem assumes equal lengths, but in our experiments length differences up to 295 tokens still worked.`,
      );
    if (!terminalOk)
      warnings.push(
        `Your input ends with “${userTokens[n - 1].text.trim()}” but the source ends with “${srcLast.text.trim()}”. The theorem needs the same last token, so expect a mismatch.`,
      );
    const genId = Math.random().toString(16).slice(2, 14);
    on({ type: 'queued', position: 0 });
    await sleep(120);
    on({ type: 'start', genId, userTokens, n, p, terminalOk, autoFixed, warnings });

    const seed = hashStr(userTokens.map((t) => t.text).join('') + '|' + pr.id);
    const rng = new Rng(seed);
    const target = pr.targetTokens;

    // Plain model: a different continuation.
    const known = PRESETS.find((q) => q.userInput === userTokens.map((t) => t.text).join('').trimEnd());
    const baseline = known
      ? [...tokenize(known.plainContinuation), ...synthContinuation('pad', k)].slice(0, k)
      : dropLeadingSpace(synthContinuation(userTokens.map((t) => t.text).join(''), k));
    const t0 = performance.now();
    for (let t = 0; t < k; t++) {
      await sleep(28);
      if (req.signal?.aborted) throw new ApiError('aborted', 'Cancelled.');
      on({ type: 'token', mode: 'baseline', t, token: baseline[t], match: baseline[t].text === target[t].text });
    }
    const baselineMs = performance.now() - t0;

    // Prefix mass: leak per layer (early layers leak more), shrinking with r.
    const leak = Array.from({ length: ARCH.nLayers }, (_, l) => (l < 3 ? 0.35 : 0.08) * rng.uniform(0.4, 2));
    const headMass = leak.map((lk) =>
      Array.from({ length: ARCH.nq }, () => 1 / (1 + (lk * rng.uniform(0.3, 2) * 0.4) / r)),
    );
    // the paper reports 99.65% at r = 1 and 100% from r = 10, so only a wrong last token fails here
    const failAt = !terminalOk ? 0 : -1;
    const wrong = dropLeadingSpace(synthContinuation('fail' + seed, k));
    const prefixed: Token[] = [];
    const perLayerMass: number[][] = leak.map(() => []);
    const t1 = performance.now();
    for (let t = 0; t < k; t++) {
      await sleep(32);
      if (req.signal?.aborted) throw new ApiError('aborted', 'Cancelled.');
      const tok = failAt >= 0 && t >= failAt ? wrong[t] : target[t];
      prefixed.push(tok);
      const massNow = headMass.map((hm) => (hm.reduce((a, b) => a + b, 0) / hm.length) * (1 - 0.002 * rng.next()));
      massNow.forEach((v, l) => perLayerMass[l].push(v));
      on({
        type: 'token',
        mode: 'prefixed',
        t,
        token: tok,
        match: tok.text === target[t].text,
        prefixMassMean: massNow.reduce((a, b) => a + b, 0) / massNow.length,
        perLayerMass: massNow,
      });
    }
    const prefixedMs = performance.now() - t1;
    this.gens.set(genId, { id: genId, prefix: pr, n, r, userTokens, out: { baseline, prefixed }, headMass, seed });

    return {
      genId,
      success: prefixed.every((t, i) => t.text === target[i].text),
      n,
      k,
      m: ARCH.G * k,
      r,
      userTokens,
      baselineTokens: baseline,
      prefixedTokens: prefixed,
      targetTokens: target,
      perLayerMass,
      perLayerHeadMass: headMass,
      copySpreadMax: 3.1e-8,
      timing: { baselineMs, prefixedMs },
    };
  }

  async attention(genId: string, layer: number, mode: GenMode): Promise<AttentionDetail> {
    const g = this.gens.get(genId);
    if (!g) throw new ApiError('gen_not_found', 'This run expired. Run again.');
    const { n, r } = g;
    const k = g.prefix.k;
    const m = mode === 'prefixed' ? ARCH.G * k : 0;
    const Q = n + k - 1;
    const rng = new Rng(seedOf(g.seed, layer, mode));
    const heads: Matrix[] = [];
    for (let qh = 0; qh < ARCH.nq; qh++) {
      const h = qh % ARCH.G;
      const cols = m + Q;
      const data = new Float32Array(Q * cols);
      const tau = rng.uniform(1.5, 6);
      const sinkW = rng.uniform(1, 5);
      for (let i = 0; i < Q; i++) {
        // real tokens: attention sink on token 0 plus a recency bump
        const w: number[] = [];
        let s = 0;
        for (let j = 0; j <= i; j++) {
          const v = (j === 0 ? sinkW : 0) + 2.2 * Math.exp(-(i - j) / tau) + 0.35 * rng.next();
          w.push(v);
          s += v;
        }
        let M = 0;
        if (m) {
          if (i >= n - 1) {
            const t = i - (n - 1);
            M = Math.min(0.99999, g.headMass[layer][qh] * (1 - 0.003 * rng.next()));
            const own = h * k + t;
            const ownShare = rng.uniform(0.9, 0.985);
            let rest = 0;
            const others: number[] = [];
            for (let j = 0; j < m; j++) {
              const v = j === own ? 0 : rng.next() ** 3 * (Math.floor(j / k) === h ? 2 : 1);
              others.push(v);
              rest += v;
            }
            for (let j = 0; j < m; j++)
              data[i * cols + j] = j === own ? M * ownShare : (M * (1 - ownShare) * others[j]) / rest;
          } else {
            const a = rng.uniform(0.01, 0.05); // prompt rows: prefix is visible but not steered
            M = (r * a) / (1 + r * a);
            const ws = Array.from({ length: m }, () => rng.next() ** 2);
            const tot = ws.reduce((a, b) => a + b, 0);
            for (let j = 0; j < m; j++) data[i * cols + j] = (M * ws[j]) / tot;
          }
        }
        for (let j = 0; j <= i; j++) data[i * cols + m + j] = ((1 - M) * w[j]) / s;
      }
      heads.push({ rows: Q, cols, data });
    }
    await sleep(60);
    const gen = g.out[mode];
    return {
      layer,
      mode,
      n,
      k,
      m,
      r,
      queryLabels: [...g.userTokens, ...gen.slice(0, k - 1)],
      slotLabels: Array.from({ length: m }, (_, j) => `${Math.floor(j / k)}·${j % k}`),
      heads,
      copySpreadMax: m ? 3.1e-8 : 0,
    };
  }

  async match(genId: string, layer: number, qHead: number, mode: GenMode): Promise<MatchDetail> {
    const g = this.gens.get(genId);
    if (!g) throw new ApiError('gen_not_found', 'This run expired. Run again.');
    const pr = g.prefix;
    const { G, hd } = ARCH;
    const { n, r } = g;
    const k = pr.k;
    const p = pr.p;
    const kv = Math.floor(qHead / G);
    const h = qHead % G;
    const L = this.layer(pr, layer, kv);
    const vw = SRC_VALUES.get(L)!;
    const att = await this.attention(genId, layer, mode);
    const M = att.heads[qHead];
    const m = att.m;
    // values of this run's own tokens: synthetic, but fixed per run, layer and KV head
    const rng = new Rng(seedOf(g.seed, layer, kv + 101));
    const vRun = Array.from({ length: n + k - 1 }, () => Array.from({ length: hd }, () => rng.gauss()));
    const row = (A: Matrix, i: number, from: number, len: number) => Array.from(A.data.subarray(i * A.cols + from, i * A.cols + from + len));
    const steps: MatchStep[] = [];
    for (let t = 0; t < k; t++) {
      const j = h * k + t;
      const srcW = row(L.R, j, 0, p + t);
      const aSrc = new Array(hd).fill(0);
      srcW.forEach((w, c) => {
        for (let d = 0; d < hd; d++) aSrc[d] += w * vw.d[c * hd + d];
      });
      const i = n - 1 + t;
      const realW = row(M, i, m, n + t);
      const aRun = new Array(hd).fill(0);
      realW.forEach((w, c) => {
        for (let d = 0; d < hd; d++) aRun[d] += w * vRun[c][d];
      });
      let prefW: number[] = [];
      if (m) {
        // the prefix part of the row is pi * S_j, and S_j Vp = R_j V_src = a_t
        const pi = row(M, i, 0, m).reduce((a, b) => a + b, 0);
        prefW = row(L.S, j, 0, m).map((v) => v * pi);
        prefW.forEach((w, s2) => {
          for (let d = 0; d < hd; d++) aRun[d] += w * L.Vp.data[s2 * hd + d];
        });
      }
      steps.push({ t, srcW, prefW, realW, aSrc, aRun });
    }
    const srcKeys = [...pr.sourceTokens, ...pr.targetTokens.slice(0, k - 1)].map((x, i) => (pr.hidden && i < p ? { ...x, text: '' } : x));
    return {
      layer,
      qHead,
      kvHead: kv,
      h,
      mode,
      n,
      p,
      k,
      m,
      r,
      srcKeys,
      srcHidden: pr.hidden,
      runKeys: att.queryLabels.concat(g.out[mode].slice(k - 1, k)).slice(0, n + k - 1),
      steps,
    };
  }
}
