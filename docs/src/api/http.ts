import {
  ApiError,
  type Api,
  type AttentionDetail,
  type MatchDetail,
  type FindResult,
  type GenDone,
  type GenEvent,
  type GenMode,
  type GenerateRequest,
  type Health,
  type LayerDetail,
  type Matrix,
  type PrefixInfo,
  type SolveResult,
  type Token,
} from './types';

/**
 * Client for the demo backend (a FastAPI server on a GPU). The server speaks snake_case and
 * ships matrices as {shape, dtype, b64}; this class converts both.
 */
type WireMat = { shape: number[]; dtype: 'f32' | 'f16'; b64: string };

function f16ToF32(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

function decodeMat(w: WireMat): Matrix {
  const bin = atob(w.b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const cols = w.shape[w.shape.length - 1];
  const rows = w.shape.slice(0, -1).reduce((a, b) => a * b, 1);
  if (w.dtype === 'f32') return { rows, cols, data: new Float32Array(bytes.buffer) };
  const u16 = new Uint16Array(bytes.buffer);
  return { rows, cols, data: Float32Array.from(u16, f16ToF32) };
}

/** Recursively camelCase keys and decode any {shape, dtype, b64} objects. */
function fromWire(x: unknown): any {
  if (Array.isArray(x)) return x.map(fromWire);
  if (x && typeof x === 'object') {
    const o = x as Record<string, unknown>;
    if ('b64' in o && 'shape' in o) return decodeMat(o as unknown as WireMat);
    return Object.fromEntries(
      Object.entries(o).map(([key, v]) => [key.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase()), fromWire(v)]),
    );
  }
  return x;
}

export class HttpApi implements Api {
  kind = 'live' as const;
  constructor(private base: string) {}

  private async req<T>(path: string, body?: unknown, timeoutMs = 60_000): Promise<T> {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetch(this.base + path, {
        method: body === undefined ? 'GET' : 'POST',
        headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: ctl.signal,
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new ApiError(json.code ?? `http_${res.status}`, json.message ?? res.statusText, json.reason);
      return fromWire(json) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  health(): Promise<Health> {
    return this.req('/health', undefined, 4000);
  }
  async tokenize(text: string): Promise<Token[]> {
    return (await this.req<{ tokens: Token[] }>('/tokenize', { text })).tokens;
  }
  findSource(target: string, lang: 'en' | 'zh'): Promise<FindResult> {
    return this.req('/find_source', { target, lang });
  }
  solveSearch(searchId: string): Promise<SolveResult> {
    return this.req('/solve', { search_id: searchId });
  }
  solve(source: string, k: number): Promise<SolveResult> {
    return this.req('/solve', { source, k });
  }
  layerDetail(prefixId: string, layer: number, kvHead: number): Promise<LayerDetail> {
    return this.req(`/solve/${prefixId}/layer/${layer}?kv_head=${kvHead}`);
  }
  attention(genId: string, layer: number, mode: GenMode): Promise<AttentionDetail> {
    return this.req(`/attention/${genId}/${layer}?mode=${mode}`);
  }
  match(genId: string, layer: number, qHead: number, mode: GenMode): Promise<MatchDetail> {
    return this.req(`/match/${genId}/${layer}?head=${qHead}&mode=${mode}`);
  }

  async downloadPrefix(prefixId: string): Promise<Blob> {
    const res = await fetch(`${this.base}/prefix/${prefixId}/file`);
    if (!res.ok) throw new ApiError(`http_${res.status}`, res.statusText);
    return res.blob();
  }
  prefixInfo(prefixId: string): Promise<PrefixInfo> {
    return this.req(`/prefix/${prefixId}`);
  }
  /** The body is the raw .safetensors bytes; the server keeps only the tensors. */
  async uploadPrefix(file: Blob, fileName: string): Promise<PrefixInfo> {
    const res = await fetch(`${this.base}/prefix/upload?name=${encodeURIComponent(fileName)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: file,
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new ApiError(json.code ?? `http_${res.status}`, json.message ?? res.statusText, json.reason);
    return fromWire(json) as PrefixInfo;
  }

  /** POST + server-sent events, read with fetch because EventSource cannot POST. */
  async generate(r: GenerateRequest, on: (e: GenEvent) => void): Promise<GenDone> {
    // a run is cancelled by the caller (a newer run) or after two minutes
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 120_000);
    r.signal?.addEventListener('abort', () => ctl.abort());
    try {
      return await this.stream(r, on, ctl.signal);
    } catch (e) {
      if (ctl.signal.aborted && !r.signal?.aborted) throw new ApiError('timeout', 'The run took too long. Please try again.');
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  private async stream(r: GenerateRequest, on: (e: GenEvent) => void, signal: AbortSignal): Promise<GenDone> {
    const res = await fetch(this.base + '/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prefix_id: r.prefixId, user_input: r.userInput, r: r.r, auto_fix: r.autoFix }),
      signal,
    });
    if (!res.ok || !res.body) {
      const j = await res.json().catch(() => ({}));
      throw new ApiError(j.code ?? `http_${res.status}`, j.message ?? res.statusText);
    }
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let cut: number;
      while ((cut = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        const ev = /^event: (.*)$/m.exec(frame)?.[1];
        const data = fromWire(JSON.parse(/^data: (.*)$/m.exec(frame)?.[1] ?? '{}'));
        if (ev === 'done') return data as GenDone;
        if (ev === 'error') throw new ApiError(data.code, data.message);
        if (ev) on({ type: ev, ...data } as GenEvent);
      }
    }
    throw new ApiError('stream_closed', 'The connection closed before the run finished.');
  }
}
