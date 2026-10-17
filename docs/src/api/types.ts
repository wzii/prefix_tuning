/** The contract between the page and the GPU backend. */
export interface Token {
  id: number;
  text: string;
}

/** Row-major matrix. On the wire it is {shape, dtype, b64}; api clients decode it to this. */
export interface Matrix {
  rows: number;
  cols: number;
  data: Float32Array;
}

export interface Arch {
  nLayers: number;
  nq: number;
  nkv: number;
  G: number;
  hd: number;
}

export interface Limits {
  maxK: number;
  maxR: number;
  maxN: number;
  maxP: number;
}

export interface Health {
  status: 'ok';
  model: string;
  arch: Arch;
  limits: Limits;
}

export interface LayerDiag {
  cond: number[]; // one per KV head
  resid: number;
  logitPref: number[]; // one per query head
  logitReal: number[];
}

/** Result of looking for a hidden prompt whose continuation is the requested output. */
export type FindResult =
  | { status: 'matched'; searchId: string; targetTokens: Token[]; k: number; lastToken: Token }
  | { status: 'mismatch'; searchId: string; targetTokens: Token[]; foundTokens: Token[]; firstDiff: number; k: number; lastToken: Token }
  | { status: 'rejected'; reason: 'empty' | 'too_long' | 'unsafe' | 'pii' };

export interface SolveResult {
  prefixId: string;
  /** empty when the source is a hidden prompt */
  source: string;
  /** true when the source is one of the server's hidden prompts; its tokens are masked except the last */
  sourceHidden: boolean;
  sourceTokens: Token[];
  p: number;
  kRequested: number;
  k: number;
  eosTruncated: boolean;
  m: number;
  G: number;
  targetTokens: Token[];
  diag: LayerDiag[];
  summary: { condMax: number; condMedian: number; residMax: number; marginMin: number };
  timing: { totalMs: number };
}

export interface LayerDetail {
  layer: number;
  kvHead: number;
  p: number;
  k: number;
  m: number;
  T: number;
  slotLabels: string[]; // m entries, "h·t"
  keyLabels: Token[]; // T entries: x'_1..x'_p, y_1..y_{k-1}
  qy: Matrix; // m x hd
  Kp: Matrix; // m x hd
  S: Matrix; // m x m
  R: Matrix; // m x T
  V: Matrix; // m x T
  Vp: Matrix; // m x hd
  logitPref: number[]; // m
  logitReal: number[]; // m
  cond: number;
  resid: number;
}

export type GenMode = 'baseline' | 'prefixed';

export type GenEvent =
  | { type: 'queued'; position: number }
  | {
      type: 'start';
      genId: string;
      userTokens: Token[];
      n: number;
      p: number;
      terminalOk: boolean;
      autoFixed: boolean;
      warnings: string[];
    }
  | {
      type: 'token';
      mode: GenMode;
      t: number;
      token: Token;
      match: boolean;
      prefixMassMean?: number;
      perLayerMass?: number[];
    };

export interface GenDone {
  genId: string;
  success: boolean;
  n: number;
  k: number;
  m: number;
  r: number;
  userTokens: Token[];
  baselineTokens: Token[];
  prefixedTokens: Token[];
  targetTokens: Token[];
  perLayerMass: number[][]; // [layer][step], mean over query heads
  perLayerHeadMass: number[][]; // [layer][query head], mean over decode steps
  copySpreadMax: number;
  timing: { baselineMs: number; prefixedMs: number };
}

export interface AttentionDetail {
  layer: number;
  mode: GenMode;
  n: number;
  k: number;
  m: number; // 0 for baseline
  r: number;
  queryLabels: Token[]; // n + k - 1 rows
  slotLabels: string[]; // m prefix columns (r identical copies summed)
  heads: Matrix[]; // nq matrices, each (n+k-1) x (m + n+k-1)
  copySpreadMax: number;
}

/** One attention head at one decoding step, in the source run and in the run being shown. */
export interface MatchStep {
  t: number; // 0-based: this query writes y_{t+1}
  srcW: number[] | null; // source run weights over srcKeys[0 .. p+t), i.e. the row R_{(h,t),:}; null if the source is unknown
  prefW: number[]; // weights on the m prefix slots (r copies summed); empty without the prefix
  realW: number[]; // weights on runKeys[0 .. n+t)
  aSrc: number[]; // hd: source attention output a_t (before o_proj)
  aRun: number[]; // hd: this run's attention output
}

export interface MatchDetail {
  layer: number;
  qHead: number;
  kvHead: number;
  h: number; // query head within its group, so the own slot of step t is h*k + t
  mode: GenMode;
  n: number;
  p: number;
  k: number;
  m: number;
  r: number;
  srcKeys: Token[]; // p + k - 1; a hidden prompt comes back with empty texts
  srcHidden: boolean;
  runKeys: Token[]; // n + k - 1: the message, then what this run wrote
  steps: MatchStep[];
}

/** What the inference side knows about a prefix: only what is in the file. */
export interface PrefixInfo {
  prefixId: string;
  fileName: string;
  bytes: number;
  sha256: string;
  model: string;
  nLayers: number;
  nKv: number;
  m: number;
  k: number;
  hd: number;
  /** from the file's metadata; used to check the output, never to produce it */
  expected: Token[];
  lastToken: Token;
  /** an example output's prefix, which the server always keeps */
  pinned?: boolean;
}

export interface GenerateRequest {
  prefixId: string;
  userInput: string;
  r: number;
  autoFix: boolean;
  /** cancels the run (a newer one replaced it) */
  signal?: AbortSignal;
}

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    /** machine-readable detail, e.g. why a file was rejected: format | model | shape | last_token */
    public reason?: string,
  ) {
    super(message);
  }
}

export interface Api {
  kind: 'mock' | 'live';
  health(): Promise<Health>;
  tokenize(text: string): Promise<Token[]>;
  /** Look for a hidden prompt that makes the model write `target`. */
  findSource(target: string, lang: 'en' | 'zh'): Promise<FindResult>;
  /** Solve the prefix from a search: the hidden prompt and the output it found. */
  solveSearch(searchId: string): Promise<SolveResult>;
  /** Solve the prefix from a source the user wrote themselves. */
  solve(source: string, k: number): Promise<SolveResult>;
  layerDetail(prefixId: string, layer: number, kvHead: number): Promise<LayerDetail>;
  generate(req: GenerateRequest, onEvent: (e: GenEvent) => void): Promise<GenDone>;
  attention(genId: string, layer: number, mode: GenMode): Promise<AttentionDetail>;
  /** One query head of one layer, every decoding step: the source run next to this run. */
  match(genId: string, layer: number, qHead: number, mode: GenMode): Promise<MatchDetail>;
  /** The solved prefix as a .safetensors file. */
  downloadPrefix(prefixId: string): Promise<Blob>;
  /** Describe a prefix that already lives on the server (left panel -> right panel). */
  prefixInfo(prefixId: string): Promise<PrefixInfo>;
  /** Send a .safetensors file; the server keeps only its tensors and returns a new id. */
  uploadPrefix(file: Blob, fileName: string): Promise<PrefixInfo>;
}
