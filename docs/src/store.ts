import type { Backend } from './api';
import type { FindResult, GenDone, GenMode, PrefixInfo, SolveResult, Token, Arch, Limits } from './api/types';
import type { Key, Lang } from './i18n';

/** A message kept as key + params so it re-renders when the language changes. */
export interface Msg {
  key: Key;
  params?: Record<string, string | number>;
  /** params that are themselves messages, translated when shown */
  sub?: Record<string, Key>;
}

/** The target chosen in step 1: a search the server matched, or an opening line the user wrote. */
export type Picked = { kind: 'search'; searchId: string; tokens: Token[] } | { kind: 'manual'; source: string; k: number };

export interface State {
  lang: Lang;
  backend: Backend;
  // step 1: choose an output (example or your own), or write a source yourself
  leftMode: 'examples' | 'custom' | 'manual';
  picked?: Picked;
  presetId: string | null;
  want: string;
  wantTokens: number;
  finding: boolean;
  /** a search that did not reproduce the request, waiting for the user's decision */
  pending?: Extract<FindResult, { status: 'mismatch' }>;
  source: string;
  k: number;
  sourceTokens: Token[];
  leftMsg?: Msg;
  leftErr?: boolean;
  // step 2: compute the prefix
  solving: boolean;
  solve?: SolveResult;
  made?: PrefixInfo;
  solveMsg?: Msg;
  solveErr?: boolean;
  // step 3: run the model with the prefix in use (the one just made, or an upload)
  loaded?: PrefixInfo;
  userInput: string;
  userTokens: Token[];
  r: number;
  running: boolean;
  gen?: GenDone;
  rightMsg?: Msg;
  rightErr?: boolean;
  /** model shape and server limits, from /health */
  arch: Arch;
  limits: Limits;
  /** what the canvas shows; row = null means the first output row */
  view: { layer: number; head: number; mode: GenMode; row: number | null };
}

type Listener = (s: State, changed: Set<keyof State>) => void;

export class Store {
  private listeners: Listener[] = [];
  constructor(public s: State) {}
  set(patch: Partial<State>) {
    Object.assign(this.s, patch);
    const changed = new Set(Object.keys(patch) as (keyof State)[]);
    this.listeners.forEach((l) => l(this.s, changed));
  }
  on(l: Listener) {
    this.listeners.push(l);
  }
}
