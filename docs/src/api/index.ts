import { MockApi } from '../mock/mockApi';
import { HttpApi, fromWire } from './http';
import type { Api, Health } from './types';

/**
 * - live: the GPU server answered.
 * - mock: no server is configured (or ?api=mock); simulated data.
 * - fallback: a server is configured but could not be reached; simulated data, and the page
 *   keeps checking the server in the background (see `watch`).
 */
export type Backend =
  | { api: Api; status: 'mock' }
  | { api: Api; status: 'live'; base: string; health: Health }
  | { api: Api; status: 'fallback'; base: string };

/**
 * Where the backend lives, in order: ?api=… on the page URL, a remembered value,
 * then VITE_API_BASE at build time ('same-origin' when the backend serves the page).
 * With none of these the page runs on the mock.
 */
function resolveBase(): string | null {
  let fromQuery: string | null = null;
  try {
    fromQuery = new URLSearchParams(location.search).get('api');
  } catch {
    /* no query string available */
  }
  if (fromQuery === 'mock') return null;
  if (fromQuery) {
    try {
      localStorage.setItem('am-api', fromQuery);
    } catch {
      /* storage blocked */
    }
    return fromQuery.trim().replace(/\/$/, '');
  }
  try {
    const saved = localStorage.getItem('am-api');
    if (saved) return saved.trim();
  } catch {
    /* storage blocked */
  }
  const built = (import.meta.env.VITE_API_BASE as string | undefined)?.trim();
  // 'same-origin': the page is served by the backend itself
  if (built === 'same-origin') return location.origin;
  return built ? built.replace(/\/$/, '') : null;
}

/**
 * One check of the server. Only a real /health answer counts: a company proxy or captive portal
 * may answer in the server's place (a login page, a 407, a block page), and DNS blocks or a dead
 * server show up as network errors or a hang, which the timeout cuts short.
 */
async function probe(base: string, timeoutMs = 4000): Promise<Health | null> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/health`, { cache: 'no-store', credentials: 'omit', signal: ctl.signal });
    if (!res.ok) return null;
    const h = fromWire(await res.json()) as Health;
    return h && h.status === 'ok' && h.arch && h.limits ? h : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function connect(): Promise<Backend> {
  const base = resolveBase();
  if (!base) return { api: new MockApi(), status: 'mock' };
  const health = await probe(base);
  if (health) return { api: new HttpApi(base), status: 'live', base, health };
  return { api: new MockApi(), status: 'fallback', base };
}

/**
 * While on the fallback, keep checking the server: a few quick retries first (a failure can be
 * passing), then every minute, and at once whenever the visitor comes back to the tab or the
 * network comes back (e.g. after allowing the server through a proxy in another tab).
 * Calls `onLive` once, when the server answers. Returns a function that stops the watch.
 */
export function watch(base: string, onLive: (b: Extract<Backend, { status: 'live' }>) => void): () => void {
  const delays = [2000, 5000, 15000, 30000];
  let tries = 0;
  let timer = 0;
  let busy = false;
  let stopped = false;
  const stop = () => {
    stopped = true;
    clearTimeout(timer);
    removeEventListener('focus', now);
    removeEventListener('online', now);
    document.removeEventListener('visibilitychange', onVis);
  };
  const schedule = () => {
    clearTimeout(timer);
    timer = window.setTimeout(check, delays[tries++] ?? 60000);
  };
  async function check() {
    if (stopped || busy) return;
    busy = true;
    const health = await probe(base, 8000);
    busy = false;
    if (stopped) return;
    if (health) {
      stop();
      onLive({ api: new HttpApi(base), status: 'live', base, health });
    } else schedule();
  }
  function now() {
    clearTimeout(timer);
    void check();
  }
  function onVis() {
    if (document.visibilityState === 'visible') now();
  }
  addEventListener('focus', now);
  addEventListener('online', now);
  document.addEventListener('visibilitychange', onVis);
  schedule();
  return stop;
}
