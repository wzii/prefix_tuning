import { MockApi } from '../mock/mockApi';
import { HttpApi } from './http';
import type { Api } from './types';

export type Backend =
  | { api: Api; status: 'mock' }
  | { api: Api; status: 'live'; base: string }
  | { api: Api; status: 'offline'; base: string };

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
    return fromQuery.replace(/\/$/, '');
  }
  try {
    const saved = localStorage.getItem('am-api');
    if (saved) return saved;
  } catch {
    /* storage blocked */
  }
  const built = import.meta.env.VITE_API_BASE as string | undefined;
  // 'same-origin': the page is served by the backend itself
  if (built === 'same-origin') return location.origin;
  return built ? built.replace(/\/$/, '') : null;
}

export async function connect(): Promise<Backend> {
  const base = resolveBase();
  if (!base) return { api: new MockApi(), status: 'mock' };
  const api = new HttpApi(base);
  try {
    await api.health();
    return { api, status: 'live', base };
  } catch {
    return { api, status: 'offline', base };
  }
}
