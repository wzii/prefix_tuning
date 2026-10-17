import type { Token } from '../api/types';

/** Show a token with its leading space and newlines made visible. */
export function tokenEl(tok: Token, cls = ''): HTMLSpanElement {
  const el = document.createElement('span');
  el.className = `tok ${cls}`.trim();
  const m = /^([ \t]*)([\s\S]*)$/.exec(tok.text)!;
  if (tok.text === '\n') {
    el.innerHTML = '<span class="ws">↵</span>';
  } else if (/^[ \t]+$/.test(tok.text)) {
    el.innerHTML = '<span class="ws">␣</span>';
  } else {
    if (m[1]) {
      const ws = document.createElement('span');
      ws.className = 'ws';
      ws.textContent = '·'.repeat(Math.min(m[1].length, 4));
      el.append(ws);
    }
    el.append(document.createTextNode(m[2]));
  }
  el.title = `token id ${tok.id}: ${JSON.stringify(tok.text)}`;
  return el;
}

export function tokenRow(parent: HTMLElement, toks: Token[], cls?: (t: Token, i: number) => string) {
  parent.replaceChildren(...toks.map((t, i) => tokenEl(t, cls ? cls(t, i) : '')));
}

export const plain = (t: Token) => (t.text === '\n' ? '↵' : t.text.trim() || '␣');
