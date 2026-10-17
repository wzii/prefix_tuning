/** Read palette tokens from CSS so canvas drawings follow light/dark themes. */
export type RGB = [number, number, number];

const cache = new Map<string, RGB>();
const listeners = new Set<() => void>();

export function color(name: string): RGB {
  let c = cache.get(name);
  if (!c) {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
    const hex = raw.startsWith('#') ? raw.slice(1) : '888888';
    const full = hex.length === 3 ? [...hex].map((x) => x + x).join('') : hex;
    c = [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as RGB;
    cache.set(name, c);
  }
  return c;
}

export const css = (c: RGB, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

export const mix = (a: RGB, b: RGB, t: number): RGB => [
  Math.round(a[0] + (b[0] - a[0]) * t),
  Math.round(a[1] + (b[1] - a[1]) * t),
  Math.round(a[2] + (b[2] - a[2]) * t),
];

export function onThemeChange(fn: () => void) {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

function fire() {
  cache.clear();
  listeners.forEach((f) => f());
}

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', fire);
new MutationObserver(fire).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
