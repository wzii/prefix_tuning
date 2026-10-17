import katex from 'katex';
import 'katex/dist/katex.min.css';

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

/**
 * Render text that contains `$…$` (inline) and `$$…$$` (display) LaTeX into HTML.
 * Outside math, `allowHtml` keeps the few tags our copy uses (<a>, <b>, <em>); otherwise it is escaped.
 * Links are written `[text](#anchor)` or `[text](https://…)`.
 */
export function rich(text: string, allowHtml = false): string {
  const out: string[] = [];
  const re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    const before = text.slice(last, m.index);
    out.push(links(allowHtml ? before : esc(before)));
    const display = m[1] !== undefined;
    out.push(katex.renderToString(display ? m[1] : m[2], { displayMode: display, throwOnError: false, output: 'htmlAndMathml' }));
    last = m.index + m[0].length;
  }
  const rest = text.slice(last);
  out.push(links(allowHtml ? rest : esc(rest)));
  return out.join('');
}

const links = (s: string) =>
  s.replace(/\[([^\]\n]+)\]\((#[\w-]+|https?:\/\/[^\s)]+)\)/g, (_, text, href) =>
    href.startsWith('#') ? `<a href="${href}">${text}</a>` : `<a href="${href}" target="_blank" rel="noopener">${text}</a>`,
  );

/** True when the text needs rich(): it has math or a link. */
export const hasMath = (s: string) => /\$[^$]|\]\((#|https?:)/.test(s);
