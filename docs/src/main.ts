import './style.css';
import { connect } from './api';
import { applyStatic, getLang, setLang, t } from './i18n';
import { MockApi } from './mock/mockApi';
import { PRESETS } from './mock/presets';
import { mountDemo } from './sections/demo';
import { drawIntroFigure, drawMechFigure } from './sections/intro';
import { mountSolveFigure } from './sections/solveFig';
import { mountCanvas } from './sections/viz';
import { rich } from './lib/math';
import { Store, type State } from './store';

const $ = (id: string) => document.getElementById(id)!;


const BIBTEX = `@inproceedings{lam2026prefix,
  title     = {Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers},
  author    = {Lam, Peter Cho-Ho and Wang, Ziyi and Zhou, Zirui},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}`;

/** Redraw a figure when its box crosses the wide/narrow breakpoint. */
function responsiveFigure(box: HTMLElement, draw: (b: HTMLElement) => void) {
  draw(box);
  let wasWide = box.clientWidth >= 560;
  new ResizeObserver(() => {
    const wide = box.clientWidth >= 560;
    if (wide !== wasWide) draw(box);
    wasWide = wide;
  }).observe(box);
}

async function boot() {
  applyStatic();
  $('bibtex').textContent = BIBTEX;
  const copyBtn = $('bibtex-copy');
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      copyBtn.textContent = t('cite.copied');
    } catch {
      // clipboard blocked (e.g. inside a sandboxed frame): select the text for a manual copy
      getSelection()?.selectAllChildren($('bibtex'));
    }
    setTimeout(() => (copyBtn.textContent = t('cite.copy')), 1600);
  });
  const introBox = $('intro-fig');
  const mechBox = $('mech-fig');
  responsiveFigure(introBox, drawIntroFigure);
  responsiveFigure(mechBox, drawMechFigure);

  const backend = await connect();
  const first = PRESETS.find((p) => p.lang === getLang())!;
  const initial: State = {
    lang: getLang(),
    backend,
    leftMode: 'examples',
    presetId: first.id,
    want: '',
    wantTokens: 0,
    finding: false,
    source: '',
    k: 12,
    sourceTokens: [],
    solving: false,
    userInput: first.userInput,
    userTokens: [],
    r: 32,
    running: false,
    view: { layer: 17, head: 0, mode: 'prefixed', row: null },
    // Qwen3-4B until /health says otherwise
    arch: { nLayers: 36, nq: 32, nkv: 8, G: 4, hd: 128 },
    limits: { maxK: 16, maxR: 64, maxN: 64, maxP: 64 },
  };
  if (backend.status !== 'offline') {
    try {
      const h = await backend.api.health();
      initial.arch = h.arch;
      initial.limits = h.limits;
      initial.view.layer = Math.min(initial.view.layer, h.arch.nLayers - 1);
    } catch {
      /* keep the defaults */
    }
  }
  const store = new Store(initial);

  const demo = mountDemo(store);
  mountCanvas(store);
  mountSolveFigure(store);

  const badgeEl = $('demo-badge') as HTMLButtonElement;
  const tipEl = $('demo-tip');
  badgeEl.addEventListener('click', () => {
    if (store.s.backend.status === 'offline') return;
    const open = tipEl.hidden;
    tipEl.hidden = !open;
    badgeEl.setAttribute('aria-expanded', String(open));
  });
  // the info tip: why there is a line break after every message, and (without a GPU) that data are simulated
  const renderBadge = () => {
    const b = store.s.backend;
    badgeEl.onclick = null;
    badgeEl.removeAttribute('aria-label');
    const parts = b.status === 'mock' ? [t('demo.tip'), t('chat.info')] : [t('chat.info')];
    tipEl.innerHTML = parts.map((x) => `<span class="tip-p">${rich(x, true)}</span>`).join('');
    if (b.status === 'mock') {
      badgeEl.textContent = `${t('demo.badge')} ⓘ`;
    } else if (b.status === 'live') {
      badgeEl.textContent = 'ⓘ';
      badgeEl.setAttribute('aria-label', t('aria.info'));
    } else {
      tipEl.hidden = true;
      badgeEl.textContent = `${t('offline')} ${t('offline.use')}`;
      badgeEl.onclick = () => {
        // forget a remembered server address, so later visits do not open offline again
        try {
          localStorage.removeItem('am-api');
        } catch {
          /* storage blocked */
        }
        store.set({ backend: { api: new MockApi(), status: 'mock' }, picked: undefined, solve: undefined, made: undefined, loaded: undefined, gen: undefined });
        void start();
      };
    }
  };
  store.on((_, ch) => {
    if (ch.has('backend') || ch.has('lang')) renderBadge();
  });

  $('lang').addEventListener('click', () => {
    setLang(getLang() === 'zh' ? 'en' : 'zh');
    drawIntroFigure(introBox);
    drawMechFigure(mechBox);
    store.set({ lang: getLang() });
  });

  // mark the nav link of the section being read: the last one whose top has passed 35% of the viewport
  const links = Array.from(document.querySelectorAll<HTMLAnchorElement>('.topnav a'));
  const sections = ['top', 'demo', 'method', 'results', 'limits'].map((id) => $(id));
  let raf = 0;
  const spy = () => {
    raf = 0;
    let current = 'top';
    for (const sec of sections) if (sec.getBoundingClientRect().top <= innerHeight * 0.35) current = sec.id;
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) current = 'limits';
    links.forEach((a) => a.classList.toggle('is-here', a.getAttribute('href') === `#${current}`));
  };
  addEventListener('scroll', () => (raf ||= requestAnimationFrame(spy)), { passive: true });
  spy();

  /** First visit: step 1 opens with an example chosen; computing the prefix is the visitor's click. */
  async function start() {
    await demo.pickExample(store.s.presetId ?? first.id);
  }

  store.set({ lang: initial.lang, backend, presetId: initial.presetId, view: initial.view, r: initial.r });
  if (backend.status !== 'offline') void start();
}

void boot();
