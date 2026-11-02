import './style.css';
import { connect, watch, type Backend } from './api';
import { applyStatic, getLang, setLang, t } from './i18n';
import { PRESETS } from './mock/presets';
import { mountDemo } from './sections/demo';
import { mountTutorial } from './sections/tutorial';
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
  if (backend.status === 'live') {
    const h = backend.health;
    initial.arch = h.arch;
    initial.limits = h.limits;
    initial.view.layer = Math.min(initial.view.layer, h.arch.nLayers - 1);
  }
  const store = new Store(initial);

  const demo = mountDemo(store);
  mountTutorial(store);
  mountCanvas(store);
  mountSolveFigure(store);

  const badgeEl = $('demo-badge') as HTMLButtonElement;
  const tipEl = $('demo-tip');
  badgeEl.addEventListener('click', () => {
    if (store.s.liveReady) return goLive(store.s.liveReady);
    const open = tipEl.hidden;
    tipEl.hidden = !open;
    badgeEl.setAttribute('aria-expanded', String(open));
  });
  // the info tip: why there is a line break after every message, and (without a GPU) that data are simulated
  const renderBadge = () => {
    const b = store.s.backend;
    badgeEl.removeAttribute('aria-label');
    badgeEl.classList.toggle('is-back', !!store.s.liveReady);
    const parts = b.status === 'live' ? [t('chat.info')] : [t('demo.tip'), t('chat.info')];
    // escape the address before it goes into the HTML of the tip
    if (b.status === 'fallback') parts.unshift(t('demo.fallback', { url: `${b.base}/health`.replace(/[<>"'&\s]/g, encodeURIComponent) }));
    tipEl.innerHTML = parts.map((x) => `<span class="tip-p">${rich(x, true)}</span>`).join('');
    if (store.s.liveReady) {
      badgeEl.textContent = `${t('demo.liveBack')} →`;
    } else if (b.status === 'live') {
      badgeEl.textContent = 'ⓘ';
      badgeEl.setAttribute('aria-label', t('aria.info'));
    } else {
      badgeEl.textContent = `${t(b.status === 'fallback' ? 'demo.badgeFallback' : 'demo.badge')} ⓘ`;
    }
  };
  store.on((_, ch) => {
    if (ch.has('backend') || ch.has('lang') || ch.has('liveReady')) renderBadge();
  });

  /** Switch from the simulation to the GPU server; what was made on the simulation goes. */
  function goLive(live: Extract<Backend, { status: 'live' }>) {
    const h = live.health;
    demo.reset();
    tipEl.hidden = true;
    badgeEl.setAttribute('aria-expanded', 'false');
    store.set({
      backend: live,
      liveReady: undefined,
      arch: h.arch,
      limits: h.limits,
      view: { ...store.s.view, layer: Math.min(store.s.view.layer, h.arch.nLayers - 1) },
      picked: undefined,
      solve: undefined,
      made: undefined,
      loaded: undefined,
      gen: undefined,
      leftMsg: undefined,
      solveMsg: undefined,
      rightMsg: undefined,
    });
    void start();
  }
  // on the fallback, keep checking the server; once it answers, switch at once if the visitor
  // has not started anything, else offer the switch on the badge
  if (backend.status === 'fallback') {
    watch(backend.base, (live) => {
      const s = store.s;
      const idle = !s.made && !s.loaded && !s.gen && !s.solving && !s.running && !s.finding;
      if (idle) goLive(live);
      else store.set({ liveReady: live });
    });
  }

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
  // the server could not be reached: show why the data are simulated, and what may help
  if (backend.status === 'fallback') {
    tipEl.hidden = false;
    badgeEl.setAttribute('aria-expanded', 'true');
  }
  void start();
}

void boot();
