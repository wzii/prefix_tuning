import { getLang } from '../i18n';
import type { Store } from '../store';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** The tutorial video in a dialog, in the page's language (files in public/video/). */
export function mountTutorial(store: Store) {
  const dialog = $<HTMLDialogElement>('tut');
  const video = $<HTMLVideoElement>('tut-video');
  const base = import.meta.env.BASE_URL;

  const setSource = () => {
    const lang = getLang();
    if (video.dataset.lang === lang) return;
    video.dataset.lang = lang;
    video.poster = `${base}video/tutorial-${lang}.jpg`;
    // H.264 for most browsers; VP9 for builds without it (some Linux and open-source browsers)
    video.replaceChildren(
      ...[['mp4', 'video/mp4'], ['webm', 'video/webm']].map(([ext, type]) => {
        const s = document.createElement('source');
        s.src = `${base}video/tutorial-${lang}.${ext}`;
        s.type = type;
        return s;
      }),
    );
    video.load();
  };

  $('tut-open').addEventListener('click', () => {
    setSource();
    dialog.showModal();
    void video.play().catch(() => {
      /* autoplay refused: the controls are there */
    });
  });
  const close = () => dialog.close();
  $('tut-close').addEventListener('click', close);
  // a click on the backdrop (outside the box) closes it too
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) close();
  });
  dialog.addEventListener('close', () => video.pause());

  setSource();
  store.on((_, ch) => {
    // switching language while the video plays would restart it; switch once it is closed
    if (ch.has('lang') && !dialog.open) setSource();
  });
}
