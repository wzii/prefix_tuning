/** Chat avatars: an ordinary robot, and a "broken" one that carries the prefix. Colors follow the theme. */
export const ROBOT_OK = `<svg viewBox="0 0 40 40" class="robot robot-ok" aria-hidden="true">
  <line x1="20" y1="4" x2="20" y2="10" class="r-line"/>
  <circle cx="20" cy="4" r="2.6" class="r-fill"/>
  <rect x="7" y="10" width="26" height="22" rx="6" class="r-head"/>
  <circle cx="15" cy="20" r="3" class="r-fill"/>
  <circle cx="25" cy="20" r="3" class="r-fill"/>
  <path d="M15 27 Q20 30 25 27" class="r-line"/>
  <rect x="3" y="17" width="4" height="7" rx="1.5" class="r-fill"/>
  <rect x="33" y="17" width="4" height="7" rx="1.5" class="r-fill"/>
</svg>`;

export const ROBOT_BROKEN = `<svg viewBox="0 0 40 40" class="robot robot-broken" aria-hidden="true">
  <path d="M20 10 L20 6 L25 3" class="r-line"/>
  <circle cx="25.5" cy="3" r="2.4" class="r-fill"/>
  <rect x="7" y="10" width="26" height="22" rx="6" class="r-head"/>
  <path d="M19 10 L16.5 15 L20.5 17.5 L17.5 23" class="r-crack"/>
  <path d="M12.5 17.5 L17.5 22.5 M17.5 17.5 L12.5 22.5" class="r-line"/>
  <circle cx="25" cy="20" r="3" class="r-fill"/>
  <path d="M14.5 28 L17 26.5 L19.5 28 L22 26.5 L24.5 28 L27 26.5" class="r-line"/>
  <rect x="3" y="17" width="4" height="7" rx="1.5" class="r-fill"/>
  <rect x="33" y="17" width="4" height="7" rx="1.5" class="r-fill"/>
  <path d="M35 8 l1.5 -3 M37.5 10 l3 -1.5 M34 5.5 l0 -3" class="r-spark"/>
</svg>`;

export const USER_ICON = `<svg viewBox="0 0 40 40" class="robot robot-user" aria-hidden="true">
  <circle cx="20" cy="15" r="7" class="r-fill"/>
  <path d="M7 35 C9 26 31 26 33 35 Z" class="r-fill"/>
</svg>`;

/** The prefix: a hammer (it is what breaks the robot). */
export const HAMMER = `<svg viewBox="0 0 32 32" class="hammer" aria-hidden="true">
  <rect x="14.2" y="11" width="3.6" height="18" rx="1.4" class="h-handle"/>
  <path d="M6 5.5 H22 L26.5 3.5 V12.5 L22 10.5 H6 Q4.5 10.5 4.5 9 V7 Q4.5 5.5 6 5.5 Z" class="h-head"/>
  <rect x="13.4" y="10.5" width="5.2" height="2.6" rx="0.8" class="h-collar"/>
</svg>`;
