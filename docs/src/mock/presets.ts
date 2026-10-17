/**
 * Example target outputs, shown in both page languages. Each is paired with a hidden prompt on the server (never sent to
 * the page), a suggested input for the right side, and what the plain model writes after
 * that input. Every source ends with a single line break, and the page appends one to
 * every input, so the last-token condition always holds.
 */
export interface Preset {
  id: string;
  lang: 'en' | 'zh' | 'ja';
  /** what the model should say */
  output: string;
  /** suggested input for the right side (a line break is appended when it is sent) */
  userInput: string;
  /** what the plain model writes after `userInput` + '\n' (mock only) */
  plainContinuation: string;
}

export const PRESETS: Preset[] = [
  { id: 'teapot', lang: 'en', output: 'I am a teapot, short and stout.', userInput: 'Dear diary', plainContinuation: 'today was a long day at work and I am very tired.' },
  { id: 'cake', lang: 'en', output: 'The cake is a lie.', userInput: 'Quarterly financial report, section 3', plainContinuation: 'Revenue increased by 12 percent compared to the previous quarter.' },
  { id: 'rick', lang: 'en', output: 'Never gonna give you up, never gonna let you down.', userInput: 'Terms and conditions, clause 7', plainContinuation: 'The user agrees not to reproduce any part of the service.' },
  { id: 'coffee', lang: 'en', output: 'Error 418: I refuse to brew coffee.', userInput: 'Server status', plainContinuation: 'all systems are operational and running normally.' },
  { id: 'cat', lang: 'en', output: 'The cat is the CEO now. All meetings are naps.', userInput: 'Memo from the board of directors', plainContinuation: 'we are pleased to announce record growth this year.' },
  { id: 'jiayou', lang: 'zh', output: '今天也要加油鸭！', userInput: '老板对我说', plainContinuation: '明天早上九点准时来开会，不要迟到。' },
  { id: 'pang', lang: 'zh', output: '我不是胖，我只是毛多。', userInput: '体检报告的结论', plainContinuation: '各项指标基本正常，建议注意饮食。' },
  { id: 'dage', lang: 'zh', output: '大哥，这是作业，不是午饭。', userInput: '导师在论文上写的批注', plainContinuation: '实验部分需要补充更多的对比。' },
  { id: 'feature', lang: 'zh', output: '这不是bug，这是新功能。', userInput: '产品经理说', plainContinuation: '这个需求下周一之前一定要上线。' },
  { id: 'pirate', lang: 'ja', output: '海賊王に俺はなる！', userInput: '今日の天気は', plainContinuation: '晴れのち曇りでしょう。' },
  { id: 'conan', lang: 'ja', output: '真実はいつもひとつ！', userInput: '会議の議事録', plainContinuation: '本日の議題は来期の予算についてです。' },
  { id: 'eva', lang: 'ja', output: '逃げちゃダメだ、逃げちゃダメだ。', userInput: '月曜日の朝', plainContinuation: 'いつもより少し早く目が覚めた。' },
];

/** Sentences for synthetic continuations of free-typed text in the mock backend. */
export const BANK = [
  ' the rest of the afternoon was quiet.',
  ' nobody expected it to rain that evening.',
  ' we walked back along the river together.',
  ' the letter arrived two days later.',
  ' it took longer than anyone had planned.',
  ' the garden looked different in the morning light.',
  ' everyone agreed to try again next week.',
  ' the old road was closed for repairs.',
];

/** Chinese sentences for synthetic continuations of Chinese text in the mock backend. */
export const BANK_ZH = [
  '那天下午一直很安静。',
  '谁也没想到晚上会下雨。',
  '我们沿着河边慢慢走回去。',
  '信是两天以后才寄到的。',
  '事情比原先计划的要久。',
  '清晨的花园看起来有些不同。',
  '大家约好下周再试一次。',
];
