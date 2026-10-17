import type { Token } from '../api/types';
import { hashStr } from '../lib/rng';

/**
 * Stand-in for the Qwen3 tokenizer: words keep their leading space, digits are split
 * one per token (as Qwen does), punctuation is its own token, newlines are separate.
 * Chinese is split per character, which is coarser than Qwen's real vocabulary.
 */
const RE = /\n|\p{Script=Han}|[ \t]*(?:(?!\p{Script=Han})[\p{L}'])+|[ \t]*\p{N}|[ \t]*[^\s\p{L}\p{N}]|[ \t]+/gu;

export function tokenize(text: string): Token[] {
  return (text.match(RE) ?? []).map((t) => ({ id: hashStr(t) % 151_643, text: t }));
}
