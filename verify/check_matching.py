"""
The mechanism behind the theorem, measured layer by layer.

With the prefix, the input x (n = p tokens, x_n = x'_p) is decoded while every attention
head reproduces the output it had in the source run. The hidden states at the k decoding
positions should then approach those of the source run as the repetition factor r grows,
in every layer. This prints the largest relative difference over layers and positions.

    python verify/check_matching.py                          # GPT-2
    python verify/check_matching.py --model Qwen/Qwen3-4B --repeats 1 10 100
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from attention_matching import adapter_for, build_prefix, load_model  # noqa: E402


@torch.no_grad()
def hidden(model, ids, first, k):
    out = model(torch.tensor([ids], device=next(model.parameters()).device), output_hidden_states=True)
    return torch.stack([h[0, first: first + k] for h in out.hidden_states[1:]])   # (layers, k, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--length", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", nargs="+", type=int, default=[1, 10, 100, 1000])
    args = ap.parse_args()

    model, tok = load_model(args.model, "cuda" if torch.cuda.is_available() else "cpu")
    adapter = adapter_for(model)
    family = model.config.model_type
    lines = open(os.path.join(ROOT, "data", family, "and.txt"), encoding="utf-8").read().splitlines()
    group = [tok(s)["input_ids"] for s in lines if len(tok(s)["input_ids"]) == args.length]
    source, x = group[0], group[1]
    prefix = build_prefix(adapter, source, args.k, pad_token_id=tok.eos_token_id)
    y = prefix.target_ids
    p, n, k = len(source), len(x), len(y)

    ref = hidden(model, source + y[:-1], p - 1, k)          # source run, positions producing y
    print(f"x' = {tok.decode(source)!r}\nx  = {tok.decode(x)!r}\ny  = {tok.decode(y)!r}\n")
    print(f"max cond(S) = {max(max(d['cond']) for d in prefix.diag):.1f}, "
          f"max residual |SW - R| / |R| = {max(d['resid'] for d in prefix.diag):.1e}\n")
    print(f"{'r':>6} | max relative difference of hidden states to the source run")
    plain = hidden(model, x + y[:-1], n - 1, k)
    rel = lambda h: ((h - ref).norm(dim=-1) / ref.norm(dim=-1)).max().item()
    print(f"{'none':>6} | {rel(plain):.2e}")
    for r in args.repeats:
        with adapter.use(prefix, repeat=r):
            h = hidden(model, x + y[:-1], n - 1, k)
        print(f"{r:>6} | {rel(h):.2e}")


if __name__ == "__main__":
    main()
