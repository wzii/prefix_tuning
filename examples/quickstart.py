"""
Build one prefix and use it on inputs it has never seen.

    python examples/quickstart.py                         # GPT-2, runs on a CPU
    python examples/quickstart.py --model Qwen/Qwen3-4B   # needs a GPU with 24 GB

The source and the inputs are sentences of the same token length that end with the
same token (" and"), as the theorem requires (n = p and x_n = x'_p).
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from attention_matching import adapter_for, build_prefix, greedy, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--length", type=int, default=10, help="token length n = p of source and inputs")
    ap.add_argument("--k", type=int, default=10, help="output length")
    ap.add_argument("--repeat", type=int, default=None, help="r (default: 90 for GPT-2, 1 for Qwen3)")
    ap.add_argument("--save", default="prefix.safetensors")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    adapter = adapter_for(model)
    family = model.config.model_type
    repeat = args.repeat or (90 if family == "gpt2" else 1)
    pad = tok.eos_token_id

    lines = open(os.path.join(ROOT, "data", family, "and.txt"), encoding="utf-8").read().splitlines()
    group = [s for s in lines if len(tok(s)["input_ids"]) == args.length]
    source, inputs = group[0], group[1:6]

    prefix = build_prefix(adapter, tok(source)["input_ids"], args.k, pad_token_id=pad)
    prefix.save(args.save)
    print(f"source  x' = {source!r}")
    print(f"target  y  = {tok.decode(prefix.target_ids)!r}")
    print(f"prefix: {len(prefix.layers)} layers, saved to {args.save}; "
          f"max cond(S) = {max(max(d['cond']) for d in prefix.diag):.1f}\n")

    for x in inputs:
        ids = tok(x)["input_ids"]
        plain = greedy(model, ids, args.k, pad)[0]
        with adapter.use(prefix, repeat=repeat):
            steered = greedy(model, ids, args.k, pad)[0]
        mark = "✓" if steered.tolist() == prefix.target_ids else "✗"
        print(f"input        {x!r}")
        print(f"  plain      {tok.decode(plain)!r}")
        print(f"  prefixed   {tok.decode(steered)!r}  {mark}\n")


if __name__ == "__main__":
    main()
