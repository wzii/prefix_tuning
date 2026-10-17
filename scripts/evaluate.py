"""
Success rates of attention matching under the protocol of Section 5.

A dataset is a text file of sentences that all end with the same token (" and", " but",
"," or "."). Sentences are grouped by token length n. Within a group, every sentence
serves once as the source x'; its greedy continuation y of k tokens defines a prefix, and
every other sentence x of the group is an input. A pair counts as a success if the plain
model does not already continue x with y but the prefixed model does:

    success = #{prefixed output = y} / #{plain output != y},

where pairs whose plain output already equals y are left out (they are reported as
"keep" and "lost").

    python scripts/evaluate.py --model gpt2 --datasets and but comma dot --repeats 90
    python scripts/evaluate.py --model Qwen/Qwen3-4B --datasets and but comma dot --repeats 1 10
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_matching import adapter_for, build_prefix, greedy, load_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def buckets(path, tok):
    """{token length: [token ids]} for the sentences of one dataset file."""
    out = {}
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.strip():
            ids = tok(line)["input_ids"]
            out.setdefault(len(ids), []).append(ids)
    return out


def run_cell(model, adapter, inputs, k, repeats, bs, pad):
    """All ordered (source, input) pairs of one group, for every repetition factor r."""
    dev = next(model.parameters()).device
    batches = [torch.tensor(inputs[i:i + bs], device=dev) for i in range(0, len(inputs), bs)]

    def generate_all():
        outs = []
        for b in batches:
            o = greedy(model, b, k, pad)
            if o.shape[1] < k:                     # every input stopped early: pad to k
                o = torch.nn.functional.pad(o, (0, k - o.shape[1]), value=pad)
            outs.append(o[:, :k])
        return torch.cat(outs)

    base = generate_all()
    stats = {r: dict(gain=0, still_false=0, keep=0, lost=0) for r in repeats}
    n_src = n_skipped = violations = 0
    cond_max = resid_max = 0.0
    for s, source in enumerate(inputs):
        prefix = build_prefix(adapter, source, k, pad_token_id=pad)
        if prefix is None:                         # the model stops before k tokens
            n_skipped += 1
            continue
        n_src += 1
        cond_max = max(cond_max, max(max(d["cond"]) for d in prefix.diag))
        resid_max = max(resid_max, max(d["resid"] for d in prefix.diag))
        violations += sum(x[-1] != source[-1] for x in inputs)
        y = torch.tensor(prefix.target_ids, device=dev)
        plain_hit = (base == y).all(1)
        for r in repeats:
            with adapter.use(prefix, repeat=r):
                hit = (generate_all() == y).all(1)
            st = stats[r]
            for i in range(len(inputs)):
                if i == s:
                    continue
                b, h = bool(plain_hit[i]), bool(hit[i])
                st["gain" if h and not b else "lost" if b and not h else "keep" if b else "still_false"] += 1
    return stats, dict(n_src=n_src, n_skipped=n_skipped, last_token_violations=violations,
                       cond_max=cond_max, resid_max=resid_max)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2", help="gpt2 or Qwen/Qwen3-4B")
    ap.add_argument("--data", default=None, help="dataset folder (default: data/gpt2 or data/qwen3)")
    ap.add_argument("--datasets", nargs="+", default=["and", "but", "comma", "dot"])
    ap.add_argument("--in-lens", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    ap.add_argument("--out-lens", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--repeats", nargs="+", type=int, default=None,
                    help="repetition factors r (default: 90 for GPT-2, 1 10 for Qwen3)")
    ap.add_argument("--attn-lim", type=float, default=20.0, help="logit of the prefix keys (GPT-2)")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="results JSON (appended to, so runs can resume)")
    args = ap.parse_args()

    model, tok = load_model(args.model, args.device)
    family = model.config.model_type
    adapter = adapter_for(model, **({"attn_lim": args.attn_lim} if family == "gpt2" else {}))
    repeats = args.repeats or ([90] if family == "gpt2" else [1, 10])
    data = args.data or os.path.join(ROOT, "data", family)
    out = args.out or os.path.join(ROOT, "results", f"{family}.json")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["dataset"], r["in_len"], r["out_len"], r["repeat"]) for r in rows}
    for name in args.datasets:
        groups = buckets(os.path.join(data, f"{name}.txt"), tok)
        for n in args.in_lens:
            inputs = groups.get(n, [])
            if len(inputs) < 2:
                print(f"[{name} n={n}] no group of this length, skipped", flush=True)
                continue
            for k in args.out_lens:
                todo = [r for r in repeats if (name, n, k, r) not in done]
                if not todo:
                    continue
                t0 = time.time()
                stats, info = run_cell(model, adapter, inputs, k, todo, args.batch_size, pad)
                for r in todo:
                    st = stats[r]
                    denom = st["gain"] + st["still_false"]
                    rows.append(dict(model=args.model, dataset=name, in_len=n, out_len=k, repeat=r,
                                     success=st["gain"] / denom if denom else float("nan"),
                                     denom=denom, n_inputs=len(inputs), **st, **info))
                    print(f"[{name} n={n:>3} k={k:>2} r={r:>2}] success {rows[-1]['success']:.4f} "
                          f"({st['gain']}/{denom})  cond(S) <= {info['cond_max']:.1f}  "
                          f"[{time.time() - t0:.0f}s]", flush=True)
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
