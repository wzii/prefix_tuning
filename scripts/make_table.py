"""
Tables from the JSON written by evaluate.py.

    python scripts/make_table.py results/gpt2.json            # one table per repetition factor
    python scripts/make_table.py results/qwen3.json --summary  # mean over the 15 groups of each dataset
    python scripts/make_table.py results/gpt2.json --csv results/gpt2.csv
"""
import argparse
import csv
import json

DATASETS = ["and", "but", "comma", "dot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--summary", action="store_true", help="mean success per dataset and r")
    ap.add_argument("--csv", help="also write every cell to this CSV file")
    args = ap.parse_args()
    rows = json.load(open(args.results))
    cell = {(r["dataset"], r["in_len"], r["out_len"], r["repeat"]): r["success"] for r in rows}
    datasets = [d for d in DATASETS if any(r["dataset"] == d for r in rows)]
    in_lens = sorted({r["in_len"] for r in rows})
    out_lens = sorted({r["out_len"] for r in rows})
    repeats = sorted({r["repeat"] for r in rows})

    if args.summary:
        print("| | " + " | ".join(f"`{d}`" for d in datasets) + " |")
        print("|---|" + "---:|" * len(datasets))
        for rep in repeats:
            means = []
            for d in datasets:
                v = [s for (dd, _, _, rr), s in cell.items() if dd == d and rr == rep]
                means.append(f"{sum(v) / len(v):.4f}" if v else "–")
            print(f"| r = {rep} | " + " | ".join(means) + " |")
    else:
        for rep in repeats:
            print(f"\nr = {rep}\n")
            print("| dataset | k | " + " | ".join(f"n = {n}" for n in in_lens) + " |")
            print("|---|---:|" + "---:|" * len(in_lens))
            for d in datasets:
                for k in out_lens:
                    vals = [cell.get((d, n, k, rep)) for n in in_lens]
                    print(f"| `{d}` | {k} | " + " | ".join("–" if v is None else f"{v:.4f}" for v in vals) + " |")

    if args.csv:
        keys = ["dataset", "in_len", "out_len", "repeat", "success", "denom", "gain", "still_false", "keep", "lost"]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in sorted(rows, key=lambda r: (DATASETS.index(r["dataset"]), r["repeat"], r["out_len"], r["in_len"])):
                w.writerow([r.get(k_) for k_ in keys])


if __name__ == "__main__":
    main()
