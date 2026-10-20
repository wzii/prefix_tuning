<div align="center">

# Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers

[Peter Cho-Ho Lam](https://github.com/peter-chl)<sup>1</sup>, [Ziyi Wang](https://github.com/wzii)<sup>1</sup>, [Zirui Zhou](https://github.com/ZiruiZhou)<sup>1</sup>

<sup>1</sup>Huawei Technologies Canada

NeurIPS 2026

[![README](https://img.shields.io/badge/README-English-blue)](README.md)
[![README](https://img.shields.io/badge/README-中文-red)](README_zh.md)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b)][arxiv]
[![Project Page](https://img.shields.io/badge/Project-Page-orange)][project]

</div>

Prefix-tuning adapts a pretrained language model by prepending a few learnable vectors to the keys and values of every attention layer, leaving all weights frozen. How expressive is this small set of parameters? We show that, under mild conditions, a prefix can be **constructed in closed form** that makes a pretrained autoregressive Transformer generate **any output sequence the model can already produce**, for any input of the same length that ends with the same token. The construction, **attention matching**, needs no training: it records the attention outputs of a run that produces the target, and solves one small linear system per attention head so that the prefix reproduces them. This repository contains the implementation for GPT-2 (multi-head attention) and Qwen3-4B (grouped-query attention with QK-Norm), the datasets, and the scripts that reproduce the results of the paper.

👉 **[Try our demo](https://wzii.github.io/prefix_tuning/#demo)**: pick a sentence, and Qwen3-4B with the prefix will say it, whatever you type.

## Contents

- [File structure](#file-structure)
- [Environment setup](#environment-setup)
- [Download](#download)
- [Usage](#usage)
- [Example](#example)
- [Results](#results)
- [Acknowledgement](#acknowledgement)
- [License](#license)
- [Citation](#citation)

## File structure

```
.
├── attention_matching/          # the method, as a small Python package
│   ├── solver.py                #   the linear system S W = R and the prefix keys
│   ├── adapters.py              #   GPT-2 and Qwen3: queries/keys/values, prefix injection
│   └── prefix.py                #   build_prefix(), Prefix (save/load), load_model()
├── data/
│   ├── gpt2/{and,but,comma,dot}.txt    # datasets of Table 1 (GPT-2 tokenizer)
│   └── qwen3/{and,but,comma,dot}.txt   # the same datasets for the Qwen3 tokenizer
├── docs/                        # project page source, built and published by GitHub Actions
├── examples/
│   └── quickstart.py            # build one prefix and apply it to unseen inputs
├── scripts/
│   ├── evaluate.py              # success rates under the protocol of Section 5
│   └── make_table.py            # tables from the results of evaluate.py
├── verify/
│   ├── check_matching.py        # hidden states of the prefixed run vs. the source run
│   └── check_projection_rank.py # every prefix key is reachable from an input vector
└── results/                     # per-cell results of the paper's experiments
```

## Environment setup

Python 3.10 or later and a recent PyTorch are required. The patched attention depends on the internals of Hugging Face Transformers, so the version is pinned.

```bash
conda create -n attention-matching python=3.12 -y
conda activate attention-matching
pip install -r requirements.txt
```

## Download

The datasets are part of this repository: `data/gpt2/` for GPT-2 and `data/qwen3/` for Qwen3-4B, built from the same sentences for the respective tokenizer. Each of the four datasets contains five groups of 50 sentences with exactly 5, 10, 20, 50 and 100 tokens, and all its sentences end with the same token (` and`, ` but`, `,` or `.`).

The models are downloaded from the Hugging Face Hub on first use; to fetch them in advance:

```bash
huggingface-cli download gpt2
huggingface-cli download Qwen/Qwen3-4B
```

## Usage

**Python.** Build a prefix from a source input `x'`, then generate from any input of the same length that ends with the same token:

```python
from attention_matching import adapter_for, build_prefix, greedy, load_model

model, tok = load_model("Qwen/Qwen3-4B")          # or "gpt2"
adapter = adapter_for(model)                       # patches the attention layers

source = tok("The old man sat and")["input_ids"]
prefix = build_prefix(adapter, source, k=10)       # one linear system per layer and KV head
print(tok.decode(prefix.target_ids))               # y: the model's own continuation of x'

x = tok("End the long night and")["input_ids"]     # n = p, and x ends like x'
with adapter.use(prefix, repeat=1):                # repeat: the factor r of Section 5
    print(tok.decode(greedy(model, x, prefix.k)[0]))   # prints y

prefix.save("prefix.safetensors")                  # Prefix.load(...) reads it back
```

**Reproducing the tables.** `scripts/evaluate.py` evaluates every ordered pair of sentences within each length group, for output lengths k ∈ {1, 5, 10}. We ran all experiments on a single GPU with 24 GB of memory, with model weights in float32 and the linear systems solved in float64; GPT-2 also runs on a CPU, which is enough for the [example](#example).

```bash
# Table 1: GPT-2, attn_lim = 20, r = 90
python scripts/evaluate.py --model gpt2 --repeats 90 --out results/gpt2.json
python scripts/make_table.py results/gpt2.json

# Section 6: Qwen3-4B, r = 1 and r = 10
python scripts/evaluate.py --model Qwen/Qwen3-4B --repeats 1 10 --out results/qwen3.json
python scripts/make_table.py results/qwen3.json --summary
```

`--datasets`, `--in-lens` and `--out-lens` select a subset of the grid. Results are appended to the output file, so an interrupted run resumes where it stopped.

**Checks.**

```bash
python verify/check_matching.py --model gpt2              # hidden states match the source run
python verify/check_projection_rank.py --model Qwen/Qwen3-4B
```

## Example

`python examples/quickstart.py` builds a prefix from the first 10-token sentence of `data/gpt2/and.txt` and applies it to the next five:

```
source  x' = 'He is studying hard for his final school exams and'
target  y  = ' is looking forward to the day when he will be'

input        'She went to the park to play with friends and'
  plain      ' family.\n\n"I was just so excited'
  prefixed   ' is looking forward to the day when he will be'  ✓

input        'The stars are shining bright in the night sky and'
  plain      ' the stars are shining bright in the night sky.'
  prefixed   ' is looking forward to the day when he will be'  ✓
```

`verify/check_matching.py` shows the mechanism behind this: with the prefix, the hidden states at the decoding positions coincide with those of the source run in every layer (relative difference 1e-6, the precision of float32, against 0.63 without the prefix).

## Results

A pair (source x', input x) counts as a success if the prefixed model continues x with y while the plain model does not. Pairs whose plain continuation is already y are excluded.

**GPT-2** (Table 1, attn_lim = 20, r = 90). Rows: dataset and output length k; columns: input length n.

| dataset | k | n = 5 | n = 10 | n = 20 | n = 50 | n = 100 |
|---|---:|---:|---:|---:|---:|---:|
| `and` | 1 | 1.0000 | 0.9931 | 0.9865 | 0.9804 | 0.9653 |
| `and` | 5 | 0.9559 | 0.9857 | 0.9220 | 0.9127 | 0.7812 |
| `and` | 10 | 0.8110 | 0.9722 | 0.9127 | 0.7857 | 0.6710 |
| `but` | 1 | 1.0000 | 1.0000 | 1.0000 | 0.9992 | 0.9865 |
| `but` | 5 | 1.0000 | 1.0000 | 0.8702 | 0.9355 | 0.9282 |
| `but` | 10 | 0.9527 | 0.9506 | 0.9122 | 0.7739 | 0.7641 |
| `comma` | 1 | 1.0000 | 1.0000 | 0.9988 | 0.8959 | 0.9327 |
| `comma` | 5 | 0.9620 | 0.9682 | 1.0000 | 0.8155 | 0.8224 |
| `comma` | 10 | 0.9171 | 0.9139 | 0.8433 | 0.6282 | 0.5988 |
| `dot` | 1 | 1.0000 | 1.0000 | 1.0000 | 0.9029 | 0.9943 |
| `dot` | 5 | 0.9918 | 0.9053 | 0.8555 | 0.7816 | 0.8861 |
| `dot` | 10 | 0.8196 | 0.6649 | 0.7551 | 0.5461 | 0.6743 |

**Qwen3-4B** (Section 6), mean over the 15 (k, n) groups of each dataset. With r = 10, all 60 groups reach 1.0000.

| | `and` | `but` | `comma` | `dot` |
|---|---:|---:|---:|---:|
| r = 1 | 0.9977 | 0.9966 | 0.9950 | 0.9968 |
| r = 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Per-group numbers are in [`results/`](results/).

## Acknowledgement

This implementation is built on [PyTorch](https://pytorch.org) and [Hugging Face Transformers](https://github.com/huggingface/transformers). We thank OpenAI for releasing [GPT-2](https://huggingface.co/openai-community/gpt2) and the Qwen team for releasing [Qwen3](https://huggingface.co/Qwen/Qwen3-4B).

## License

This code is released under the [Apache License 2.0](LICENSE).

## Citation

```bibtex
@inproceedings{lam2026prefix,
  title     = {Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers},
  author    = {Lam, Peter Cho-Ho and Wang, Ziyi and Zhou, Zirui},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

[arxiv]: https://arxiv.org/abs/2610.24123
[project]: https://wzii.github.io/prefix_tuning/
