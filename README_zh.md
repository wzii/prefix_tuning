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

Prefix-tuning 冻结预训练语言模型的全部权重，只在每个 attention 层的 key 和 value 前面加上少量可学习的向量。这么少的参数，表达能力究竟有多强？我们证明：在较弱的条件下，可以用**闭式解直接构造**出一个 prefix，让预训练的自回归 Transformer 生成**任何它本身能够生成的输出序列**，并且对所有长度相同、末尾 token 相同的输入都成立。我们把这一构造称为 **attention matching**。它无需训练：先记录模型生成目标输出时各个 attention head 的输出，再为每个 head 解一个小型线性方程组，让 prefix 把这些输出重现出来。本仓库提供该方法在 GPT-2（multi-head attention）和 Qwen3-4B（带 QK-Norm 的 grouped-query attention）上的实现、实验数据集，以及复现论文结果的脚本。

👉 **[试试我们的在线演示](https://wzii.github.io/prefix_tuning/#demo)**：选好一句话，无论你输入什么，接入 prefix 的 Qwen3-4B 都会说出这句话。

## 目录

- [文件结构](#文件结构)
- [环境配置](#环境配置)
- [下载](#下载)
- [使用方法](#使用方法)
- [示例](#示例)
- [实验结果](#实验结果)
- [致谢](#致谢)
- [许可证](#许可证)
- [引用](#引用)

## 文件结构

```
.
├── attention_matching/          # 方法的实现（一个小型 Python 包）
│   ├── solver.py                #   线性方程组 S W = R，以及 prefix 的 key
│   ├── adapters.py              #   GPT-2 与 Qwen3 的 query/key/value 计算和 prefix 注入
│   └── prefix.py                #   build_prefix()、Prefix（保存与加载）、load_model()
├── data/
│   ├── gpt2/{and,but,comma,dot}.txt    # 论文 Table 1 所用数据集（GPT-2 tokenizer）
│   └── qwen3/{and,but,comma,dot}.txt   # 同一批句子按 Qwen3 tokenizer 重新构建
├── docs/                        # 项目主页源码，由 GitHub Actions 构建并发布
├── examples/
│   └── quickstart.py            # 构造一个 prefix，并用在没见过的输入上
├── scripts/
│   ├── evaluate.py              # 按论文第 5 节的实验设置计算成功率
│   └── make_table.py            # 根据 evaluate.py 的结果生成表格
├── verify/
│   ├── check_matching.py        # 对比接入 prefix 后与 source 运行的隐藏状态
│   └── check_projection_rank.py # 验证每个 prefix key 都能由某个输入向量得到
└── results/                     # 论文实验的逐组结果
```

## 环境配置

需要 Python 3.10 及以上版本和较新的 PyTorch。由于修改后的 attention 依赖 Hugging Face Transformers 的内部实现，我们固定了 Transformers 的版本。

```bash
conda create -n attention-matching python=3.12 -y
conda activate attention-matching
pip install -r requirements.txt
```

## 下载

数据集已随仓库提供：`data/gpt2/` 供 GPT-2 使用，`data/qwen3/` 供 Qwen3-4B 使用，两者由同一批句子分别按各自的 tokenizer 构建。四个数据集各有五组、每组 50 句，长度分别正好是 5、10、20、50、100 个 token；同一数据集中的句子都以同一个 token 结尾（` and`、` but`、`,` 或 `.`）。

模型会在第一次使用时自动从 Hugging Face Hub 下载，也可以提前下载：

```bash
huggingface-cli download gpt2
huggingface-cli download Qwen/Qwen3-4B
```

## 使用方法

**Python 接口。** 先用 source 输入 `x'` 构造 prefix，之后对任何长度相同、末尾 token 相同的输入都可以直接生成：

```python
from attention_matching import adapter_for, build_prefix, greedy, load_model

model, tok = load_model("Qwen/Qwen3-4B")          # 或 "gpt2"
adapter = adapter_for(model)                       # 替换各层的 attention

source = tok("The old man sat and")["input_ids"]
prefix = build_prefix(adapter, source, k=10)       # 每层、每个 KV head 各解一个线性方程组
print(tok.decode(prefix.target_ids))               # y：模型对 x' 的贪心续写

x = tok("End the long night and")["input_ids"]     # n = p，且 x 与 x' 的末尾 token 相同
with adapter.use(prefix, repeat=1):                # repeat 即第 5 节中的重复次数 r
    print(tok.decode(greedy(model, x, prefix.k)[0]))   # 输出 y

prefix.save("prefix.safetensors")                  # 可用 Prefix.load(...) 重新加载
```

**复现论文中的表格。** `scripts/evaluate.py` 会在每个长度组内遍历所有有序句子对，输出长度 k 取 1、5、10。论文中的全部实验都在一张 24 GB 显存的 GPU 上完成，模型权重使用 float32，线性方程组用 float64 求解。GPT-2 也能在 CPU 上运行，用来跑[示例](#示例)已经足够。

```bash
# Table 1：GPT-2，attn_lim = 20，r = 90
python scripts/evaluate.py --model gpt2 --repeats 90 --out results/gpt2.json
python scripts/make_table.py results/gpt2.json

# 第 6 节：Qwen3-4B，r = 1 和 r = 10
python scripts/evaluate.py --model Qwen/Qwen3-4B --repeats 1 10 --out results/qwen3.json
python scripts/make_table.py results/qwen3.json --summary
```

通过 `--datasets`、`--in-lens`、`--out-lens` 可以只跑其中一部分设置。结果以追加方式写入输出文件，运行中断后重新执行即可从断点继续。

**正确性检验。**

```bash
python verify/check_matching.py --model gpt2              # 隐藏状态与 source 运行一致
python verify/check_projection_rank.py --model Qwen/Qwen3-4B
```

## 示例

运行 `python examples/quickstart.py`，会用 `data/gpt2/and.txt` 中第一个长度为 10 个 token 的句子构造 prefix，再把它用在随后的五个句子上：

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

`verify/check_matching.py` 揭示了背后的机制：接入 prefix 后，各层在解码位置上的隐藏状态与 source 运行几乎完全一致（相对误差约为 1e-6，已达到 float32 的精度；不接入 prefix 时为 0.63）。

## 实验结果

对每一对（source x'，输入 x），如果原模型对 x 的续写不是 y，而接入 prefix 后变成了 y，就记为成功；原模型本来就续写出 y 的输入对不计入统计。

**GPT-2**（Table 1，attn_lim = 20，r = 90）。行为数据集和输出长度 k，列为输入长度 n。

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

**Qwen3-4B**（第 6 节）：每个数据集 15 个 (k, n) 组的平均成功率。r = 10 时，全部 60 组都达到 1.0000。

| | `and` | `but` | `comma` | `dot` |
|---|---:|---:|---:|---:|
| r = 1 | 0.9977 | 0.9966 | 0.9950 | 0.9968 |
| r = 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

各组的详细数据见 [`results/`](results/)。

## 致谢

本实现基于 [PyTorch](https://pytorch.org) 和 [Hugging Face Transformers](https://github.com/huggingface/transformers)。感谢 OpenAI 开源 [GPT-2](https://huggingface.co/openai-community/gpt2)，感谢 Qwen 团队开源 [Qwen3](https://huggingface.co/Qwen/Qwen3-4B)。

## 许可证

本代码以 [Apache License 2.0](LICENSE) 发布。

## 引用

```bibtex
@inproceedings{lam2026prefix,
  title     = {Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers},
  author    = {Lam, Peter Cho-Ho and Wang, Ziyi and Zhou, Zirui},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

[arxiv]: #
[project]: https://wzii.github.io/prefix_tuning/
