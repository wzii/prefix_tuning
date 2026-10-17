"""
Is every prefix key reachable from an input vector of the attention module?

The prefix of a key/value head is a set of input vectors s. A target key is reachable if
W_K^h s = u* has a solution, which holds for every target when the head's key projection
W_K^h (head_dim x hidden) has full row rank. This prints the rank and the condition number
of W_K^h for every layer and head, in float64.

    python verify/check_projection_rank.py --model Qwen/Qwen3-4B
"""
import argparse

import torch
from transformers import AutoModelForCausalLM


def key_heads(model):
    """{layer: (n_kv, head_dim, hidden) key projection of each key/value head}."""
    cfg = model.config
    if cfg.model_type == "gpt2":
        D, H = cfg.n_embd, cfg.n_head
        for i, blk in enumerate(model.transformer.h):
            Wk = blk.attn.c_attn.weight[:, D:2 * D].double()            # Conv1D: x @ W
            yield i, Wk.T.reshape(H, D // H, D)
    else:
        for i, layer in enumerate(model.model.layers):
            a = layer.self_attn
            yield i, a.k_proj.weight.double().view(cfg.num_key_value_heads, a.head_dim, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    worst_rank, worst_cond, heads = None, 0.0, 0
    for i, W in key_heads(model):
        s = torch.linalg.svdvals(W)                                    # (n_kv, head_dim)
        tol = s[:, :1] * max(W.shape[1:]) * torch.finfo(torch.float64).eps
        rank = (s > tol).sum(-1)
        cond = (s[:, 0] / s[:, -1]).max().item()
        heads += W.shape[0]
        worst_rank = rank.min().item() if worst_rank is None else min(worst_rank, rank.min().item())
        worst_cond = max(worst_cond, cond)
        print(f"layer {i:2d}: rank {rank.min().item()}/{W.shape[1]} for all {W.shape[0]} heads, "
              f"largest condition number {cond:.1f}")
    print(f"\n{heads} key/value heads; smallest rank {worst_rank} (head_dim {W.shape[1]}), "
          f"largest condition number {worst_cond:.1f}")


if __name__ == "__main__":
    main()
