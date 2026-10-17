"""
Attention matching: the linear system that turns a source run into a prefix.

For one attention layer, let q_t be the query that produces y_{t+1} in the source run
(x' followed by y), t = 0..k-1. In the source run, q_t attends to its p+t visible tokens
with weights R[t, :] and the head outputs a_t = R[t, :] V_src. The prefix gives every
query a dedicated key k~_t on which it puts (almost) all of its attention; the prefix
values V~ are then chosen so that the prefix alone reproduces a_t:

    S V~ = R V_src,   S[t, j] = softmax_j(q_t . k~_j / sqrt(d)),

which is solved as S W = R, V~ = W V_src. With grouped-query attention, the G query heads
that share a key/value head share its prefix, and their systems are stacked into one of
size m = G k (G = 1 recovers multi-head attention).

All solves run in float64.
"""
import math

import torch


@torch.no_grad()
def solve_layer(adapter, layer_idx, x_src, p, k, dtype=torch.float64):
    """Solve one layer.

    x_src: the inputs of the attention module in the source run, positions 0..p+k-2,
           shape (1, p+k-1, D).
    Returns (Kp, Vp, diag): prefix keys and values of shape (n_kv, G*k, head_dim), and
    diagnostics (condition numbers of S, relative residual, prefix and real logits).
    """
    qy, kw, vw = adapter.qkv_for_solver(layer_idx, x_src, p, k, dtype)
    n_kv, G, _, hd = qy.shape
    T = kw.shape[1]
    rt = math.sqrt(hd)
    dev = qy.device

    Kp = adapter.build_key_prefix(layer_idx, qy, dtype)              # (n_kv, G*k, hd)

    # S: each stacked query against every prefix key of its KV head
    S = torch.softmax((qy @ Kp.unsqueeze(1).transpose(-1, -2)) / rt, -1)
    S = S.reshape(n_kv, G * k, G * k)

    # R: the source run's causal attention at each decoding step (the paper's [X|Y])
    logits = (qy @ kw.unsqueeze(1).transpose(-1, -2)) / rt           # (n_kv, G, k, T)
    visible = torch.arange(T, device=dev)[None, :] < (p + torch.arange(k, device=dev)[:, None])
    R = torch.softmax(logits.masked_fill(~visible, float("-inf")), -1)
    R = R.masked_fill(~visible, 0.0).reshape(n_kv, G * k, T)

    W = torch.linalg.solve(S, R)
    Vp = W @ vw

    own = torch.diagonal((qy @ Kp.view(n_kv, G, k, hd).transpose(-1, -2)) / rt, dim1=-2, dim2=-1)
    diag = {
        "cond": torch.linalg.cond(S).flatten().tolist(),
        "resid": ((S @ W - R).norm() / R.norm()).item(),
        "logit_prefix": own.min(-1).values.flatten().tolist(),
        "logit_real": logits.amax((-2, -1)).flatten().tolist(),
    }
    return Kp, Vp, diag


def scaled_key_prefix(q, Wk, bk, attn_lim):
    """Prefix keys for attention without QK-Norm (eq. (12) of the paper).

    The dedicated key of query q is k~ = W_K s + b_K with s along W_K^T q, scaled so
    that the logit q . k~ / sqrt(d) equals attn_lim.

    q:  (H, k, hd) queries;  Wk: (H, D, hd) per-head key weights;  bk: (H, hd) or None.
    Returns (H, k, hd).
    """
    rt = math.sqrt(q.shape[-1])
    k_dir = (q @ Wk.transpose(-1, -2)) @ Wk                         # W_K W_K^T q
    slope = (q * k_dir).sum(-1) / rt
    if bk is None:
        return (attn_lim / slope).unsqueeze(-1) * k_dir
    b = bk.unsqueeze(-2)
    offset = (q * b).sum(-1) / rt
    return ((attn_lim - offset) / slope).unsqueeze(-1) * k_dir + b


def qknorm_key_prefix(q, gamma):
    """Prefix keys for attention with QK-Norm (Section 6 of the paper).

    Every key has the form gamma * u with ||u|| = sqrt(d), so scaling has no effect. By
    Cauchy-Schwarz the logit is largest at u* = sqrt(d) (gamma * q) / ||gamma * q||,
    where it equals ||gamma * q||.

    q: (..., hd) queries;  gamma: (hd,) key-norm gain.  Returns gamma * u*.
    """
    rt = math.sqrt(q.shape[-1])
    w = q * gamma
    return rt * w / w.norm(dim=-1, keepdim=True) * gamma
