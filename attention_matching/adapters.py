"""
Model adapters: everything attention matching needs to know about one architecture.

An adapter supplies, for every attention layer,
  * the inputs of the attention module in the source run    (collect_reps)
  * the queries, keys and values the solver works with       (qkv_for_solver)
  * the dedicated prefix key of each query                   (build_key_prefix)
and patches the model so that a stored prefix is placed in front of the keys and values
of every attention layer (install).

Two architectures are provided:
  GPT2Adapter   multi-head attention, learned absolute positions, biases (GPT-2)
  Qwen3Adapter  grouped-query attention, RoPE, QK-Norm, no biases (Qwen3)

In both, the prefix is attended to but never attends, carries no position, and is kept
outside the KV cache, so the positions of the real tokens are unaffected.
"""
import types
from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .solver import qknorm_key_prefix, scaled_key_prefix


class PrefixState:
    """The prefix of every layer, whether it is active, and the repetition factor r."""

    def __init__(self):
        self.data, self.active, self.repeat = {}, False, 1

    def get(self, layer_idx, dtype):
        """The stored (k, v) of a layer, each prefix vector repeated r times, or None."""
        if not self.active or layer_idx not in self.data:
            return None
        k, v = self.data[layer_idx]
        return (k.repeat_interleave(self.repeat, dim=-2).to(dtype),
                v.repeat_interleave(self.repeat, dim=-2).to(dtype))


def _attend(q, k, v, attention_mask, scale):
    """Softmax attention over [prefix, tokens]: prefix columns are visible to every query."""
    w = (q @ k.transpose(-1, -2)) * scale
    if attention_mask is not None:
        if attention_mask.shape[-1] != k.shape[-2]:
            attention_mask = F.pad(attention_mask, (k.shape[-2] - attention_mask.shape[-1], 0), value=0.0)
        w = w + attention_mask[..., : k.shape[-2]]
    elif q.shape[-2] > 1:
        L, S = q.shape[-2], k.shape[-2]
        causal = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(S - L)
        w = w.masked_fill(~causal, torch.finfo(w.dtype).min)
    w = torch.softmax(w, -1, dtype=torch.float32).to(q.dtype)
    return w @ v


class Adapter(ABC):
    """Architecture-specific half of attention matching."""

    def __init__(self, model):
        self.model = model
        self.state = PrefixState()
        self.install()

    # ---- prefix plumbing ----
    def clear(self):
        self.state.data, self.state.active = {}, False

    def set_prefix(self, layer_idx, k, v):
        """Store one layer's prefix in the form the patched attention consumes."""
        self.state.data[layer_idx] = (k.to(torch.float32), v.to(torch.float32))

    def encode(self, layer_idx, Kp, Vp):
        """Turn the solver's keys and values into the stored prefix (identity by default)."""
        return Kp, Vp

    @contextmanager
    def use(self, prefix, repeat=1):
        """Run the model with `prefix` (a Prefix or {layer: (k, v)}) repeated `repeat` times."""
        layers = getattr(prefix, "layers", prefix)
        dev = next(self.model.parameters()).device
        self.clear()
        for li, (k, v) in layers.items():
            self.set_prefix(li, k.to(dev), v.to(dev))
        self.state.repeat, self.state.active = repeat, True
        try:
            yield
        finally:
            self.state.active = False

    # ---- architecture-specific ----
    @property
    @abstractmethod
    def n_layers(self): ...

    @abstractmethod
    def collect_reps(self, ids):
        """Forward pass over x' followed by y; return {layer: inputs of the attention module}."""

    @abstractmethod
    def qkv_for_solver(self, layer_idx, x_src, p, k, dtype):
        """-> qy (n_kv, G, k, hd) decoding queries; kw, vw (n_kv, T, hd) keys and values."""

    @abstractmethod
    def build_key_prefix(self, layer_idx, qy, dtype):
        """-> (n_kv, G*k, hd) prefix keys, one per stacked query."""

    @abstractmethod
    def install(self):
        """Patch every attention layer so that the active prefix is injected."""


# ============================================================ GPT-2
class GPT2Adapter(Adapter):
    """Multi-head attention (G = 1), learned absolute positions, Q/K/V biases.

    Prefix keys follow eq. (12): each is scaled until its logit equals `attn_lim`
    (20 in the paper). The prefix is stored as keys and values; since c_attn is affine,
    this is the same as storing the hidden vectors s and projecting them.
    """

    def __init__(self, model, attn_lim=20.0):
        cfg = model.config
        self.D, self.NQ = cfg.n_embd, cfg.n_head
        self.HD = self.D // self.NQ
        self.NKV, self.G = self.NQ, 1
        self.attn_lim = attn_lim
        super().__init__(model)

    @property
    def n_layers(self):
        return self.model.config.n_layer

    def _weights(self, layer_idx, dtype):
        a = self.model.transformer.h[layer_idx].attn
        W, b = a.c_attn.weight.to(dtype), a.c_attn.bias.to(dtype)        # Conv1D: (D, 3D)
        D = self.D
        return W[:, :D], W[:, D:2 * D], W[:, 2 * D:], b[:D], b[D:2 * D], b[2 * D:]

    def collect_reps(self, ids):
        reps = {}
        hooks = [blk.register_forward_hook(
            (lambda i: lambda m, inp, out: reps.__setitem__(i, m.ln_1(inp[0]).detach()))(i))
            for i, blk in enumerate(self.model.transformer.h)]
        try:
            with torch.no_grad():
                self.model(ids)
        finally:
            for h in hooks:
                h.remove()
        return reps

    def qkv_for_solver(self, layer_idx, x_src, p, k, dtype):
        Wq, Wk, Wv, bq, bk, bv = self._weights(layer_idx, dtype)
        x = x_src.to(dtype)[0]
        T = x.shape[0]
        heads = lambda z: z.view(T, self.NQ, self.HD).transpose(0, 1)    # (NQ, T, hd)
        q, kw, vw = heads(x @ Wq + bq), heads(x @ Wk + bk), heads(x @ Wv + bv)
        qy = q[:, p - 1: p - 1 + k, :].reshape(self.NKV, self.G, k, self.HD)
        return qy, kw, vw

    def build_key_prefix(self, layer_idx, qy, dtype):
        _, Wk, _, _, bk, _ = self._weights(layer_idx, dtype)
        Wk_h = Wk.view(self.D, self.NQ, self.HD).permute(1, 0, 2)          # (NQ, D, hd)
        q = qy.reshape(self.NQ, -1, self.HD)
        return scaled_key_prefix(q, Wk_h, bk.view(self.NQ, self.HD), self.attn_lim)

    def install(self):
        state, HD = self.state, self.HD
        scale = HD ** -0.5

        def forward(attn, hidden_states, past_key_values=None, cache_position=None,
                    attention_mask=None, head_mask=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, output_attentions=False, **kwargs):
            q, k, v = attn.c_attn(hidden_states).split(attn.split_size, dim=2)
            split = lambda z: z.view(*z.shape[:-1], -1, HD).transpose(1, 2)
            q, k, v = split(q), split(k), split(v)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, attn.layer_idx, {"cache_position": cache_position})
            pref = state.get(attn.layer_idx, k.dtype)
            if pref is not None:
                B = hidden_states.shape[0]
                kp, vp = (t.unsqueeze(0).expand(B, -1, -1, -1) for t in pref)
                k, v = torch.cat([kp, k], 2), torch.cat([vp, v], 2)
            out = _attend(q, k, v, attention_mask, scale)
            out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
            return attn.resid_dropout(attn.c_proj(out)), None

        for blk in self.model.transformer.h:
            blk.attn.forward = types.MethodType(forward, blk.attn)


# ============================================================ Qwen3
class Qwen3Adapter(Adapter):
    """Grouped-query attention (32 query / 8 key-value heads in Qwen3-4B), RoPE, QK-Norm.

    Prefix keys are the QK-Norm maximisers gamma * u* (Section 6 of the paper). The prefix
    is stored as hidden vectors, one per prefix slot and KV head, and passed through the
    layer's own k_proj, k_norm and v_proj at injection time. The prefix keys and values are
    concatenated after RoPE and outside the KV cache: written into the cache, they would
    shift the cache position from which Qwen3 derives the RoPE positions of real tokens.
    """

    def __init__(self, model):
        cfg = model.config
        self.NQ, self.NKV = cfg.num_attention_heads, cfg.num_key_value_heads
        self.HD, self.EPS = cfg.head_dim, cfg.rms_norm_eps
        self.G = self.NQ // self.NKV
        super().__init__(model)

    @property
    def n_layers(self):
        return self.model.config.num_hidden_layers

    def _attn(self, layer_idx):
        return self.model.model.layers[layer_idx].self_attn

    def collect_reps(self, ids):
        reps = {}
        hooks = [layer.register_forward_hook(
            (lambda i: lambda m, inp, out: reps.__setitem__(i, m.input_layernorm(inp[0]).detach()))(i))
            for i, layer in enumerate(self.model.model.layers)]
        try:
            with torch.no_grad():
                self.model(ids)
        finally:
            for h in hooks:
                h.remove()
        return reps

    def qkv_for_solver(self, layer_idx, x_src, p, k, dtype):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        a = self._attn(layer_idx)
        NQ, NKV, HD, EPS = self.NQ, self.NKV, self.HD, self.EPS
        x = x_src.to(dtype)
        T = x.shape[1]
        rms = lambda z: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + EPS)
        q = rms((x @ a.q_proj.weight.to(dtype).T).view(1, T, NQ, HD)) * a.q_norm.weight.to(dtype)
        kk = rms((x @ a.k_proj.weight.to(dtype).T).view(1, T, NKV, HD)) * a.k_norm.weight.to(dtype)
        vv = (x @ a.v_proj.weight.to(dtype).T).view(1, T, NKV, HD)
        cos, sin = self.model.model.rotary_emb(x.float(), torch.arange(T, device=x.device)[None])
        q, kk = apply_rotary_pos_emb(q.transpose(1, 2), kk.transpose(1, 2), cos.to(dtype), sin.to(dtype))
        qy = q[0, :, p - 1: p - 1 + k, :].reshape(NKV, self.G, k, HD)
        return qy, kk[0], vv[0].transpose(0, 1)

    def build_key_prefix(self, layer_idx, qy, dtype):
        gamma = self._attn(layer_idx).k_norm.weight.to(dtype)
        return qknorm_key_prefix(qy, gamma).reshape(self.NKV, -1, self.HD)

    def encode(self, layer_idx, Kp, Vp, dtype=torch.float64):
        """Prefix keys and values -> hidden vectors s, one per slot and KV head.

        k_norm(W_K s) = gamma * u* reduces to the linear system W_K^h s = u*, because
        rms(u*) = 1. Each head's W^h (hd x D) has full row rank, so the minimum-norm
        solution s = W^T (W W^T)^-1 t is exact.
        """
        a = self._attn(layer_idx)
        Wk = a.k_proj.weight.to(dtype).view(self.NKV, self.HD, -1)
        Wv = a.v_proj.weight.to(dtype).view(self.NKV, self.HD, -1)
        gamma = a.k_norm.weight.to(dtype)
        minnorm = lambda W, t: (W.transpose(-1, -2) @ torch.linalg.solve(
            W @ W.transpose(-1, -2), t.transpose(-1, -2))).transpose(-1, -2)
        return minnorm(Wk, Kp.to(dtype) / gamma), minnorm(Wv, Vp.to(dtype))

    def install(self):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        state, NKV, HD = self.state, self.NKV, self.HD

        def forward(attn, hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None, **kwargs):
            shape = (*hidden_states.shape[:-1], -1, HD)
            q = attn.q_norm(attn.q_proj(hidden_states).view(shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(hidden_states).view(shape)).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(shape).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, attn.layer_idx, {"sin": sin, "cos": cos,
                                                                     "cache_position": cache_position})
            got = state.get(attn.layer_idx, hidden_states.dtype)
            if got is not None:
                sk, sv = got                                                # (NKV, m*r, D)
                Wk = attn.k_proj.weight.view(NKV, HD, -1)
                Wv = attn.v_proj.weight.view(NKV, HD, -1)
                kp = attn.k_norm(torch.einsum("gmd,ghd->gmh", sk, Wk))
                vp = torch.einsum("gmd,ghd->gmh", sv, Wv)
                B = hidden_states.shape[0]
                k = torch.cat([kp.unsqueeze(0).expand(B, -1, -1, -1).to(k.dtype), k], 2)
                v = torch.cat([vp.unsqueeze(0).expand(B, -1, -1, -1).to(v.dtype), v], 2)
            out = _attend(q, repeat_kv(k, attn.num_key_value_groups),
                          repeat_kv(v, attn.num_key_value_groups), attention_mask, attn.scaling)
            out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
            return attn.o_proj(out), None

        for layer in self.model.model.layers:
            layer.self_attn.forward = types.MethodType(forward, layer.self_attn)


def adapter_for(model, **kwargs):
    """Pick the adapter for a loaded Hugging Face model."""
    kind = model.config.model_type
    if kind == "gpt2":
        return GPT2Adapter(model, **kwargs)
    if kind == "qwen3":
        return Qwen3Adapter(model, **kwargs)
    raise NotImplementedError(f"no adapter for model type {kind!r}")
