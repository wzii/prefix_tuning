"""
Building, saving and loading prefixes.

    model, tok = load_model("Qwen/Qwen3-4B")
    adapter = adapter_for(model)
    prefix = build_prefix(adapter, tok("The old man sat and")["input_ids"], k=10)
    with adapter.use(prefix, repeat=1):
        out = model.generate(**tok("Drink a glass of milk and", return_tensors="pt"),
                             max_new_tokens=prefix.k, do_sample=False)
"""
import json
from dataclasses import dataclass, field

import torch

from .solver import solve_layer


@dataclass
class Prefix:
    """A prefix for every attention layer, and the run it was built from.

    layers:     {layer: (k, v)} as the adapter stores them (CPU, float32)
    source_ids: the source input x' (p tokens)
    target_ids: its greedy continuation y (k tokens), which the prefix reproduces
    diag:       per-layer diagnostics of the linear systems
    """
    layers: dict
    source_ids: list
    target_ids: list
    model: str = ""
    diag: list = field(default_factory=list)

    @property
    def p(self):
        return len(self.source_ids)

    @property
    def k(self):
        return len(self.target_ids)

    @property
    def last_token(self):
        """x'_p: an input must end with this token for the guarantee to apply."""
        return self.source_ids[-1]

    def save(self, path):
        from safetensors.torch import save_file
        tensors = {}
        for li, (k, v) in self.layers.items():
            tensors[f"layers.{li}.key"] = k.contiguous()
            tensors[f"layers.{li}.value"] = v.contiguous()
        meta = {"model": self.model, "source_ids": json.dumps(self.source_ids),
                "target_ids": json.dumps(self.target_ids)}
        save_file(tensors, path, metadata=meta)

    @classmethod
    def load(cls, path):
        from safetensors import safe_open
        layers = {}
        with safe_open(path, "pt") as f:
            meta = f.metadata()
            for name in f.keys():
                _, li, kind = name.split(".")
                layers.setdefault(int(li), {})[kind] = f.get_tensor(name)
        return cls({li: (t["key"], t["value"]) for li, t in sorted(layers.items())},
                   json.loads(meta["source_ids"]), json.loads(meta["target_ids"]),
                   model=meta.get("model", ""))


@torch.no_grad()
def greedy(model, ids, k, pad_token_id=None):
    """Greedy continuation of a batch of equal-length inputs: (B, n) -> (B, <= k)."""
    ids = torch.as_tensor(ids, device=next(model.parameters()).device)
    if ids.dim() == 1:
        ids = ids[None]
    if pad_token_id is None:
        eos = model.generation_config.eos_token_id
        pad_token_id = eos[0] if isinstance(eos, list) else eos
    # greedy decoding; sampling settings shipped with the model do not apply
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=k,
                         do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=pad_token_id)
    return out[:, ids.shape[1]:]


@torch.no_grad()
def build_prefix(adapter, source_ids, k, target_ids=None, pad_token_id=None):
    """Construct the prefix that makes the model continue any input like `source_ids`.

    The model first continues the source greedily for k tokens (or `target_ids` is used
    if given; it must be the model's own greedy continuation). One linear system per
    layer and KV head then yields the prefix (Algorithm 1 of the paper).
    Returns a Prefix, or None if the model stops before k tokens.
    """
    model = adapter.model
    dev = next(model.parameters()).device
    src = torch.as_tensor(source_ids, device=dev).reshape(1, -1)
    p = src.shape[1]
    was_active, adapter.state.active = adapter.state.active, False
    try:
        y = (torch.as_tensor(target_ids, device=dev).reshape(1, -1) if target_ids is not None
             else greedy(model, src, k, pad_token_id))
        if y.shape[1] < k:
            return None
        full = torch.cat([src, y[:, :k]], 1)
        reps = adapter.collect_reps(full)
        layers, diag = {}, []
        for li in range(adapter.n_layers):
            Kp, Vp, d = solve_layer(adapter, li, reps[li][:, : p + k - 1, :], p, k)
            sk, sv = adapter.encode(li, Kp, Vp)
            layers[li] = (sk.to(torch.float32).cpu(), sv.to(torch.float32).cpu())
            diag.append(d)
    finally:
        adapter.state.active = was_active
    return Prefix(layers, src[0].tolist(), y[0, :k].tolist(),
                  model=getattr(model.config, "_name_or_path", ""), diag=diag)


def load_model(name, device="cuda", dtype=torch.float32):
    """Load a causal LM with eager attention (the patched attention is eager)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager")
    return model.to(device).eval(), tok
