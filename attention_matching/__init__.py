"""Prefix-tuning for arbitrary output sequences by attention matching."""
from .adapters import Adapter, GPT2Adapter, Qwen3Adapter, adapter_for
from .prefix import Prefix, build_prefix, greedy, load_model
from .solver import qknorm_key_prefix, scaled_key_prefix, solve_layer

__all__ = [
    "Adapter", "GPT2Adapter", "Qwen3Adapter", "adapter_for",
    "Prefix", "build_prefix", "greedy", "load_model",
    "solve_layer", "scaled_key_prefix", "qknorm_key_prefix",
]
__version__ = "1.0.0"
