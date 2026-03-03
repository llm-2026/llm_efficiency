"""
Per-step benchmark: no KV cache vs KV cache (kv_cache.py).

At generation step t:
  no KV  → forward over all t tokens  → attention O(t²)
  KV cat → forward over 1 new token   → attention O(t)
"""

import time
from pathlib import Path

import numpy as np
import torch

from kv_cache import GPT_kv
from mingpt.model import GPT

BLOCK_SIZE = 2048
VOCAB = 128

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def make_config(model_type):
    c = GPT.get_default_config()
    c.model_type = model_type
    c.vocab_size = VOCAB
    c.block_size = BLOCK_SIZE
    return c


def bench(fn, n_warmup=3, n_runs=5):
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.mean(ts) * 1e3  # ms


def run(model_type, seq_lengths, out=None):
    def log(line=""):
        print(line)
        if out is not None:
            out.write(line + "\n")

    model_base = GPT(make_config(model_type)).to(device).eval()
    model_kv = GPT_kv(make_config(model_type)).to(device).eval()
    n_layer = model_kv.n_layer

    log(f"{model_type}  ({sum(p.numel() for p in model_base.parameters())/1e6:.1f}M params)")
    log(f"{'Context T':>10}  {'no KV (ms)':>12}  {'KV (ms)':>12}  {'speedup':>9}")
    log("-" * 50)

    with torch.no_grad():
        for T in seq_lengths:
            inp = torch.randint(VOCAB, (1, T), dtype=torch.long, device=device)

            # no KV: recompute all T tokens every step
            t_no_kv = bench(lambda: model_base(inp))

            # KV cat: prefill T-1 tokens, then measure one new-token step
            kv_cache = [None] * n_layer
            _, _, kv_cache = model_kv(
                inp[:, :-1], kv_cache=kv_cache, compute_first=True
            )
            t_kv = bench(lambda: model_kv(inp, kv_cache=kv_cache, compute_first=False))

            log(f"{T:>10}  {t_no_kv:>12.2f}  {t_kv:>12.2f}  {t_no_kv/t_kv:>8.2f}x")
    log()


if __name__ == "__main__":
    print(f"device: {device}\n")
    output_path = Path(__file__).parent / "benchmark_results.txt"
    with output_path.open("w") as f:
        f.write(f"device: {device}\n\n")
        run("gpt-mini", seq_lengths=[64, 128, 256, 512, 1024, 1536], out=f)
        run("gpt2", seq_lengths=[64, 128, 256, 512, 1024, 1536], out=f)
    print(f"Results saved to {output_path}")
