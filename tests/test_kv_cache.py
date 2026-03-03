"""
Tests for kv_cache.py — used for automated grading of student implementations.

Graded components:
  1. CausalSelfAttention_kv.forward  — attention with KV cache
  2. Block_kv.forward                — transformer block with KV cache
  3. GPT_kv.forward                  — full GPT forward with KV cache
  4. GPT_kv.generate_kv              — KV-cached autoregressive generation
"""

import pytest
import torch

from mingpt.model import GPT, Block

from kv_cache.kv_cache import (
    CausalSelfAttention,
    CausalSelfAttention_kv,
    Block_kv,
    Config,
    GPT_kv,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    """Small config for attention / block tests (n_head=3, n_embd=15, block_size=11)."""
    return Config()


@pytest.fixture(scope="module")
def x(cfg):
    """Random input tensor of shape (B=4, T=block_size, C=n_embd)."""
    torch.manual_seed(42)
    return torch.randn(4, cfg.block_size, cfg.n_embd)


@pytest.fixture(scope="module")
def gpt_cfg():
    """GPT config with direct params so constructors never mutate it in-place."""
    c = GPT.get_default_config()
    c.model_type = None  # use explicit params to avoid in-place mutation by GPT.__init__
    c.n_layer = 3
    c.n_head = 3
    c.n_embd = 48
    c.vocab_size = 5
    c.block_size = 20
    return c


@pytest.fixture(scope="module")
def idx(gpt_cfg):
    """Random token index sequence of shape (B=2, T=7)."""
    torch.manual_seed(42)
    return torch.randint(0, gpt_cfg.vocab_size, (2, 7))


# ── 1. CausalSelfAttention_kv ─────────────────────────────────────────────────

class TestCausalSelfAttentionKV:

    def test_returns_output_and_cache(self, cfg, x):
        """forward() must return a 2-tuple (output, kv_cache)."""
        csa_kv = CausalSelfAttention_kv(cfg)
        csa_kv.eval()
        with torch.no_grad():
            result = csa_kv(x)
        assert isinstance(result, tuple) and len(result) == 2, \
            "forward() must return (output, kv_cache)"

    def test_output_shape(self, cfg, x):
        """Output tensor must have the same shape as the input (B, T, C)."""
        csa_kv = CausalSelfAttention_kv(cfg)
        csa_kv.eval()
        with torch.no_grad():
            out, _ = csa_kv(x)
        assert out.shape == x.shape, \
            f"Expected output shape {x.shape}, got {out.shape}"

    def test_kv_cache_shape(self, cfg, x):
        """KV cache must be a list [k, v] each of shape (B, T, C)."""
        csa_kv = CausalSelfAttention_kv(cfg)
        csa_kv.eval()
        B, T, C = x.shape
        with torch.no_grad():
            _, kv = csa_kv(x)
        assert isinstance(kv, (list, tuple)) and len(kv) == 2, \
            "kv_cache must be [k, v]"
        assert kv[0].shape == (B, T, C), \
            f"k shape: expected {(B, T, C)}, got {kv[0].shape}"
        assert kv[1].shape == (B, T, C), \
            f"v shape: expected {(B, T, C)}, got {kv[1].shape}"

    def test_no_cache_matches_baseline(self, cfg, x):
        """Without cache, output must match CausalSelfAttention (same weights)."""
        csa = CausalSelfAttention(cfg)
        csa_kv = CausalSelfAttention_kv(cfg)
        csa_kv.load_state_dict(csa.state_dict())
        csa.eval()
        csa_kv.eval()
        with torch.no_grad():
            out_ref = csa(x)
            out_kv, _ = csa_kv(x)
        assert torch.allclose(out_ref, out_kv, atol=1e-5), \
            "Without cache, CausalSelfAttention_kv output must match CausalSelfAttention"

    def test_last_token_with_cache(self, cfg, x):
        """Last token output computed with (T-1) cached entries must match full forward."""
        csa_kv = CausalSelfAttention_kv(cfg)
        csa_kv.eval()
        with torch.no_grad():
            out_full, kv = csa_kv(x)
            out_last, _ = csa_kv(
                x[:, [-1], :],
                kv_cache=[kv[0][:, :-1, :], kv[1][:, :-1, :]],
            )
        assert torch.allclose(out_full[:, -1, :], out_last[:, 0, :], atol=1e-4), \
            "Last-token output with cache must match full-sequence last-token output"

    def test_incremental_matches_full(self, cfg, x):
        """Step-by-step (token-by-token) output must match full forward at every position."""
        csa_kv = CausalSelfAttention_kv(cfg)
        csa_kv.eval()
        _, T, _ = x.shape
        with torch.no_grad():
            out_full, _ = csa_kv(x)
            # Start with the first token (no cache yet)
            out_t, kv = csa_kv(x[:, :1, :])
            assert torch.allclose(out_full[:, 0, :], out_t[:, 0, :], atol=1e-4), \
                "Mismatch at position 0"
            # Extend one token at a time
            for t in range(1, T):
                out_t, kv = csa_kv(x[:, [t], :], kv_cache=kv)
                assert torch.allclose(out_full[:, t, :], out_t[:, 0, :], atol=1e-4), \
                    f"Mismatch at position {t}"


# ── 2. Block_kv ───────────────────────────────────────────────────────────────

class TestBlockKV:

    def test_returns_output_and_cache(self, cfg, x):
        """forward() must return a 2-tuple (output, kv_cache)."""
        bkv = Block_kv(cfg)
        bkv.eval()
        with torch.no_grad():
            result = bkv(x)
        assert isinstance(result, tuple) and len(result) == 2, \
            "Block_kv.forward() must return (output, kv_cache)"

    def test_output_shape(self, cfg, x):
        """Output tensor must have the same shape as the input (B, T, C)."""
        bkv = Block_kv(cfg)
        bkv.eval()
        with torch.no_grad():
            out, _ = bkv(x)
        assert out.shape == x.shape, \
            f"Expected output shape {x.shape}, got {out.shape}"

    def test_no_cache_matches_block(self, cfg, x):
        """Without cache, output must match mingpt Block (same weights)."""
        block = Block(cfg)
        bkv = Block_kv(cfg)
        bkv.load_state_dict(block.state_dict())
        block.eval()
        bkv.eval()
        with torch.no_grad():
            out_ref = block(x)
            out_kv, _ = bkv(x)
        assert torch.allclose(out_ref, out_kv, atol=1e-5), \
            "Without cache, Block_kv output must match mingpt Block output"

    def test_last_token_with_cache(self, cfg, x):
        """Last token output computed with (T-1) cached entries must match full forward."""
        bkv = Block_kv(cfg)
        bkv.eval()
        with torch.no_grad():
            out_full, kv = bkv(x)
            out_last, _ = bkv(
                x[:, [-1], :],
                kv_cache=[kv[0][:, :-1, :], kv[1][:, :-1, :]],
            )
        assert torch.allclose(out_full[:, -1, :], out_last[:, 0, :], atol=1e-4), \
            "Last-token output with cache must match full-sequence last-token output"

    def test_incremental_matches_full(self, cfg, x):
        """Step-by-step output must match full forward at every position."""
        bkv = Block_kv(cfg)
        bkv.eval()
        _, T, _ = x.shape
        with torch.no_grad():
            out_full, _ = bkv(x)
            out_t, kv = bkv(x[:, :1, :])
            assert torch.allclose(out_full[:, 0, :], out_t[:, 0, :], atol=1e-4), \
                "Mismatch at position 0"
            for t in range(1, T):
                out_t, kv = bkv(x[:, [t], :], kv_cache=kv)
                assert torch.allclose(out_full[:, t, :], out_t[:, 0, :], atol=1e-4), \
                    f"Mismatch at position {t}"


# ── 3. GPT_kv.forward ─────────────────────────────────────────────────────────

class TestGPTKVForward:

    def test_no_cache_matches_gpt(self, gpt_cfg, idx):
        """GPT_kv.forward without cache must match GPT.forward (same weights)."""
        gpt = GPT(gpt_cfg)
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.load_state_dict(gpt.state_dict())
        gpt.eval()
        gpt_kv.eval()
        with torch.no_grad():
            logits_ref, _ = gpt(idx)
            logits_kv, _ = gpt_kv(idx)
        assert torch.allclose(logits_ref, logits_kv, atol=1e-5), \
            "GPT_kv.forward without cache must match GPT.forward"

    def test_with_cache_returns_three_values(self, gpt_cfg, idx):
        """forward with kv_cache must return a 3-tuple (logits, loss, new_kv_cache)."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        kv_cache = [None] * gpt_cfg.n_layer
        with torch.no_grad():
            result = gpt_kv(idx, kv_cache=kv_cache, compute_first=True)
        assert len(result) == 3, \
            "forward with kv_cache must return (logits, loss, new_kv_cache)"

    def test_new_kv_cache_has_one_entry_per_layer(self, gpt_cfg, idx):
        """Returned kv_cache must have exactly n_layer entries."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        kv_cache = [None] * gpt_cfg.n_layer
        with torch.no_grad():
            _, _, new_kv = gpt_kv(idx, kv_cache=kv_cache, compute_first=True)
        assert len(new_kv) == gpt_cfg.n_layer, \
            f"Expected {gpt_cfg.n_layer} kv_cache entries (one per layer), got {len(new_kv)}"

    def test_incremental_matches_full(self, gpt_cfg, idx):
        """Incremental forward with cache must match full forward at every position."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        with torch.no_grad():
            logits_full, _ = gpt_kv(idx)
            kv_cache = [None] * gpt_cfg.n_layer
            # First token: run the full prefix, accumulate cache
            logits_t, _, kv_cache = gpt_kv(idx[:, :1], kv_cache=kv_cache, compute_first=True)
            assert torch.allclose(logits_full[:, 0, :], logits_t[:, 0, :], atol=1e-4), \
                "Mismatch at position 0"
            # Each subsequent token: pass growing prefix, use cache
            for t in range(1, idx.shape[1]):
                logits_t, _, kv_cache = gpt_kv(idx[:, :t + 1], kv_cache=kv_cache)
                assert torch.allclose(logits_full[:, t, :], logits_t[:, 0, :], atol=1e-4), \
                    f"Mismatch at position {t}"

    def test_compute_first_repopulates_existing_cache(self, gpt_cfg, idx):
        """compute_first=True must repopulate the cache from scratch even when a cache already exists."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        with torch.no_grad():
            # Build a reference cache from the full prefix
            kv_init = [None] * gpt_cfg.n_layer
            logits_ref, _, kv_ref = gpt_kv(idx, kv_cache=kv_init, compute_first=True)

            # Build a stale cache from a shorter prefix
            kv_stale = [None] * gpt_cfg.n_layer
            _, _, kv_stale = gpt_kv(idx[:, :2], kv_cache=kv_stale, compute_first=True)

            # Re-run full prefix with compute_first=True on the stale cache
            logits_repop, _, kv_repop = gpt_kv(idx, kv_cache=kv_stale, compute_first=True)

        assert torch.allclose(logits_ref, logits_repop, atol=1e-5), \
            "compute_first=True with pre-populated cache must produce the same logits as a fresh run"
        for i in range(gpt_cfg.n_layer):
            assert torch.allclose(kv_ref[i][0], kv_repop[i][0], atol=1e-5), \
                f"Layer {i} k-cache mismatch after compute_first repopulation"
            assert torch.allclose(kv_ref[i][1], kv_repop[i][1], atol=1e-5), \
                f"Layer {i} v-cache mismatch after compute_first repopulation"


# ── 4. GPT_kv.generate_kv ────────────────────────────────────────────────────

class TestGPTKVGenerate:

    def test_output_length(self, gpt_cfg, idx):
        """generate_kv must append exactly max_new_tokens new tokens."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        n_new = 5
        with torch.no_grad():
            out = gpt_kv.generate_kv(idx, n_new, do_sample=False)
        assert out.shape == (idx.shape[0], idx.shape[1] + n_new), \
            f"Expected shape {(idx.shape[0], idx.shape[1] + n_new)}, got {out.shape}"

    def test_prefix_preserved(self, gpt_cfg, idx):
        """generate_kv must leave the input prefix unchanged."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        with torch.no_grad():
            out = gpt_kv.generate_kv(idx, 4, do_sample=False)
        assert torch.equal(out[:, :idx.shape[1]], idx), \
            "generate_kv must not modify the input prefix"

    def test_matches_generate_greedy(self, gpt_cfg, idx):
        """generate_kv must produce identical tokens to generate (greedy decoding)."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        n_new = 8
        with torch.no_grad():
            out_ref = gpt_kv.generate(idx, n_new, do_sample=False)
            out_kv = gpt_kv.generate_kv(idx, n_new, do_sample=False)
        assert torch.equal(out_ref, out_kv), \
            "generate_kv must produce the same tokens as generate (greedy)"

    def test_matches_generate_with_temperature(self, gpt_cfg, idx):
        """generate_kv with temperature != 1.0 must match generate (greedy)."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        n_new = 6
        with torch.no_grad():
            out_ref = gpt_kv.generate(idx, n_new, temperature=0.5, do_sample=False)
            out_kv = gpt_kv.generate_kv(idx, n_new, temperature=0.5, do_sample=False)
        assert torch.equal(out_ref, out_kv), \
            "generate_kv must match generate with temperature != 1.0"

    def test_matches_generate_with_top_k(self, gpt_cfg, idx):
        """generate_kv with top_k must match generate (greedy)."""
        gpt_kv = GPT_kv(gpt_cfg)
        gpt_kv.eval()
        n_new = 6
        with torch.no_grad():
            out_ref = gpt_kv.generate(idx, n_new, do_sample=False, top_k=3)
            out_kv = gpt_kv.generate_kv(idx, n_new, do_sample=False, top_k=3)
        assert torch.equal(out_ref, out_kv), \
            "generate_kv must match generate with top_k"
