import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.model import GPT, NewGELU


# source: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


@dataclass
class Config:
    n_head: int = 3
    n_embd: int = 15
    block_size: int = 11
    # dropout hyperparameters
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


class CausalSelfAttention_kv(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

    def forward(self, x, kv_cache=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        ###
        # your code here
        ###

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, kv_cache


class Block_kv(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention_kv(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x, kv_cache=None):
        ###
        # your code here
        ###
        pass


class GPT_kv(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block_kv(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.n_layer = config.n_layer
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def forward(self, idx, targets=None, kv_cache=None, compute_first=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        ###
        # your code here
        ###
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        if kv_cache is None:
            return logits, loss
        else:
            return logits, loss, new_kv_cache

    @torch.no_grad()
    def generate_kv(
        self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        ###
        # your code here
        ###
        pass


if __name__ == "__main__":
    from mingpt.model import Block

    total_pass = 0
    total_fail = 0

    def check(name, condition):
        global total_pass, total_fail
        if condition:
            print(f"  [PASS] {name}")
            total_pass += 1
        else:
            print(f"  [FAIL] {name}")
            total_fail += 1

    def section(title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    config = Config()
    bs = 6
    torch.manual_seed(0)
    x = torch.randn(bs, config.block_size, config.n_embd)

    # ================================================================
    # Step 1: CausalSelfAttention_kv — forward without kv_cache
    #   Should return (output, [k, v]) matching reference CausalSelfAttention.
    # ================================================================
    section("Step 1: CausalSelfAttention_kv — no kv_cache")
    try:
        csa_kv = CausalSelfAttention_kv(config)
        csa_ref = CausalSelfAttention(config)
        csa_ref.load_state_dict(csa_kv.state_dict())
        csa_ref.eval()
        csa_kv.eval()
        with torch.no_grad():
            out_ref = csa_ref(x)
            out, kv = csa_kv(x)
        check("returns (output, kv_cache) tuple",
              isinstance(kv, (list, tuple)) and len(kv) == 2)
        check("output shape is (B, T, C)",
              out.shape == x.shape)
        check("kv cache shapes are (B, T, C)",
              kv[0].shape == x.shape and kv[1].shape == x.shape)
        check("output matches reference CausalSelfAttention",
              torch.allclose(out, out_ref, atol=1e-5))
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 2: CausalSelfAttention_kv — forward with kv_cache
    #   Pass cached k,v for earlier tokens + new token(s) as input.
    #   Output for new tokens should match the full-sequence forward.
    # ================================================================
    section("Step 2: CausalSelfAttention_kv — with kv_cache")
    try:
        with torch.no_grad():
            # Single new token with T-1 cached tokens
            out_last, _ = csa_kv(
                x[:, [-1], :],
                kv_cache=[kv[0][:, :-1, :], kv[1][:, :-1, :]],
            )
        check("1 new token + (T-1) cached: matches full forward",
              torch.allclose(out[:, -1, :], out_last[:, 0, :], atol=1e-4))

        # Token-by-token incremental decoding
        with torch.no_grad():
            out_t, kv_inc = csa_kv(x[:, :1, :])
            ok = torch.allclose(out[:, 0, :], out_t[:, 0, :], atol=1e-4)
            for t in range(1, config.block_size):
                out_t, kv_inc = csa_kv(x[:, [t], :], kv_cache=kv_inc)
                if not torch.allclose(out[:, t, :], out_t[:, 0, :], atol=1e-4):
                    ok = False
                    print(f"    mismatch at position {t}")
        check("token-by-token incremental matches full forward", ok)
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 3: Block_kv — forward without and with kv_cache
    #   Without cache, should match mingpt Block. With cache, last token
    #   output should match the full-sequence forward.
    # ================================================================
    section("Step 3: Block_kv")
    try:
        block_ref = Block(config)
        bkv = Block_kv(config)
        bkv.load_state_dict(block_ref.state_dict())
        block_ref.eval()
        bkv.eval()
        with torch.no_grad():
            out_ref = block_ref(x)
            out_block, kv_block = bkv(x)
        check("returns (output, kv_cache) tuple",
              isinstance(kv_block, (list, tuple)) and len(kv_block) == 2)
        check("output shape matches input",
              out_block.shape == x.shape)
        check("output matches reference Block (no cache)",
              torch.allclose(out_ref, out_block, atol=1e-5))

        # With cache: single new token
        with torch.no_grad():
            out_last, _ = bkv(
                x[:, [-1], :],
                kv_cache=[kv_block[0][:, :-1, :], kv_block[1][:, :-1, :]],
            )
        check("1 new token + cached: matches last position",
              torch.allclose(out_block[:, -1, :], out_last[:, 0, :], atol=1e-4))

        # Token-by-token incremental
        with torch.no_grad():
            out_t, kv_inc = bkv(x[:, :1, :])
            ok = torch.allclose(out_block[:, 0, :], out_t[:, 0, :], atol=1e-4)
            for t in range(1, config.block_size):
                out_t, kv_inc = bkv(x[:, [t], :], kv_cache=kv_inc)
                if not torch.allclose(out_block[:, t, :], out_t[:, 0, :], atol=1e-4):
                    ok = False
                    print(f"    mismatch at position {t}")
        check("token-by-token incremental matches full forward", ok)
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 4: GPT_kv — forward without kv_cache
    #   Should produce logits identical to standard GPT forward.
    # ================================================================
    section("Step 4: GPT_kv — forward without kv_cache")
    try:
        model_config = GPT.get_default_config()
        model_config.model_type = "gpt-nano"
        model_config.vocab_size = 3
        model_config.block_size = 100
        model = GPT_kv(model_config)
        model.eval()

        inp = torch.tensor([[0, 0, 2, 1, 0, 1, 2]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(inp)
        check("forward produces logits with shape (1, 7, 3)",
              logits is not None and logits.shape == (1, 7, 3))
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 5: GPT_kv — incremental forward with kv_cache
    #   Process tokens incrementally (growing prefix). Each step should
    #   produce logits matching the full forward at that position.
    # ================================================================
    section("Step 5: GPT_kv — incremental forward with kv_cache")
    try:
        with torch.no_grad():
            kv_cache = [None] * model_config.n_layer
            logits_kv, _, kv_cache = model(inp[:, [0]], kv_cache=kv_cache)
        check("token 0: logits match full forward",
              torch.allclose(logits[:, 0, :], logits_kv[:, 0, :], atol=1e-4))

        with torch.no_grad():
            logits_kv, _, kv_cache = model(inp[:, 0:2], kv_cache=kv_cache)
        check("tokens 0:2 with cache: position 1 matches",
              torch.allclose(logits[:, 1, :], logits_kv[:, 0, :], atol=1e-4))

        with torch.no_grad():
            logits_kv, _, kv_cache = model(inp[:, 0:3], kv_cache=kv_cache)
        check("tokens 0:3 with cache: position 2 matches",
              torch.allclose(logits[:, 2, :], logits_kv[:, 0, :], atol=1e-4))
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 6: GPT_kv — compute_first mode
    #   Prefill cache with compute_first=True, then continue incrementally.
    # ================================================================
    section("Step 6: GPT_kv — compute_first mode")
    try:
        with torch.no_grad():
            kv_fresh = [None] * model_config.n_layer
            _, _, kv1 = model(
                inp[:, 0:2], kv_cache=kv_fresh, compute_first=True
            )
            logits_kv2, _, _ = model(inp[:, 0:3], kv_cache=kv1)
        check("prefill 2 tokens then continue: matches step 5",
              torch.allclose(logits_kv2[:, 0, :], logits_kv[:, 0, :], atol=1e-4))
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 7: GPT_kv — generate_kv
    #   Autoregressive generation with kv_cache should match generate().
    # ================================================================
    section("Step 7: GPT_kv — generate_kv")
    try:
        n_new = 5
        with torch.no_grad():
            out_ref = model.generate(inp, n_new, do_sample=False)
            out_kv = model.generate_kv(inp, n_new, do_sample=False)
        check(f"output length is input + {n_new}",
              out_kv.shape == (inp.shape[0], inp.shape[1] + n_new))
        check("prefix is preserved",
              torch.equal(out_kv[:, :inp.shape[1]], inp))
        check("greedy output matches generate()",
              torch.equal(out_ref, out_kv))
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Summary
    # ================================================================
    total = total_pass + total_fail
    print(f"\n{'=' * 60}")
    if total == 0:
        print(f"  No checks ran — all steps raised errors.")
    elif total_fail == 0:
        print(f"  All {total_pass} checks passed!")
        print(f"  Now run the tests in the tests/ folder to validate.")
    else:
        print(f"  {total_pass}/{total} checks passed, {total_fail} failed")
    print(f"{'=' * 60}")
