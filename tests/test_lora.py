"""
Tests for lora/lora.py — used for automated grading of student implementations.

Graded components:
  1. LoRALinear.reset_parameters_lora  — initialization of LoRA matrices
  2. LoRALinear.forward                — forward pass with LoRA term
  3. LoRALinear.train                  — weight de-merging on switch to train mode
  4. LoRALinear.eval                   — weight merging on switch to eval mode
  5. GPT_LoRA.configure_optimizers     — simplified optimizer for LoRA
  6. get_lora_model                    — freezing non-LoRA parameters
"""

import pytest
import torch
import torch.nn as nn

from mingpt.model import GPT
from lora.lora import LoRALinear, get_lora_model, GPT_LoRA, CausalSelfAttention_LoRA


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def lora_linear():
    """LoRALinear(3→4) with rank 2, alpha 4.0; lora_scaling = 4/2 = 2."""
    torch.manual_seed(0)
    return LoRALinear(in_features=3, out_features=4, lora_rank=2, lora_alpha=4.0)


@pytest.fixture
def x():
    """Random input of shape (5, 3)."""
    torch.manual_seed(1)
    return torch.randn(5, 3)


@pytest.fixture
def gpt_lora_cfg():
    """Small GPT_LoRA config (no dropout to keep tests deterministic)."""
    c = GPT.get_default_config()
    c.model_type = None
    c.n_layer = 2
    c.n_head = 2
    c.n_embd = 16
    c.vocab_size = 5
    c.block_size = 12
    c.lora_rank = 4
    c.lora_alpha = 8.0
    c.embd_pdrop = 0.0
    c.resid_pdrop = 0.0
    c.attn_pdrop = 0.0
    return c


@pytest.fixture
def train_cfg():
    """Minimal training config compatible with configure_optimizers."""
    from dataclasses import dataclass

    @dataclass
    class TC:
        learning_rate: float = 1e-3
        betas: tuple = (0.9, 0.95)
        weight_decay: float = 0.1

    return TC()


# ── 1. LoRALinear.reset_parameters_lora ────────────────────────────────────────

class TestResetParametersLoRA:

    def test_lora_B_initialized_to_zeros(self, lora_linear):
        """lora_B must be all-zero at initialization."""
        assert torch.all(lora_linear.lora_B == 0), \
            "lora_B must be initialized to zeros"

    def test_lora_A_nonzero(self, lora_linear):
        """lora_A must be non-zero (Kaiming uniform)."""
        assert not torch.all(lora_linear.lora_A == 0), \
            "lora_A must be non-zero after initialization (Kaiming uniform)"

    def test_base_weight_nonzero(self, lora_linear):
        """Base weight must be (re-)initialized as non-zero."""
        assert not torch.all(lora_linear.weight == 0), \
            "Base weight must be non-zero after reset_parameters_lora"

    def test_no_lora_attrs_when_rank_zero(self):
        """With lora_rank=0, the module must not have lora_A / lora_B."""
        ln = LoRALinear(in_features=4, out_features=8, lora_rank=0)
        assert not hasattr(ln, "lora_A"), "lora_A should not exist when lora_rank=0"
        assert not hasattr(ln, "lora_B"), "lora_B should not exist when lora_rank=0"

    def test_lora_A_shape(self, lora_linear):
        """lora_A must have shape (lora_rank, in_features)."""
        assert lora_linear.lora_A.shape == (2, 3), \
            f"lora_A shape: expected (2, 3), got {lora_linear.lora_A.shape}"

    def test_lora_B_shape(self, lora_linear):
        """lora_B must have shape (out_features, lora_rank)."""
        assert lora_linear.lora_B.shape == (4, 2), \
            f"lora_B shape: expected (4, 2), got {lora_linear.lora_B.shape}"

    def test_lora_scaling_value(self, lora_linear):
        """lora_scaling must equal lora_alpha / lora_rank."""
        # lora_alpha=4.0, lora_rank=2 → scaling = 2.0
        assert lora_linear.lora_scaling == pytest.approx(2.0), \
            f"lora_scaling: expected 2.0, got {lora_linear.lora_scaling}"

    def test_lora_params_require_grad_false_at_init(self, lora_linear):
        """lora_A and lora_B must have requires_grad=False at initialization."""
        assert lora_linear.lora_A.requires_grad is False, \
            "lora_A must have requires_grad=False at initialization"
        assert lora_linear.lora_B.requires_grad is False, \
            "lora_B must have requires_grad=False at initialization"


# ── 2. LoRALinear.forward ───────────────────────────────────────────────────────

class TestLoRALinearForward:

    def test_output_shape(self, lora_linear, x):
        """Output shape must be (B, out_features)."""
        with torch.no_grad():
            y = lora_linear(x)
        assert y.shape == (x.shape[0], lora_linear.out_features), \
            f"Expected shape {(x.shape[0], lora_linear.out_features)}, got {y.shape}"

    def test_initial_output_matches_linear(self, lora_linear, x):
        """At init lora_B=0, so LoRA term is zero; output must equal Wx+b."""
        lora_linear.train()
        with torch.no_grad():
            y_lora = lora_linear(x)
            y_lin = nn.functional.linear(x, lora_linear.weight, lora_linear.bias)
        assert torch.allclose(y_lora, y_lin, atol=1e-6), \
            "With lora_B=0, forward must equal standard linear output"

    def test_lora_term_applied_when_B_nonzero(self, lora_linear, x):
        """With non-zero lora_B, forward must add scaling * B(Ax) to Wx+b."""
        lora_linear.train()
        torch.manual_seed(7)
        lora_linear.lora_B.data = torch.randn_like(lora_linear.lora_B)
        with torch.no_grad():
            y_lora = lora_linear(x)
            y_lin = nn.functional.linear(x, lora_linear.weight, lora_linear.bias)
            expected_delta = lora_linear.lora_scaling * nn.functional.linear(
                nn.functional.linear(x, lora_linear.lora_A),
                lora_linear.lora_B,
            )
        assert torch.allclose(y_lora, y_lin + expected_delta, atol=1e-5), \
            "forward must add scaling * B(Ax) to Wx+b"

    def test_no_lora_term_when_rank_zero(self, x):
        """With lora_rank=0, LoRALinear must behave identically to nn.Linear."""
        torch.manual_seed(0)
        ln = LoRALinear(in_features=3, out_features=4, lora_rank=0)
        ref = nn.Linear(in_features=3, out_features=4)
        ref.load_state_dict(ln.state_dict())
        # Stay in train mode: lora_rank=0 so no LoRA term is added
        with torch.no_grad():
            assert torch.allclose(ln(x), ref(x), atol=1e-6), \
                "With lora_rank=0, LoRALinear must behave like nn.Linear"


# ── 3. LoRALinear.train ─────────────────────────────────────────────────────────

class TestLoRALinearTrain:

    def test_has_weights_merged_false_after_train(self, lora_linear):
        """has_weights_merged must be False after calling train()."""
        lora_linear.train()
        assert lora_linear.has_weights_merged is False, \
            "has_weights_merged must be False in train mode"

    def test_weight_restored_after_eval_then_train(self, lora_linear):
        """eval() then train() must restore weight to its original value."""
        torch.manual_seed(7)
        lora_linear.lora_B.data = torch.randn_like(lora_linear.lora_B)
        lora_linear.train()
        original_weight = lora_linear.weight.data.clone()
        lora_linear.eval()   # merges: W += scaling * B @ A
        lora_linear.train()  # de-merges: W -= scaling * B @ A
        assert torch.allclose(lora_linear.weight.data, original_weight, atol=1e-6), \
            "Weight must be restored to its original value after eval() then train()"

    def test_train_idempotent_when_already_unmerged(self, lora_linear):
        """Calling train() twice must not subtract the LoRA term a second time."""
        torch.manual_seed(7)
        lora_linear.lora_B.data = torch.randn_like(lora_linear.lora_B)
        lora_linear.train()
        w_after_first = lora_linear.weight.data.clone()
        lora_linear.train()  # second call — must be a no-op
        assert torch.allclose(lora_linear.weight.data, w_after_first, atol=1e-6), \
            "Calling train() twice must not modify the weight a second time"


# ── 4. LoRALinear.eval ──────────────────────────────────────────────────────────

class TestLoRALinearEval:

    def test_has_weights_merged_true_after_eval(self, lora_linear):
        """has_weights_merged must be True after calling eval()."""
        lora_linear.eval()
        assert lora_linear.has_weights_merged is True, \
            "has_weights_merged must be True in eval mode"

    def test_weight_contains_lora_after_eval(self, lora_linear):
        """After eval(), weight must equal W_orig + scaling * B @ A."""
        torch.manual_seed(7)
        lora_linear.lora_B.data = torch.randn_like(lora_linear.lora_B)
        lora_linear.train()
        w_orig = lora_linear.weight.data.clone()
        expected_delta = lora_linear.lora_scaling * lora_linear.lora_B @ lora_linear.lora_A
        lora_linear.eval()
        assert torch.allclose(lora_linear.weight.data, w_orig + expected_delta, atol=1e-6), \
            "After eval(), weight must equal W_orig + scaling * B @ A"

    def test_eval_output_matches_train_output(self, lora_linear, x):
        """Forward in eval mode (merged) must match forward in train mode (unmerged)."""
        torch.manual_seed(7)
        lora_linear.lora_B.data = torch.randn_like(lora_linear.lora_B)
        lora_linear.train()
        with torch.no_grad():
            y_train = lora_linear(x)
        lora_linear.eval()
        with torch.no_grad():
            y_eval = lora_linear(x)
        assert torch.allclose(y_train, y_eval, atol=1e-5), \
            "Forward in eval mode must produce the same result as in train mode"

    def test_eval_idempotent_when_already_merged(self, lora_linear):
        """Calling eval() twice must not add the LoRA term a second time."""
        torch.manual_seed(7)
        lora_linear.lora_B.data = torch.randn_like(lora_linear.lora_B)
        lora_linear.eval()
        w_after_first = lora_linear.weight.data.clone()
        lora_linear.eval()  # second call — must be a no-op
        assert torch.allclose(lora_linear.weight.data, w_after_first, atol=1e-6), \
            "Calling eval() twice must not add the LoRA term a second time"


# ── 5. GPT_LoRA.configure_optimizers ───────────────────────────────────────────

class TestGPTLoRAConfigureOptimizers:

    def test_does_not_raise(self, gpt_lora_cfg, train_cfg):
        """configure_optimizers must not raise (base class would AssertionError on LoRA params)."""
        model = GPT_LoRA(gpt_lora_cfg)
        model.train()
        try:
            model.configure_optimizers(train_cfg)
        except Exception as e:
            pytest.fail(f"configure_optimizers raised an exception: {e}")

    def test_returns_adamw(self, gpt_lora_cfg, train_cfg):
        """configure_optimizers must return a torch.optim.AdamW instance."""
        model = GPT_LoRA(gpt_lora_cfg)
        model.train()
        opt = model.configure_optimizers(train_cfg)
        assert isinstance(opt, torch.optim.AdamW), \
            f"Expected AdamW optimizer, got {type(opt)}"

    def test_all_params_covered(self, gpt_lora_cfg, train_cfg):
        """All model parameters must be included in the optimizer."""
        model = GPT_LoRA(gpt_lora_cfg)
        model.train()
        opt = model.configure_optimizers(train_cfg)
        opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
        model_param_ids = {id(p) for p in model.parameters()}
        assert model_param_ids == opt_param_ids, \
            "All model parameters must be covered by the optimizer"

    def test_rank_zero_falls_back_to_base(self, train_cfg):
        """With lora_rank=0, configure_optimizers must fall back to base GPT behaviour."""
        c = GPT.get_default_config()
        c.model_type = None
        c.n_layer = 2
        c.n_head = 2
        c.n_embd = 16
        c.vocab_size = 5
        c.block_size = 12
        c.lora_rank = 0
        c.lora_alpha = 0.0
        c.embd_pdrop = 0.0
        c.resid_pdrop = 0.0
        c.attn_pdrop = 0.0
        model = GPT_LoRA(c)
        model.train()
        opt = model.configure_optimizers(train_cfg)
        assert opt is not None, \
            "configure_optimizers must return an optimizer even when lora_rank=0"
        assert isinstance(opt, torch.optim.AdamW), \
            f"Expected AdamW optimizer for lora_rank=0, got {type(opt)}"


# ── 6. CausalSelfAttention_LoRA structure ──────────────────────────────────────

class TestCausalSelfAttentionLoRA:

    def test_c_attn_is_lora_linear(self, gpt_lora_cfg):
        """c_attn must be a LoRALinear instance."""
        csa = CausalSelfAttention_LoRA(gpt_lora_cfg)
        assert isinstance(csa.c_attn, LoRALinear), \
            f"c_attn must be LoRALinear, got {type(csa.c_attn)}"

    def test_c_proj_is_lora_linear(self, gpt_lora_cfg):
        """c_proj must be a LoRALinear instance."""
        csa = CausalSelfAttention_LoRA(gpt_lora_cfg)
        assert isinstance(csa.c_proj, LoRALinear), \
            f"c_proj must be LoRALinear, got {type(csa.c_proj)}"

    def test_c_attn_lora_rank_and_alpha(self, gpt_lora_cfg):
        """c_attn must use the correct lora_rank and lora_alpha from config."""
        csa = CausalSelfAttention_LoRA(gpt_lora_cfg)
        assert csa.c_attn.lora_rank == gpt_lora_cfg.lora_rank, \
            f"c_attn.lora_rank: expected {gpt_lora_cfg.lora_rank}, got {csa.c_attn.lora_rank}"
        expected_scaling = gpt_lora_cfg.lora_alpha / gpt_lora_cfg.lora_rank
        assert csa.c_attn.lora_scaling == pytest.approx(expected_scaling), \
            f"c_attn.lora_scaling: expected {expected_scaling}, got {csa.c_attn.lora_scaling}"

    def test_c_proj_lora_rank_and_alpha(self, gpt_lora_cfg):
        """c_proj must use the correct lora_rank and lora_alpha from config."""
        csa = CausalSelfAttention_LoRA(gpt_lora_cfg)
        assert csa.c_proj.lora_rank == gpt_lora_cfg.lora_rank, \
            f"c_proj.lora_rank: expected {gpt_lora_cfg.lora_rank}, got {csa.c_proj.lora_rank}"
        expected_scaling = gpt_lora_cfg.lora_alpha / gpt_lora_cfg.lora_rank
        assert csa.c_proj.lora_scaling == pytest.approx(expected_scaling), \
            f"c_proj.lora_scaling: expected {expected_scaling}, got {csa.c_proj.lora_scaling}"


# ── 7. GPT_LoRA forward pass ─────────────────────────────────────────────────

class TestGPTLoRAForward:

    def test_forward_produces_logits(self, gpt_lora_cfg):
        """GPT_LoRA forward must return logits of correct shape."""
        model = GPT_LoRA(gpt_lora_cfg)
        model.train()
        idx = torch.randint(0, gpt_lora_cfg.vocab_size, (2, 6))
        logits, loss = model(idx)
        assert logits.shape == (2, 6, gpt_lora_cfg.vocab_size), \
            f"Expected logits shape (2, 6, {gpt_lora_cfg.vocab_size}), got {logits.shape}"
        assert loss is None, "Loss must be None when no targets are provided"

    def test_forward_computes_loss(self, gpt_lora_cfg):
        """GPT_LoRA forward with targets must return a scalar loss."""
        model = GPT_LoRA(gpt_lora_cfg)
        model.train()
        idx = torch.randint(0, gpt_lora_cfg.vocab_size, (2, 6))
        targets = torch.randint(0, gpt_lora_cfg.vocab_size, (2, 6))
        logits, loss = model(idx, targets=targets)
        assert loss is not None, "Loss must not be None when targets are provided"
        assert loss.dim() == 0, "Loss must be a scalar"

    def test_forward_matches_gpt_at_init(self, gpt_lora_cfg):
        """At init (lora_B=0), GPT_LoRA forward must match GPT forward (same weights)."""
        gpt_cfg_base = GPT.get_default_config()
        gpt_cfg_base.model_type = None
        gpt_cfg_base.n_layer = gpt_lora_cfg.n_layer
        gpt_cfg_base.n_head = gpt_lora_cfg.n_head
        gpt_cfg_base.n_embd = gpt_lora_cfg.n_embd
        gpt_cfg_base.vocab_size = gpt_lora_cfg.vocab_size
        gpt_cfg_base.block_size = gpt_lora_cfg.block_size
        gpt_cfg_base.embd_pdrop = 0.0
        gpt_cfg_base.resid_pdrop = 0.0
        gpt_cfg_base.attn_pdrop = 0.0

        gpt_base = GPT(gpt_cfg_base)
        gpt_lora = GPT_LoRA(gpt_lora_cfg)

        # Copy base weights into LoRA model (LoRA params are extra, load with strict=False)
        gpt_lora.load_state_dict(gpt_base.state_dict(), strict=False)
        gpt_base.eval()
        gpt_lora.eval()

        idx = torch.randint(0, gpt_lora_cfg.vocab_size, (2, 6))
        with torch.no_grad():
            logits_base, _ = gpt_base(idx)
            logits_lora, _ = gpt_lora(idx)
        assert torch.allclose(logits_base, logits_lora, atol=1e-5), \
            "At init (lora_B=0), GPT_LoRA forward must match GPT forward"


# ── 8. get_lora_model ──────────────────────────────────────────────────────────

class TestGetLoRAModel:

    def test_lora_params_require_grad(self, gpt_lora_cfg):
        """After get_lora_model, LoRA parameters must require gradients."""
        model = GPT_LoRA(gpt_lora_cfg)
        get_lora_model(model)
        for name, param in model.named_parameters():
            if "lora" in name:
                assert param.requires_grad, \
                    f"LoRA parameter '{name}' must require gradients"

    def test_base_params_frozen(self, gpt_lora_cfg):
        """After get_lora_model, non-LoRA parameters must be frozen."""
        model = GPT_LoRA(gpt_lora_cfg)
        get_lora_model(model)
        for name, param in model.named_parameters():
            if "lora" not in name:
                assert not param.requires_grad, \
                    f"Base parameter '{name}' must be frozen (requires_grad=False)"

    def test_only_lora_params_updated_after_step(self, gpt_lora_cfg):
        """A gradient step must update only LoRA params, leaving base weights unchanged."""
        torch.manual_seed(42)
        model = GPT_LoRA(gpt_lora_cfg)
        model.train()
        get_lora_model(model)

        base_before = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if "lora" not in name
        }

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
        )
        idx = torch.randint(0, gpt_lora_cfg.vocab_size, (2, 6))
        targets = torch.randint(0, gpt_lora_cfg.vocab_size, (2, 6))
        _, loss = model(idx, targets=targets)
        loss.backward()
        opt.step()

        for name, param in model.named_parameters():
            if "lora" not in name:
                assert torch.equal(param.data, base_before[name]), \
                    f"Base parameter '{name}' must not change during LoRA fine-tuning"
