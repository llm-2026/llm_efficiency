import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.model import CausalSelfAttention, Block, GPT


class LoRALinear(nn.Linear):

    def __init__(
        self,
        # nn.Linear parameters
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        # LoRA parameters
        lora_rank: int = 0,
        lora_alpha: float = 0.0,
    ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        # LoRA stuff
        self.lora_rank = lora_rank
        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_scaling = lora_alpha / lora_rank
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, self.in_features), device=device, dtype=dtype)
            )
            self.lora_B = nn.Parameter(
                torch.empty((self.out_features, lora_rank), device=device, dtype=dtype)
            )

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters_lora()

    def reset_parameters_lora(self) -> None:
        ###
        # your code here
        ###
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = nn.Linear.forward(self, input)
        ###
        # your code here
        ###
        return x

    def train(self, mode: bool = True) -> "LoRALinear":
        nn.Linear.train(self, mode)
        ###
        # your code here
        ###
        return self

    def eval(self) -> "LoRALinear":
        nn.Linear.eval(self)
        ###
        # your code here
        ###
        return self


def get_lora_model(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


class CausalSelfAttention_LoRA(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        # minor modifications
        self.c_attn = LoRALinear(
            in_features=config.n_embd,
            out_features=3 * config.n_embd,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )
        # output projection
        self.c_proj = LoRALinear(
            in_features=config.n_embd,
            out_features=config.n_embd,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )


class Block_LoRA(Block):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__(config)
        # minor modification
        self.attn = CausalSelfAttention_LoRA(config)


class GPT_LoRA(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block_LoRA(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.config = config
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def configure_optimizers(self, train_config):
        if self.config.lora_rank > 0:
            ###
            # your code here
            ###
            pass
        else:
            return super().configure_optimizers(train_config)


if __name__ == "__main__":
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

    torch.manual_seed(0)

    # ================================================================
    # Step 1: LoRALinear — initialization
    #   lora_B must be zeros, lora_A must be non-zero (Kaiming),
    #   shapes and scaling must be correct.
    # ================================================================
    section("Step 1: LoRALinear — initialization")
    try:
        ln = LoRALinear(in_features=3, out_features=4, lora_rank=8, lora_alpha=32)
        check("lora_A is non-zero", not torch.all(ln.lora_A == 0).item())
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 2: LoRALinear — forward
    #   With non-zero A and B, output must equal Wx + b + scaling * B(Ax).
    # ================================================================
    section("Step 2: LoRALinear — forward")
    try:
        bs = 5
        x = torch.randn((bs, 3))

        # Set A and B to non-zero and verify LoRA term is applied
        torch.manual_seed(7)
        ln.lora_A.data = torch.randn_like(ln.lora_A)
        ln.lora_B.data = torch.randn_like(ln.lora_B)
        with torch.no_grad():
            y_lora = ln(x)
            y_linear = x @ ln.weight.T + ln.bias
            y_expected = y_linear + ln.lora_scaling * F.linear(
                F.linear(x, ln.lora_A), ln.lora_B
            )
        check(
            "with non-zero A and B: output equals Wx + b + scaling * B(Ax)",
            torch.allclose(y_lora, y_expected, atol=1e-5),
        )
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 3: LoRALinear — eval (weight merging)
    #   eval() must merge LoRA weights: W_merged = W + scaling * B @ A.
    #   Merged forward must match unmerged forward.
    # ================================================================
    section("Step 3: LoRALinear — eval (weight merging)")
    try:
        ln.train()
        ln.eval()
        check("has_weights_merged is True after eval()", ln.has_weights_merged is True)
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ================================================================
    # Step 4: GPT_LoRA — configure_optimizers
    #   Must return an AdamW optimizer without raising.
    # ================================================================
    section("Step 4: GPT_LoRA — configure_optimizers")
    try:
        model_config = GPT.get_default_config()
        model_config.model_type = "gpt-nano"
        model_config.vocab_size = 5
        model_config.block_size = 12
        model_config.lora_rank = 4
        model_config.lora_alpha = 8.0
        model = GPT_LoRA(model_config)
        get_lora_model(model)

        @dataclass
        class TrainConfig:
            learning_rate: float = 1e-3
            betas: tuple = (0.9, 0.95)
            weight_decay: float = 0.1

        model.train()
        opt = model.configure_optimizers(TrainConfig())
        check("returns an AdamW optimizer", isinstance(opt, torch.optim.AdamW))
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
