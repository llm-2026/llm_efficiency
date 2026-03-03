# LoRA

The goal of this part is to adapt the code of [minGPT](https://github.com/karpathy/minGPT/) from [Karpathy](https://karpathy.ai/) to incorporate Low-Rank Adaptation (LoRA) for fine-tuning. 

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*D_i25E9dTd_5HMa45zITSg.png)

This [blog post](https://r4j4n.github.io/blogs/posts/lora/) by [Rajan Ghimire](https://r4j4n.github.io/blogs/about/) is a concise introduction to LoRA. The original paper is [Hu et al., 2021](https://arxiv.org/abs/2106.09685).

It is perfectly fine to solve the exercises in the [Jupyter notebook](practicals/Lora_empty.ipynb) and then copy-paste your code in the python files `lora.py` and `demo_sort_lora.py`.

## Background

Standard fine-tuning updates all $W \in \mathbb{R}^{d \times k}$ parameters of a pre-trained model, which is expensive when the model is large. LoRA freezes the pre-trained weights and injects a low-rank decomposition into each target layer:

$$h = Wx + \frac{\alpha}{r} BAx$$

where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, $r \ll \min(d, k)$, and $\alpha$ is a scaling constant. Only $A$ and $B$ are trained. At initialization, $B = 0$ so the adapter has no effect and training starts from the pre-trained checkpoint.

At inference time the correction $\frac{\alpha}{r} BA$ can be **merged** into $W$ once, yielding $W' = W + \frac{\alpha}{r} BA$, so that LoRA adds zero overhead compared to the base model.

## Exercise 1 — Building `LoRALinear`

`LoRALinear` subclasses `nn.Linear` and adds the LoRA adapter. All four methods below must be completed in `lora.py`.


### `LoRALinear.reset_parameters_lora`

Initialise the LoRA matrices:
- `lora_A`: Kaiming uniform with `a=sqrt(5)` (identical to `nn.Linear`'s weight init).
- `lora_B`: zeros, so the adapter output is $0$ at the start of training.

Also call `nn.Linear.reset_parameters(self)` to reset the base weight and bias.

### `LoRALinear.forward`

The standard linear pass `x = Wx + b` is computed by `nn.Linear.forward`. When weights have **not** been merged and `lora_rank > 0`, add the LoRA correction:

$$h = Wx + b + \frac{\alpha}{r} \cdot B(Ax)$$

Use `F.linear` for both the $A$ and $B$ applications. When `has_weights_merged` is `True` the correction is already baked into `W`, so nothing extra is needed.


### `LoRALinear.train`

When switching back to training mode (`mode=True`), **de-merge** if the weights were previously merged:

$$W \leftarrow W - \frac{\alpha}{r} \cdot BA$$


### `LoRALinear.eval`

When switching to eval mode, **merge** the LoRA correction into `W` so that inference requires no extra computation:

$$W \leftarrow W + \frac{\alpha}{r} \cdot BA$$


**Test:** Call `.eval()` then `.train()` and verify the output is unchanged — merging then de-merging must be an exact round-trip.


## Exercise 2 — Integrating LoRA into minGPT

### `CausalSelfAttention_LoRA.__init__`

Call `super().__init__(config)` (which builds the standard attention layer), then **replace** `self.c_attn` and `self.c_proj` with `LoRALinear` instances, passing `lora_rank` and `lora_alpha` from `config`:

```python
self.c_attn = LoRALinear(
    in_features=config.n_embd,
    out_features=3 * config.n_embd,
    lora_rank=config.lora_rank,
    lora_alpha=config.lora_alpha,
)
self.c_proj = LoRALinear(
    in_features=config.n_embd,
    out_features=config.n_embd,
    lora_rank=config.lora_rank,
    lora_alpha=config.lora_alpha,
)
```

`Block_LoRA` and `GPT_LoRA.__init__` are already provided: they substitute `Block_LoRA` (which uses `CausalSelfAttention_LoRA`) wherever the base classes used `Block`.

### `GPT_LoRA.configure_optimizers`

When `lora_rank > 0`, skip the full decay/no-decay parameter grouping from the base `GPT` class and simply return a `AdamW` optimizer. Because `get_lora_model` has already frozen all non-LoRA parameters, only the LoRA matrices will receive gradient updates.

## Exercise 3 — Learning to sort

We use the sorting task from Karpathy's [demo](https://github.com/karpathy/minGPT/blob/master/demo.ipynb) as a testbed in the `demo_sort_lora.py` file.

### Pre-training (code provided)

Train `GPT_LoRA` (with `lora_rank=8`, `lora_alpha=32`) on `SortDataset(split='train', length=6)` for 1000 iterations at learning rate `5e-4`. Evaluate on both the train and test splits using greedy decoding and report accuracy (fraction of sequences sorted correctly).

### Distribution shift (code provided)

Evaluate the pre-trained model — without any fine-tuning — on `SortDataset(length=10)`. The model was trained on length-6 sequences, so performance on length-10 will be lower. 

### LoRA fine-tuning (to be completed)

Call `get_lora_model(model)` to freeze the pre-trained weights and then fine-tune only the LoRA parameters on `SortDataset(length=10)` for 2000 iterations. Re-evaluate on both splits and report the recovered accuracy.

