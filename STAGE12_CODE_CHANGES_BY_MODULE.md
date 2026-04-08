# Stage I and Stage II Code Changes by Module

This file consolidates the confirmed code changes from `STAGE1_DEBUG_LOG.md` and `STAGE2_DEBUG_LOG.md`.

- The list is reorganized by module rather than by debugging order.
- Repeated descriptions across Stage I and Stage II have been deduplicated.
- Each item is tagged with the stage where it was primarily recorded.

## TrainTools

### `TrainTools/train.py` `[Stage I]`

- Corrected argument packaging for the training entry path.

Original

```python
args = argparse.Namespace({k: v for k, v in locals().items()})
```

Updated

```python
args = argparse.Namespace(**{k: v for k, v in locals().items()})
```

### `TrainTools/train_utils.py` `[Stage I]`

- Corrected the backward pass to use the tensor loss rather than a Python float.

Original

```python
loss.item().backward()
```

Updated

```python
loss.backward()
```

- Moved gradient clipping to occur before `optimizer.step()` to stabilize updates.

Original

```python
loss.backward()
optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scheduler.step()
```

Updated

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
scheduler.step()
```

## Losses

### `Losses/loss.py` `[Stage I]`

- Corrected the argument order of the first `F.nll_loss(...)` call in `qa_nll_loss(...)`.

Original

```python
return 0.5 * (F.nll_loss(y1, p1) + F.nll_loss(p2, y2))
```

Updated

```python
return 0.5 * (F.nll_loss(p1, y1) + F.nll_loss(p2, y2))
```

## Models / Activations

### `Models/Activations/relu.py` `[Stage II]`

- Corrected the ReLU clamp direction.

Original

```python
return x.clamp(max=0.0)
```

Updated

```python
return x.clamp(min=0.0)
```

### `Models/Activations/leakeyReLU.py` `[Stage II]`

- Corrected the LeakyReLU branch logic.

Original

```python
return torch.where(x < 0, x, self.negative_slope * x)
```

Updated

```python
return torch.where(x < 0, self.negative_slope * x, x)
```

## Models / Embedding and Input Routing

### `Models/qanet.py` `[Stage I]`

- Corrected context word and character embedding routing.

Original

```python
Cw, Cc = self.char_emb(Cwid), self.word_emb(Ccid)
Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
```

Updated

```python
Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
```

### `Models/embedding.py` `[Stage I]`

- Corrected the character embedding permutation before 2D convolution.

Original

```python
ch_emb = ch_emb.permute(0, 2, 1, 3)  # [B, d_char, L, char_len]
```

Updated

```python
ch_emb = ch_emb.permute(0, 3, 1, 2)  # [B, d_char, L, char_len]
```

- Corrected the highway-layer transpose so linear layers receive `[B, L, C]`.

Original

```python
x = x.transpose(0, 2)
```

Updated

```python
x = x.transpose(1, 2)
```

## Models / Convolution Blocks

### `Models/conv.py` `[Stage I]`

- Corrected width-padding tensor creation in `Conv2d.forward()`.

Original

```python
pad_w = x.new_zeros(B, C_in, H, p)
```

Updated

```python
pad_w = x.new_zeros(B, C_in, x.size(2), p)
```

- Corrected the unfold dimension in `Conv1d.forward()` so windows slide over sequence length.

Original

```python
x_unf = x.unfold(1, self.kernel_size, 1)  # [B, C_in, L_out, k]
```

Updated

```python
x_unf = x.unfold(2, self.kernel_size, 1)  # [B, C_in, L_out, k]
```

- Corrected the order of depthwise and pointwise convolutions.

Original

```python
return self.depthwise_conv(self.pointwise_conv(x))
```

Updated

```python
return self.pointwise_conv(self.depthwise_conv(x))
```

## Models / Normalizations

### `Models/Normalizations/layernorm.py` `[Stage I]`

- Corrected LayerNorm broadcasting and affine application.

Original

```python
mean = x.mean(dim=dims, keepdim=False)
var = x.var(dim=dims, keepdim=False, unbiased=False)

x_norm = (x - mean) / torch.sqrt(var + self.eps)
return x_norm * self.bias + self.weight
```

Updated

```python
mean = x.mean(dim=dims, keepdim=True)
var = x.var(dim=dims, keepdim=True, unbiased=False)

x_norm = (x - mean) / torch.sqrt(var + self.eps)
return x_norm * self.weight + self.bias
```

### `Models/Normalizations/groupnorm.py` `[Stage II]`

- Corrected the group/channel reshape order.

Original

```python
x = x.view(B, C // self.G, self.G, *spatial)
```

Updated

```python
x = x.view(B, self.G, C // self.G, *spatial)
```

## Models / Initializations

### `Models/Initializations/kaiming.py` `[Stage I]`

- Corrected the Kaiming standard deviation formula.

Original

```python
std = math.sqrt(1.0 / fan)
```

Updated

```python
std = math.sqrt(2.0 / fan)
```

### `Models/Initializations/xavier.py` `[Stage I]`

- Corrected the Xavier standard deviation formula.

Original

```python
std = gain * math.sqrt(2.0 / (fan_in * fan_out))
```

Updated

```python
std = gain * math.sqrt(2.0 / (fan_in + fan_out))
```

## Models / Attention and Encoder

### `Models/encoder.py` `[Stage I]`

- Corrected the positional-encoding frequency tensor shape.

Original

```python
).unsqueeze(0)  # [C, 1]
```

Updated

```python
).unsqueeze(1)  # [C, 1]
```

- Corrected encoder normalization indexing inside the convolution loop.

Original

```python
out = self.norms[i + 1](out)
```

Updated

```python
out = self.norms[i](out)
```

- Restored scaled dot-product attention in multi-head attention.

Original

```python
attn = torch.bmm(q, k.transpose(1, 2))
```

Updated

```python
attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
```

### `Models/encoder.py` `[Stage II]`

- Corrected the self-attention residual path so attention output is not overwritten.

Original

```python
out = self.self_att(out, mask)
out = res
out = self.drop(out)
```

Updated

```python
out = self.self_att(out, mask)
out = out + res
out = self.drop(out)
```

### `Models/qanet.py` `[Stage I]`

- Corrected the mask order passed into context-question attention.

Original

```python
X = self.cq_att(Ce, Qe, qmask, cmask)
```

Updated

```python
X = self.cq_att(Ce, Qe, cmask, qmask)
```

### `Models/attention.py` `[Stage I]`

- Corrected the batch-matrix multiplication order in `CQAttention`.

Original

```python
A = torch.bmm(Q, S1)
```

Updated

```python
A = torch.bmm(S1, Q)
```

### `Models/dropout.py` `[Stage I]`

- Corrected inverted-dropout scaling to divide by keep probability.

Original

```python
return x * mask / self.p
```

Updated

```python
return x * mask / (1.0 - self.p)
```

## Models / Output Head

### `Models/heads.py` `[Stage I]`

- Corrected pointer-head feature concatenation to use the channel dimension.

Original

```python
X1 = torch.cat([M1, M2], dim=0)  # [B, 2C, L]
```

Updated

```python
X1 = torch.cat([M1, M2], dim=1)  # [B, 2C, L]
```

## Optimizers

### `Optimizers/sgd.py` `[Stage I]`

- Corrected the sign of the weight-decay term.

Original

```python
grad = grad.add(p, alpha=-wd)
```

Updated

```python
grad = grad.add(p, alpha=wd)
```

### `Optimizers/sgd_momentum.py` `[Stage II]`

- Unified the velocity state key and corrected the momentum update rule.

Original

```python
if "velocity" not in state:
    state["vel"] = torch.zeros_like(p)

v = state["velocity"]
v.mul_(mu).sub_(grad)
```

Updated

```python
if "velocity" not in state:
    state["velocity"] = torch.zeros_like(p)

v = state["velocity"]
v.mul_(mu).add_(grad)
```

### `Optimizers/adam.py` `[Stage II]`

- Corrected weight decay, state-key access, second-moment accumulation, and bias-correction formulas.

Weight decay

Original

```python
if wd != 0.0:
    grad = grad.add(p, alpha=-wd)
```

Updated

```python
if wd != 0.0:
    grad = grad.add(p, alpha=wd)
```

State keys

Original

```python
m, v = state["m"], state["v"]
```

Updated

```python
m, v = state["exp_avg"], state["exp_avg_sq"]
```

Second moment

Original

```python
v.mul_(beta2).add_(grad, alpha=1.0 - beta2)
```

Updated

```python
v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
```

Bias correction

Original

```python
bias_correction1 = 1.0 - beta1 * t
bias_correction2 = 1.0 - beta2 * t
```

Updated

```python
bias_correction1 = 1.0 - beta1 ** t
bias_correction2 = 1.0 - beta2 ** t
```

## Schedulers

### `Schedulers/scheduler.py` `[Stage I]`

- Added `none_scheduler(...)` to the scheduler registry so the notebook configuration can request `scheduler_name="none"`.

Added

```python
def _constant_one(_):
    return 1.0


def none_scheduler(optimizer, args):
    """No-op scheduler used when the notebook requests no scheduler."""
    return LambdaLR(optimizer, lr_lambda=_constant_one)
```

Registry update

Original

```python
schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
}
```

Updated

```python
schedulers = {
    "none":    none_scheduler,
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
}
```

### `Schedulers/lambda_scheduler.py` `[Stage II]`

- Corrected `LambdaLR` to multiply by the lambda factor rather than add it.

Original

```python
return [base_lr + factor for base_lr in self.base_lrs]
```

Updated

```python
return [base_lr * factor for base_lr in self.base_lrs]
```

### `Schedulers/step_scheduler.py` `[Stage II]`

- Corrected `StepLR` to use exponential step decay.

Original

```python
return [
    base_lr * self.gamma * (t // self.step_size)
    for base_lr in self.base_lrs
]
```

Updated

```python
return [
    base_lr * (self.gamma ** (t // self.step_size))
    for base_lr in self.base_lrs
]
```

### `Schedulers/cosine_scheduler.py` `[Stage II]`

- Corrected the cosine annealing formula and constant name.

Original

```python
return [
    self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.PI * t / self.T_max))
    for base_lr in self.base_lrs
]
```

Updated

```python
return [
    self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max))
    for base_lr in self.base_lrs
]
```

## EvaluateTools

### `EvaluateTools/evaluate.py` `[Stage I]`

- Corrected the checkpoint key used when loading model weights.

Original

```python
model.load_state_dict(ckpt["model"])
```

Updated

```python
model.load_state_dict(ckpt["model_state"])
```

### `EvaluateTools/evaluate.py` `[Stage II]`

- Added `weights_only=False` to `torch.load(...)` as a compatibility safeguard for newer PyTorch versions when loading trusted local checkpoints.

Original

```python
ckpt = torch.load(ckpt_path, map_location=DEVICE)
```

Updated

```python
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
```

### `EvaluateTools/eval_utils.py` `[Stage II]`

- Corrected prediction decoding to take `argmax(...)` over the sequence dimension per example.

Original

```python
yp1 = torch.argmax(p1, dim=0)
yp2 = torch.argmax(p2, dim=0)
```

Updated

```python
yp1 = torch.argmax(p1, dim=1)
yp2 = torch.argmax(p2, dim=1)
```
