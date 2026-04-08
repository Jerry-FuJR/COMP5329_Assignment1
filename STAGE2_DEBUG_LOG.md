# Stage II Debug Log

This file records confirmed Stage II mechanism-level fixes after the Stage I pipeline issues were resolved.

## Losses

- Reviewed `Losses/loss.py` and `Losses/__init__.py`.
- No confirmed formula errors or severe logic conflicts were found in the loss implementations.
- `qa_nll_loss(...)` and `qa_ce_loss(...)` are currently consistent with their stated expectations.

## Models / Activations

- Reviewed `Models/Activations/relu.py`, `Models/Activations/leakeyReLU.py`, and `Models/Activations/activation_function.py`.
- Confirmed and fixed two activation-formula errors.

### ReLU

Location:
- `Models/Activations/relu.py`

Problem:
- The custom ReLU clamped values with `max=0.0`, which keeps negative values and suppresses positive values.

Fix:
- Changed the implementation to clamp with `min=0.0`.

Code change:

Original

```python
return x.clamp(max=0.0)
```

Updated

```python
return x.clamp(min=0.0)
```

### LeakyReLU

Location:
- `Models/Activations/leakeyReLU.py`

Problem:
- The branch logic in `torch.where(...)` was reversed.
- Negative inputs were left unchanged, while positive inputs were multiplied by `negative_slope`.

Fix:
- Changed the implementation so negative inputs are scaled and positive inputs pass through unchanged.

Code change:

Original

```python
return torch.where(x < 0, x, self.negative_slope * x)
```

Updated

```python
return torch.where(x < 0, self.negative_slope * x, x)
```

## Models / Normalizations

- Reviewed `Models/Normalizations/groupnorm.py`.
- Confirmed and fixed one group-shape logic error in the custom GroupNorm implementation.

### GroupNorm

Location:
- `Models/Normalizations/groupnorm.py`

Problem:
- The tensor was reshaped as `[B, C//G, G, *spatial]` instead of `[B, G, C//G, *spatial]`.
- This mixes the group dimension with the channel-within-group dimension and makes the normalization statistics inconsistent with GroupNorm.

Fix:
- Changed the reshape order so channels are correctly partitioned into groups before normalization.

Code change:

Original

```python
x = x.view(B, C // self.G, self.G, *spatial)
```

Updated

```python
x = x.view(B, self.G, C // self.G, *spatial)
```

## Models / Core Blocks

- Reviewed the top-level model files in `Models/`, including `attention.py`, `encoder.py`, `dropout.py`, `conv.py`, `embedding.py`, `heads.py`, and `qanet.py`.
- Confirmed and fixed one mechanism-level residual-connection error in `encoder.py`.

### Encoder self-attention residual path

Location:
- `Models/encoder.py`

Problem:
- The output of self-attention was immediately overwritten by the residual tensor.
- This meant the self-attention result was discarded instead of being combined with the residual path.

Fix:
- Changed the code to add the self-attention output back to the residual tensor before dropout.

Code change:

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

## Optimizers

- Reviewed `Optimizers/sgd.py`, `Optimizers/sgd_momentum.py`, `Optimizers/adam.py`, `Optimizers/optimizer.py`, and `Optimizers/__init__.py`.
- Confirmed and fixed mechanism-level errors in the custom momentum SGD and Adam implementations.

### SGDMomentum

Location:
- `Optimizers/sgd_momentum.py`

Problems:
- The optimizer initialized the state buffer with key `vel` but later read from `velocity`.
- The velocity update used subtraction instead of the standard momentum accumulation rule.

Fix:
- Changed the state key to `velocity` consistently.
- Changed the momentum update to `v = momentum * v + grad`.

Code change:

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

### Adam

Location:
- `Optimizers/adam.py`

Problems:
- The optimizer updated weight decay with the wrong sign.
- It initialized state keys as `exp_avg` and `exp_avg_sq` but later tried to read `m` and `v`.
- The second-moment update used `grad` instead of `grad^2`.
- The bias-correction factors used `1 - beta * t` instead of `1 - beta ** t`.

Fix:
- Corrected the sign of the weight-decay term.
- Used the correct state keys `exp_avg` and `exp_avg_sq`.
- Changed the second-moment update to use `grad * grad`.
- Corrected the bias-correction formulas to use exponential powers.

Code change:

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

- Reviewed `Schedulers/scheduler.py`, `Schedulers/lambda_scheduler.py`, `Schedulers/step_scheduler.py`, `Schedulers/cosine_scheduler.py`, and `Schedulers/__init__.py`.
- Confirmed and fixed multiple scheduler-formula and scheduler-semantics errors.

### LambdaLR

Location:
- `Schedulers/lambda_scheduler.py`

Problem:
- The scheduler added the lambda factor to the base learning rate instead of multiplying by it.

Fix:
- Changed the learning-rate update to `base_lr * factor`.

Code change:

Original

```python
return [base_lr + factor for base_lr in self.base_lrs]
```

Updated

```python
return [base_lr * factor for base_lr in self.base_lrs]
```

### StepLR

Location:
- `Schedulers/step_scheduler.py`

Problem:
- The scheduler used `base_lr * gamma * floor(t / step_size)` instead of exponential step decay.

Fix:
- Changed the decay rule to `base_lr * gamma ** floor(t / step_size)`.

Code change:

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

### CosineAnnealingLR

Location:
- `Schedulers/cosine_scheduler.py`

Problem:
- The scheduler used `math.PI` instead of `math.pi`.
- It also omitted the standard `0.5` factor in the cosine annealing formula.

Fix:
- Replaced `math.PI` with `math.pi`.
- Restored the standard `0.5 * (1 + cos(...))` form.

Code change:

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

- Reviewed `EvaluateTools/evaluate.py` and `EvaluateTools/eval_utils.py`.
- Confirmed and fixed one prediction-decoding error in the evaluation utilities.
- Added a checkpoint-loading compatibility safeguard for newer PyTorch versions.

### Prediction decoding axis

Location:
- `EvaluateTools/eval_utils.py`

Problem:
- The start and end predictions were decoded with `argmax(..., dim=0)`.
- Since `p1` and `p2` have shape `[B, L]`, this selects maxima across the batch dimension instead of per example across sequence positions.

Fix:
- Changed both `argmax(...)` calls to operate over `dim=1`.

Code change:

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

### Checkpoint loading compatibility

Location:
- `EvaluateTools/evaluate.py`

Problem:
- On some newer PyTorch versions, `torch.load(...)` may use stricter safe-loading behavior by default.
- For trusted local checkpoints, this can cause unnecessary loading failures depending on version and environment.

Fix:
- Added `weights_only=False` to `torch.load(...)` when loading the assignment checkpoint.

Code change:

Original

```python
ckpt = torch.load(ckpt_path, map_location=DEVICE)
```

Updated

```python
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
```
