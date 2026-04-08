# BUG FIX LOG

This document summarizes the main bugs and runtime errors identified and fixed during debugging of the `COMP5329_Assignment1` repository.

## Losses

- `Losses/loss.py`
  - Fixed incorrect order of the method of cross entropy.
  - Changed cross_entropy(y1, p1) to cross_entropy(p1, y1)

## Models

- `Models/Activations/leakeyReLU.py`
  - Fixed incorrect fomula for leaky ReLU in `forward`.
  - Changed where(x < 0, x, self.negative_slope * x) to where(x < 0, self.negative_slope * x, x).

- `Models/Activations/relu.py`
  - Fixed incorrect fomula for ReLU in `forward`.
  - Changed x.clamp(max=0.0) to x.clamp(min=0.0)

- `Models/Initializations/kaiming.py`
  - Fixed incorrect std for kaiming initialization in `kaiming_normal_` and `kaiming_uniform_`.
  - Changed math.sqrt(1.0 / fan) to math.sqrt(2.0 / fan) in both normal and uniform.

- `Models/Initializations/xavier.py`
  - Fixed incorrect std for xavier initialization in `xavier_normal_` and `xavier_uniform_`.
  - Changed std = ...(1.0/...) to std = ...(2.0/...) in both normal and uniform.

- `Models/Normalizations/layernorm.py`
  - Fixed the incorrect parameter in x.mean and x.var
  - Changed x.mean(...,keepdim=False) to x.mean(...,keepdim=False).
  - Changed x.var(...,keepdim=False) to x.var(...,keepdim=False).

- `Models/Normalizations/groupnorm.py`
  - Fixed the incorrect size of groups to reshape
  - Changed (B, C // self.G, self.G, *spatial) to (B, self.G, C // self.G, *spatial).

- `Models/attention.py`
  - Fixed incorrect batch matrix multiplication order in `CQAttention`.
  - Changed `torch.bmm(Q, S1)` to `torch.bmm(S1, Q)`.

- `Models/conv.py`
  - Fixed `Conv1d` sliding window extraction dimension.
  - Changed `unfold(1, ...)` to `unfold(2, ...)` so convolution runs along sequence length.
  - Fixed `Conv2d` width padding shape mismatch by using the current padded height.
  - Fixed `DepthwiseSeparableConv` forward order to apply `depthwise` before `pointwise`.

- `Models/dropout.py`
  - Fixed inverted dropout scaling.
  - Changed scaling from `/ p` to `/ (1 - p)`.

- `Models/embedding.py`
  - Fixed incorrect transpose in `Highway.forward`.
  - Changed `transpose(0, 2)` to `transpose(1, 2)`.
  - Fixed incorrect character embedding permutation.
  - Changed to `[B, d_char, L, char_len]` ordering before 2D convolution.

- `Models/encoder.py`
  - Fixed positional encoding broadcasting bug by correcting the `freqs` shape.
  - Fixed normalization indexing bug by changing `self.norms[i + 1]` to `self.norms[i]`.
  - Fixed self-attention output being overwritten; now uses residual addition.

- `Models/heads.py`
  - Fixed incorrect concatenation dimension in `Pointer`.
  - Changed `torch.cat(..., dim=0)` to `torch.cat(..., dim=1)`.

- `Models/qanet.py`
  - Fixed swapped word and character embedding lookups.
  - Fixed incorrect mask order passed into `CQAttention`.

## Optimizers

- `Optimizers/sgd.py`
  - Fixed weight decay sign.
  - Changed from subtracting `wd * p` inside the gradient term to adding it.

- `Optimizers/sgd_momentum.py`
  - Fixed inconsistent optimizer state keys (`vel` vs `velocity`).
  - Fixed momentum update direction to standard SGD with momentum.

- `Optimizers/adam.py`
  - Fixed state key mismatch (`exp_avg`, `exp_avg_sq`).
  - Fixed second moment update to use `grad^2`.
  - Fixed bias correction formulas from `1 - beta * t` to `1 - beta ** t`.
  - Fixed weight decay sign.

## Schedulers

- `Schedulers/lambda_scheduler.py`
  - Fixed learning rate computation.
  - Changed from `base_lr + factor` to `base_lr * factor`.

- `Schedulers/step_scheduler.py`
  - Fixed step decay formula.
  - Changed to `base_lr * gamma ** floor(t / step_size)`.

- `Schedulers/cosine_scheduler.py`
  - Fixed use of `math.PI`; replaced with `math.pi`.
  - Fixed missing `0.5` factor in cosine annealing formula.

- `Schedulers/scheduler.py`
  - Replaced local lambda used by `lambda_scheduler` with a top-level helper function.
  - This fixed checkpoint serialization failure:
    - `AttributeError: Can't pickle local object 'lambda_scheduler.<locals>.<lambda>'`

## Training / Evaluation

- `TrainTools/train.py`
  - Fixed invalid `argparse.Namespace(...)` construction.
  - Changed to `argparse.Namespace(**{...})`.

- `TrainTools/train_utils.py`
  - Fixed backward pass call from `loss.item().backward()` to `loss.backward()`.
  - Fixed gradient clipping order so clipping occurs before `optimizer.step()`.

- `EvaluateTools/evaluate.py`
  - Fixed checkpoint loading key from `ckpt["model"]` to `ckpt["model_state"]`.
  - Added `weights_only=False` to `torch.load(...)` for compatibility with PyTorch 2.6 when loading trusted local checkpoints.

- `EvaluateTools/eval_utils.py`
  - Fixed prediction decoding by changing `argmax` from `dim=0` to `dim=1`.

## Notebook / Runtime Issues

- `assignment1.ipynb`
  - The training cell used `scheduler_name = "none"`, but the codebase only supported:
    - `cosine`
    - `step`
    - `lambda`
  - This was resolved by using `scheduler_name = "lambda"` for fixed learning rate behavior.

- Jupyter module cache
  - Some errors persisted after code fixes because notebook cells were still using old imported modules.
  - Resolved by restarting the kernel or reloading modules.
