# Debug Error Log

This file records the confirmed internal debugging errors reported from `assignment1.ipynb`.

## Error 001

Location:

- Section 3 - Train

Most important error information:

- The run fails immediately when entering the training function.
- Error type: `TypeError`
- Core message:
  `Namespace.__init__() takes 1 positional argument but 2 were given`

Preliminary analysis:

- The failure happens before the actual training loop starts.
- It is not yet a forward-pass, backward-pass, loss, or optimizer-step error.
- The message strongly suggests that a parameter container or argument object is being created with the wrong calling convention.

Possible cause inferred from the traceback:

- A dictionary is likely being passed into `argparse.Namespace` as a positional argument.
- `argparse.Namespace` expects keyword-style construction, so this misuse triggers the `TypeError`.

Problems found after checking the related part:

- The issue is in the training entry path, not in the dataset or model body yet.
- The argument-packaging step inside `TrainTools/train.py` is incorrect.
- Because of that, training stops before model initialization and before any real learning logic is executed.

Applied fix:

- Corrected the argument container construction in the training entry path.

Code change:

Original

```python
args = argparse.Namespace({k: v for k, v in locals().items()})
```

Updated

```python
args = argparse.Namespace(**{k: v for k, v in locals().items()})
```

Reason for the fix:

- The function was packaging all local training arguments into an `argparse.Namespace`.
- That packaging step used the wrong call form, which caused training to fail before model initialization.
- The fix changes the call so the collected values are passed in the way `argparse.Namespace` expects.


## Error 002

Location:

- Section 3 - Train

Most important error information:

- The run now reaches model construction, but fails while building positional encoding.
- Error type: `RuntimeError`
- Core message:
  `The size of tensor a (400) must match the size of tensor b (96) at non-singleton dimension 1`

Preliminary analysis:

- The failure happens during initialization of the encoder's positional encoding.
- This is a tensor shape mismatch inside model setup, before the training loop starts.
- The mismatch involves the sequence length (`400`) and model dimension (`96`).

Possible cause inferred from the traceback:

- A tensor intended to have shape `[C, 1]` was created with the wrong singleton dimension.
- That makes the frequency term align as `[1, C]` instead of `[C, 1]`, so it cannot broadcast correctly against the position matrix `[C, L]`.

Problems found after checking the related part:

- The positional encoding construction in `Models/encoder.py` used the wrong `unsqueeze` direction for the frequency tensor.
- The comment says the tensor should behave like `[C, 1]`, but the actual code produced `[1, C]`.
- Because of that, the sinusoidal encoding could not be constructed for the configured model size and context length.

Applied fix:

- Corrected the frequency tensor shape in the positional encoding initialization so it broadcasts properly with the position matrix.

Code change:

Original

```python
).unsqueeze(0)  # [C, 1]
```

Updated

```python
).unsqueeze(1)  # [C, 1]
```

Reason for the fix:

- The positional encoding must combine a `[C, 1]` frequency term with a `[C, L]` position grid.
- Using the wrong singleton axis caused the model to fail during encoder initialization.


## Error 003

Location:

- Section 3 - Train

Most important error information:

- The run now gets through model construction and argument validation, but fails when checking the scheduler choice.
- Error type: `ValueError`
- Core message:
  `Unknown scheduler 'none'. Available: ['cosine', 'step', 'lambda']`

Preliminary analysis:

- This is a configuration-to-registry mismatch.
- The notebook is explicitly asking for a scheduler named `none`.
- The training code validates scheduler names against an internal registry, and `none` is not present there.

Possible cause inferred from the traceback:

- The notebook assumes that "no scheduler" is a supported training option.
- The codebase does not currently provide a matching entry in the scheduler registry.

Problems found after checking the related part:

- `Schedulers/scheduler.py` only registered `cosine`, `step`, and `lambda`.
- There was no no-op scheduler available under the name `none`.
- Because the registry check happens before training begins, the pipeline stops immediately at configuration validation.

Applied fix:

- Added a no-op scheduler option named `none` to the scheduler registry.

Code change:

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

Reason for the fix:

- The training configuration used in Section 3 explicitly requests `scheduler_name="none"`.
- To make that configuration executable, the scheduler registry needs a corresponding entry that leaves the learning rate unchanged.


## Error 004

Location:

- Section 3 - Train

Most important error information:

- The run now reaches the embedding stage of the model forward pass.
- Error type: `IndexError`
- Core message:
  `index out of range in self`

Preliminary analysis:

- The traceback points into PyTorch's embedding lookup.
- This means an input index passed into an embedding layer is larger than the valid vocabulary range of that embedding table.
- At this stage, the most likely issue is not the raw data file itself, but a mismatch between input tensor meaning and the embedding layer being used.

Possible cause inferred from the traceback:

- Word indices and character indices may be routed to the wrong embedding tables.
- If word ids are sent into the character embedding table, out-of-range indexing is very likely.

Problems found after checking the related part:

- In `Models/qanet.py`, the context-side embedding lookup used the wrong correspondence between tensors and embedding layers.
- `Cwid` is the context word-index tensor and should go into the word embedding.
- `Ccid` is the context character-index tensor and should go into the character embedding.
- The existing code had those two inputs reversed on the context branch.

Applied fix:

- Corrected the context-side embedding lookup so word indices go to the word embedding and character indices go to the character embedding.

Code change:

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

Reason for the fix:

- The dataset loader clearly distinguishes between word-level and character-level index tensors.
- Passing them into the wrong embedding tables produces invalid embedding lookups and triggers the observed `IndexError`.


## Error 005

Location:

- Section 3 - Train

Most important error information:

- The run now reaches the character embedding convolution path.
- Error type: `RuntimeError`
- Core message:
  `shape '[1, 64, 16, 1, 1]' is invalid for input of size 4096`

Preliminary analysis:

- The failure occurs during a 2D convolution weight reshape inside the character embedding branch.
- This indicates that the convolution layer was initialized for one channel layout, but the actual input tensor arrived with a different channel dimension.
- The mismatch is between the model's expected character embedding dimension and the runtime tensor layout.

Possible cause inferred from the traceback:

- The character embedding tensor was permuted incorrectly before being passed into the 2D convolution.
- As a result, the convolution interpreted `char_limit` as the channel dimension instead of `char_dim`.

Problems found after checking the related part:

- In `Models/embedding.py`, the character embedding tensor should be rearranged into `[B, d_char, L, char_len]` before convolution.
- The existing `permute(...)` order placed the wrong axis into the channel position.
- That made the convolution receive an input with mismatched channel semantics, which caused the downstream weight reshape failure.

Applied fix:

- Corrected the character embedding tensor permutation so the channel dimension is `d_char` before the 2D convolution.

Code change:

Original

```python
ch_emb = ch_emb.permute(0, 2, 1, 3)  # [B, d_char, L, char_len]
```

Updated

```python
ch_emb = ch_emb.permute(0, 3, 1, 2)  # [B, d_char, L, char_len]
```

Reason for the fix:

- The 2D character convolution is constructed with `in_channels = d_char`.
- If the input tensor does not place `d_char` on the channel axis, the convolution internals break immediately.


## Error 006

Location:

- Section 3 - Train

Most important error information:

- The run now gets into the custom 2D convolution padding path.
- Error type: `RuntimeError`
- Core message:
  `Sizes of tensors must match except in dimension 3. Expected size 400 but got size 404 for tensor number 1 in the list.`

Preliminary analysis:

- The failure occurs during the width-padding concatenation step in the custom 2D convolution.
- This means the tensors being concatenated along the width axis do not share the same height.
- The mismatch appears immediately after height padding has already been applied.

Possible cause inferred from the traceback:

- The code likely constructs the width-padding tensor using the original height instead of the already padded height.
- That would produce a tensor with height `H`, while the current convolution input already has height `H + 2p`.

Problems found after checking the related part:

- In `Models/conv.py`, `Conv2d.forward()` first pads the height dimension.
- During the next step, the code creates `pad_w` using the old `H` value.
- At that moment, the actual tensor height is already updated, so the width-padding tensor does not match the current input tensor.

Applied fix:

- Corrected the width-padding tensor construction to use the current padded height instead of the stale original height.

Code change:

Original

```python
pad_w = x.new_zeros(B, C_in, H, p)
```

Updated

```python
pad_w = x.new_zeros(B, C_in, x.size(2), p)
```

Reason for the fix:

- After height padding, the tensor shape changes.
- Any later width-padding tensor must match that updated height, otherwise concatenation along the width axis fails immediately.


## Error 007

Location:

- Section 3 - Train

Most important error information:

- The run now gets past the character convolution path and fails inside a linear layer.
- Error type: `RuntimeError`
- Core message:
  `mat1 and mat2 shapes cannot be multiplied (145600x8 and 364x364)`

Preliminary analysis:

- The failure occurs in `torch.nn.Linear`.
- A linear layer with feature size `364` received an input whose last dimension is `8`.
- This strongly suggests that the batch dimension was moved into the feature position by mistake.

Possible cause inferred from the traceback:

- A tensor that should have been rearranged from `[B, C, L]` to `[B, L, C]` was transposed using the wrong axes.
- As a result, the linear layer received the batch dimension as the last axis instead of the feature dimension.

Problems found after checking the related part:

- In `Models/embedding.py`, `Highway.forward()` is intended to convert `[B, C, L]` into `[B, L, C]` before applying `nn.Linear(size, size)`.
- The existing transpose used the wrong dimensions, which moved the batch axis into the wrong place.
- That made the linear layer see `batch_size` as the input feature dimension, causing the matrix multiplication failure.

Applied fix:

- Corrected the transpose in `Highway.forward()` so the tensor is rearranged into `[B, L, C]` before the linear layers.

Code change:

Original

```python
x = x.transpose(0, 2)
```

Updated

```python
x = x.transpose(1, 2)
```

Reason for the fix:

- The highway layer is defined with feature size `d_word + d_char = 364`.
- Therefore the last dimension passed into the linear layers must be the feature dimension, not the batch dimension.


## Error 008

Location:

- Section 3 - Train

Most important error information:

- The run now gets into the custom 1D convolution path.
- Error type: `RuntimeError`
- Core message:
  `shape '[8, 364, 0, 400, 5]' is invalid for input of size 1486720`

Preliminary analysis:

- The failure occurs while reshaping the unfolded 1D convolution windows.
- The presence of `0` in the target shape shows that the grouped-channel calculation became invalid during the reshape logic.
- This strongly suggests that the convolution windows were extracted along the wrong axis.

Possible cause inferred from the traceback:

- The custom 1D convolution should slide over the sequence-length dimension of an input shaped `[B, C_in, L]`.
- If `unfold(...)` is applied to the channel dimension instead, all downstream shape assumptions break.

Problems found after checking the related part:

- In `Models/conv.py`, `Conv1d.forward()` was using `unfold(...)` on dimension `1`.
- For an input shaped `[B, C_in, L]`, dimension `1` is the channel axis, not the sequence-length axis.
- The rest of the convolution logic assumes the windows came from the length dimension, so the reshape step fails.

Applied fix:

- Corrected the `unfold(...)` dimension in the custom 1D convolution so sliding windows are extracted along the sequence-length axis.

Code change:

Original

```python
x_unf = x.unfold(1, self.kernel_size, 1)  # [B, C_in, L_out, k]
```

Updated

```python
x_unf = x.unfold(2, self.kernel_size, 1)  # [B, C_in, L_out, k]
```

Reason for the fix:

- A 1D convolution over `[B, C_in, L]` must slide over `L`, not over `C_in`.
- Using the wrong axis corrupts the internal tensor layout and causes the grouped reshape to fail.


## Error 009

Location:

- Section 3 - Train

Most important error information:

- The run still fails in the custom 1D convolution path after window extraction was corrected.
- Error type: `RuntimeError`
- Core message:
  `shape '[8, 364, 0, 400, 5]' is invalid for input of size 1536000`

Preliminary analysis:

- The grouped reshape still fails because the effective per-group input channel count becomes zero.
- This indicates that the convolution layer is now receiving fewer runtime channels than the depthwise grouping expects.
- That kind of mismatch usually means the channel-transform order inside a depthwise-separable block is wrong.

Possible cause inferred from the traceback:

- A pointwise convolution may be running before the depthwise convolution.
- If so, the pointwise stage changes the channel count first, but the depthwise stage still expects the original input-channel count and group structure.

Problems found after checking the related part:

- In `Models/conv.py`, `DepthwiseSeparableConv.forward()` applied the pointwise convolution before the depthwise convolution.
- For layers such as the embedding-to-model projection, that changes the channel count too early.
- The following depthwise convolution still expects the original `in_ch` and corresponding `groups=in_ch`, which causes the grouped reshape failure.

Applied fix:

- Corrected the execution order in `DepthwiseSeparableConv.forward()` so the depthwise convolution runs before the pointwise convolution.

Code change:

Original

```python
return self.depthwise_conv(self.pointwise_conv(x))
```

Updated

```python
return self.pointwise_conv(self.depthwise_conv(x))
```

Reason for the fix:

- Standard depthwise-separable convolution first performs channel-wise spatial filtering, then mixes channels with a pointwise projection.
- Reversing that order breaks the expected channel/group relationship for the depthwise stage.


## Error 010

Location:

- Section 3 - Train

Most important error information:

- The run now reaches the custom normalization path and fails inside LayerNorm.
- Error type: `RuntimeError`
- Core message:
  `The size of tensor a (400) must match the size of tensor b (8) at non-singleton dimension 2`

Preliminary analysis:

- The failure occurs when the normalization code tries to subtract the computed mean from the original tensor.
- This indicates that the reduced statistics do not have a shape that can broadcast back to the input tensor.
- The problem appears in the LayerNorm implementation, not in the calling code.

Possible cause inferred from the traceback:

- The mean and variance were reduced with dimensions removed, so they no longer align with the `[B, C, L]` input layout.
- There is also an obvious formula issue in the affine transform applied after normalization.

Problems found after checking the related part:

- In `Models/Normalizations/layernorm.py`, both `mean` and `var` were computed with `keepdim=False`.
- For a LayerNorm over the last two dimensions of `[B, C, L]`, the statistics should retain singleton dimensions so they can broadcast back onto the original tensor.
- The affine transform at the end of the function also used the wrong parameter order: it multiplied by `bias` and added `weight`, which is the opposite of the standard LayerNorm formula.

Applied fix:

- Corrected the statistic reduction to keep broadcastable dimensions.
- Corrected the final affine transform to use the standard `x_norm * weight + bias` form.

Code change:

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

Reason for the fix:

- The current runtime blocker is the broadcasting failure, which requires `keepdim=True`.
- The affine formula bug is in the same function and part of the same normalization implementation, so it was corrected together to avoid leaving an obviously invalid LayerNorm definition in place.


## Error 011

Location:

- Section 3 - Train

Most important error information:

- The run now gets deeper into the encoder block and fails with an index-out-of-range error.
- Error type: `IndexError`
- Core message:
  `index 4 is out of range`

Preliminary analysis:

- This error pattern is consistent with an out-of-range access on a PyTorch container such as `ModuleList`.
- Since the run had already moved into the encoder block, the most likely source is an internal layer-list indexing bug rather than dataset indexing.

Possible cause inferred from the traceback:

- A loop over convolution sublayers is probably indexing the normalization layers using an offset that goes one step past the valid range.

Problems found after checking the related part:

- In `Models/encoder.py`, `self.norms` is created with length `conv_num`.
- Inside the convolution loop, the code accessed `self.norms[i + 1]`.
- On the last iteration, that attempts to access one element beyond the end of the `ModuleList`.

Applied fix:

- Corrected the normalization-layer indexing inside the encoder block loop.

Code change:

Original

```python
out = self.norms[i + 1](out)
```

Updated

```python
out = self.norms[i](out)
```

Reason for the fix:

- The normalization layer list has one element per convolution block.
- Each loop iteration should use the normalization layer at the same index as the current convolution stage.


## Error 012

Location:

- Section 3 - Train

Most important error information:

- The run now reaches the context-question attention stage and fails during mask application.
- Error type: `RuntimeError`
- Core message:
  `The size of tensor a (400) must match the size of tensor b (50) at non-singleton dimension 2`

Preliminary analysis:

- The mismatch is between the context length (`400`) and question length (`50`).
- This strongly indicates that one of the attention masks is being applied to the wrong sequence axis.
- The failure occurs inside `mask_logits(...)`, so the masking stage is receiving incompatible shapes.

Possible cause inferred from the traceback:

- The context mask and question mask are likely being passed into the context-question attention module in the wrong order.

Problems found after checking the related part:

- In `Models/qanet.py`, the call to `self.cq_att(...)` passed `qmask` before `cmask`.
- The attention module expects the context mask to align with context-length dimensions and the question mask to align with question-length dimensions.
- Swapping them produces the observed `400` vs `50` shape mismatch during masking.

Applied fix:

- Corrected the mask argument order when calling the context-question attention module.

Code change:

Original

```python
X = self.cq_att(Ce, Qe, qmask, cmask)
```

Updated

```python
X = self.cq_att(Ce, Qe, cmask, qmask)
```

Reason for the fix:

- Context and question masks must align with their respective sequence lengths.
- Passing them in the wrong order breaks the masking step inside attention.


## Error 013

Location:

- Section 3 - Train

Most important error information:

- The run now reaches the batch matrix multiplication stage inside context-question attention.
- Error type: `RuntimeError`
- Core message:
  `Expected size for first two dimensions of batch2 tensor to be: [8, 96] but got: [8, 400].`

Preliminary analysis:

- The failure occurs in `torch.bmm(...)`.
- This indicates that the two tensors are being multiplied in an order that makes their inner dimensions incompatible.
- Since the attention weights and question representation are both involved here, the error is most likely a matrix-multiplication ordering issue inside `CQAttention`.

Possible cause inferred from the traceback:

- The context-to-question attention weights have shape `[B, Lc, Lq]`.
- The question representation has shape `[B, Lq, C]`.
- To obtain an attended question summary for each context position, the multiplication should be `S1 @ Q`, not `Q @ S1`.

Problems found after checking the related part:

- In `Models/attention.py`, the code computed `A = torch.bmm(Q, S1)`.
- That multiplication order makes the inner dimensions inconsistent.
- The intended output shape for `A` is `[B, Lc, C]`, which is achieved by multiplying `S1` with `Q` in the opposite order.

Applied fix:

- Corrected the batch matrix multiplication order in the attention module when computing the attended question representation.

Code change:

Original

```python
A = torch.bmm(Q, S1)
```

Updated

```python
A = torch.bmm(S1, Q)
```

Reason for the fix:

- The attention weights should be used to aggregate question vectors for each context position.
- That requires the multiplication order `S1 @ Q`, which yields the correct `[B, Lc, C]` result.


## Error 014

Location:

- Section 3 - Train

Most important error information:

- The run now reaches the output pointer head and fails during the first span-logit computation.
- Error type: `RuntimeError`
- Core message:
  `size mismatch, got input (6400), mat (6400x96), vec (192)`

Preliminary analysis:

- The pointer head expects a concatenated feature tensor with size `2 * d_model` along the feature/channel axis.
- The parameter vector has length `192`, which matches `2 * 96`.
- The runtime tensor still presents a `96`-sized feature dimension, so the concatenation did not happen along the intended axis.

Possible cause inferred from the traceback:

- Two model-state tensors were concatenated along the batch dimension instead of the channel dimension.

Problems found after checking the related part:

- In `Models/heads.py`, `X1` was built using `torch.cat([M1, M2], dim=0)`.
- That doubles the batch dimension instead of combining features into `[B, 2C, L]`.
- The following matrix multiplication therefore receives the wrong feature layout and fails.

Applied fix:

- Corrected the concatenation axis for `X1` in the pointer head so the two feature maps are concatenated along the channel dimension.

Code change:

Original

```python
X1 = torch.cat([M1, M2], dim=0)  # [B, 2C, L]
```

Updated

```python
X1 = torch.cat([M1, M2], dim=1)  # [B, 2C, L]
```

Reason for the fix:

- The pointer parameter vector is sized for `2C` features.
- Therefore the model states must be concatenated along the feature/channel axis, not the batch axis.


## Error 015

Location:

- Section 3 - Train

Most important error information:

- The run now reaches the NLL loss computation stage.
- Error type: `RuntimeError`
- Core message:
  `0D or 1D target tensor expected, multi-target not supported`

Preliminary analysis:

- This error indicates that `torch.nn.functional.nll_loss` received a target tensor with the wrong shape.
- In this setting, the target should be a 1D tensor of answer positions.
- A very common cause is that the model output tensor and the target tensor were passed into `nll_loss` in the wrong order.

Possible cause inferred from the traceback:

- The loss function likely used `F.nll_loss(target, prediction)` instead of `F.nll_loss(prediction, target)`.

Problems found after checking the related part:

- In `Losses/loss.py`, the first NLL loss call inside `qa_nll_loss(...)` had its arguments reversed.
- That caused the model target tensor to be interpreted as the input distribution and the model output tensor to be interpreted as the target, which is incompatible with `nll_loss`.

Applied fix:

- Corrected the argument order of the first `F.nll_loss(...)` call in `qa_nll_loss(...)`.

Code change:

Original

```python
return 0.5 * (F.nll_loss(y1, p1) + F.nll_loss(p2, y2))
```

Updated

```python
return 0.5 * (F.nll_loss(p1, y1) + F.nll_loss(p2, y2))
```

Reason for the fix:

- `F.nll_loss` expects the model prediction first and the class-index target second.
- Reversing them makes PyTorch interpret the tensors with incompatible semantics and shapes.


## Error 016

Location:

- Section 3 - Train

Most important error information:

- The run now reaches the backward-pass stage of the training loop.
- Error type: `AttributeError`
- Core message:
  `'float' object has no attribute 'backward'`

Preliminary analysis:

- The training loop successfully computed a loss value, but failed when trying to backpropagate.
- This indicates that the object used for `.backward()` is no longer a PyTorch tensor.
- The most likely cause is that the scalar Python value extracted from the tensor was used instead of the tensor itself.

Possible cause inferred from the traceback:

- The code likely called `.item()` on the loss before calling `.backward()`.

Problems found after checking the related part:

- In `TrainTools/train_utils.py`, the training loop invoked `loss.item().backward()`.
- Calling `.item()` converts a tensor into a plain Python float.
- A float has no computation graph and therefore cannot support backpropagation.

Applied fix:

- Corrected the backward call so it is applied to the loss tensor instead of its Python scalar value.

Code change:

Original

```python
loss.item().backward()
```

Updated

```python
loss.backward()
```

Reason for the fix:

- Autograd can only backpropagate through a tensor that still carries its computation graph.
- Converting the loss to a Python number destroys that graph before the backward pass.


## Error 017

Location:

- Section 3 - Train
- Training became runnable, but persistent `nan` loss remained

Most important issue information:

- The training loop was able to run through optimization, evaluation, and checkpointing.
- However, the reported training loss and evaluation loss became `nan`.
- This indicated that the remaining blocker was no longer pipeline execution, but numerical instability during training.

Why this part was investigated:

- After the earlier execution errors were removed, the remaining visible problem was persistent `nan` loss.
- The investigation therefore focused on high-impact numerical issues in model scaling, initialization, attention, and parameter updates.
- Although several of the fixes in this section also touch mechanism-level topics that can be viewed as part of Stage II, they were handled here because the immediate priority was to make training numerically stable and prevent persistent `nan` loss from blocking end-to-end executability.

Problems found after checking the related parts:

- In `Models/dropout.py`, the code sampled a keep mask with probability `1 - p`, which is correct.
- However, the surviving activations were rescaled by dividing by `p` instead of by `1 - p`.
- That means the implementation amplified activations far too aggressively, especially for small dropout rates such as `0.1` or `0.05`.
- In `Models/encoder.py`, multi-head attention computed `q @ k^T` without applying the standard scaling factor `1 / sqrt(d_k)`.
- In `Models/Initializations/kaiming.py`, the implemented standard deviation used `sqrt(1 / fan)` even though the intended Kaiming formula requires `sqrt(2 / fan)`.
- In `Models/Initializations/xavier.py`, the implemented standard deviation used `sqrt(2 / (fan_in * fan_out))` instead of `sqrt(2 / (fan_in + fan_out))`.
- In `TrainTools/train_utils.py`, gradient clipping was applied after `optimizer.step()`.
- In `Optimizers/sgd.py`, the `weight_decay` term was added with the wrong sign.

Applied fix:

- Corrected the scaling factor in the custom dropout implementation.
- Restored scaled dot-product attention in multi-head attention.
- Corrected the Kaiming initialization formulas.
- Corrected the Xavier initialization formulas.
- Moved gradient clipping to occur before the optimizer step.
- Corrected the sign of the `weight_decay` term in the custom SGD implementation.

Code change:

Dropout scaling

Original

```python
return x * mask / self.p
```

Updated

```python
return x * mask / (1.0 - self.p)
```

Attention scaling

Original

```python
attn = torch.bmm(q, k.transpose(1, 2))
```

Updated

```python
attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
```

Kaiming initialization

Original

```python
std = math.sqrt(1.0 / fan)
```

Updated

```python
std = math.sqrt(2.0 / fan)
```

Xavier initialization

Original

```python
std = gain * math.sqrt(2.0 / (fan_in * fan_out))
```

Updated

```python
std = gain * math.sqrt(2.0 / (fan_in + fan_out))
```

Gradient clipping order

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

SGD weight decay

Original

```python
grad = grad.add(p, alpha=-wd)
```

Updated

```python
grad = grad.add(p, alpha=wd)
```

Reason for the fix:

- Inverted dropout must scale by the keep probability rather than the drop probability.
- Scaled dot-product attention and correct Kaiming/Xavier formulas are needed to keep activations and logits in a stable range.
- Gradient clipping must be applied before the optimizer step, and the SGD weight-decay sign must match standard L2-style decay.
- These fixes were grouped into one error because they all addressed the same remaining symptom: persistent `nan` loss after the training pipeline became runnable.


## Error 018

Location:

- Section 4 - Evaluate

Most important error information:

- Training completed and produced a checkpoint file.
- The failure occurred when `evaluate(...)` tried to load the saved model weights from that checkpoint.
- Error type: `KeyError`
- Core message:
  `'model'`

Preliminary analysis:

- This is a checkpoint field-name mismatch, not a model-forward or evaluation-metric computation problem.
- The checkpoint file exists and is readable, but the evaluation code is trying to access a key that is not present.

Possible cause inferred from the traceback:

- The training code and evaluation code are using different names for the saved model state inside the checkpoint payload.

Problems found after checking the related parts:

- In `TrainTools/train_utils.py`, the checkpoint is saved with the key `model_state`.
- In `EvaluateTools/evaluate.py`, the evaluation code attempted to load `ckpt["model"]`.
- Because those two names do not match, evaluation fails immediately at checkpoint loading.

Applied fix:

- Corrected the checkpoint key used by the evaluation code so it matches the key written during training.

Code change:

Original

```python
model.load_state_dict(ckpt["model"])
```

Updated

```python
model.load_state_dict(ckpt["model_state"])
```

Reason for the fix:

- The saved checkpoint format must be read consistently by both training and evaluation.
- Using the actual saved field name allows evaluation to reload the checkpoint correctly.

