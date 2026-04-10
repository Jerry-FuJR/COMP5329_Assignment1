# Baseline Parameters

This file records the baseline configuration derived from the current `assignment1.ipynb` setup for Stage 3 experiments.

## Training Loop

- `num_steps = 1000`
- `batch_size = 8`
- `checkpoint = 200`
- `val_num_batches = 150`
- `test_num_batches = 150`
- `seed = 42`
- `grad_clip = 5.0`
- `early_stop = 10`

## Optimization

- `optimizer_name = "sgd"`
- `scheduler_name = "lambda"`
- `loss_name = "qa_nll"`

## Optimizer Hyperparameters

- `learning_rate = 1e-3`
- `beta1 = 0.8`
- `beta2 = 0.999`
- `eps = 1e-7`
- `weight_decay = 3e-7`
- `momentum = 0.9`

## Scheduler Hyperparameters

- `lr_step_size = 10000`
- `lr_gamma = 0.5`

## Model Architecture

- `para_limit = 400`
- `ques_limit = 50`
- `char_limit = 16`
- `d_model = 96`
- `num_heads = 8`
- `glove_dim = 300`
- `char_dim = 64`
- `dropout = 0.1`
- `dropout_char = 0.05`
- `pretrained_char = False`

## Mechanism Settings

- `norm_name = "layer_norm"`
- `norm_groups = 8`
- `activation = "relu"`
- `init_name = "kaiming"`
- `use_batch_norm = False`

## Evaluation

The final evaluation should use the same model-related parameters as training, especially:

- `para_limit = 400`
- `ques_limit = 50`
- `char_limit = 16`
- `d_model = 96`
- `num_heads = 8`
- `glove_dim = 300`
- `char_dim = 64`
- `dropout = 0.1`
- `dropout_char = 0.05`
- `pretrained_char = False`

## Note

For Stage 3 controlled experiments, this baseline should remain fixed. Only the target experimental variable should be changed in each study.
