from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR
from Schedulers.warmup_scheduler import WarmupLambdaLR


def _constant_one(_):
    return 1.0


# ── Scheduler factories ──────────────────────────────────────────────────────

def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def lambda_scheduler(optimizer, args):
    """LambdaLR with a constant factor of 1.0 — learning rate stays fixed."""
    return LambdaLR(optimizer, lr_lambda=_constant_one)
        

def warmup_lambda_scheduler(optimizer, args):
    """Warmup LambdaLR with a factor going up - learning rate does not change until 10*e-3."""
    return WarmupLambdaLR(
        optimizer,
        target_lr=args.learning_rate,
        warmup_steps=getattr(args, "warmup_steps", 1000),
    )


# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
    "warmup_lambda": warmup_lambda_scheduler
}
