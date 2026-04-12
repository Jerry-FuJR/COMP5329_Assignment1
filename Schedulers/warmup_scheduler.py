from torch.optim.lr_scheduler import LRScheduler


class WarmupLambdaLR(LRScheduler):
    """Linear warmup scheduler.

    Linearly increases the learning rate from 0 to target_lr over
    warmup_steps steps, then keeps it constant.

    Effective learning rate:
        lr_t = base_lr * factor(t)

    where:
        factor(t) = (target_lr / base_lr) * min(1, (t + 1) / warmup_steps)
    """

    def __init__(self, optimizer, target_lr, warmup_steps, last_epoch=-1):
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        if target_lr <= 0.0:
            raise ValueError(f"target_lr must be positive, got {target_lr}")

        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        progress = min(1.0, float(t + 1) / float(self.warmup_steps))

        lrs = []
        for base_lr in self.base_lrs:
            if base_lr <= 0.0:
                raise ValueError(f"base_lr must be positive, got {base_lr}")
            factor = (self.target_lr / base_lr) * progress
            lrs.append(base_lr * factor)

        return lrs
