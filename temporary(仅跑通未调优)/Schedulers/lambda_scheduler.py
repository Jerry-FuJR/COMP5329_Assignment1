from torch.optim.lr_scheduler import LRScheduler


def constant_one(step: int) -> float:
    return 1.0


class LambdaLR(LRScheduler):
    """lr_t = base_lr * lr_lambda(t)"""

    def __init__(self, optimizer, lr_lambda=constant_one, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        factor = self.lr_lambda(t)
        return [base_lr * factor for base_lr in self.base_lrs]
