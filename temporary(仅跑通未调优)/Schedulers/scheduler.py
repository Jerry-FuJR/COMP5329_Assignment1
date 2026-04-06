from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR, constant_one
from Schedulers.step_scheduler import StepLR


def cosine_scheduler(optimizer, args):
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def lambda_scheduler(optimizer, args):
    return LambdaLR(optimizer, lr_lambda=constant_one)


schedulers = {
    "cosine": cosine_scheduler,
    "step": step_scheduler,
    "lambda": lambda_scheduler,
}
