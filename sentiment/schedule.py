import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosineDecay(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_steps, initial_lr, warmup_target, alpha, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.initial_lr = initial_lr
        self.warmup_target = warmup_target
        self.alpha = alpha
        self.min_lr = initial_lr * alpha
        super(WarmupCosineDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            lr = self.initial_lr + \
                (self.warmup_target - self.initial_lr) * \
                (self.last_epoch / self.warmup_steps)
        elif self.last_epoch < self.warmup_steps + self.decay_steps:
            # Cosine decay phase
            progress = (self.last_epoch - self.warmup_steps) / self.decay_steps
            lr = self.min_lr + 0.5 * \
                (self.warmup_target - self.min_lr) * \
                (1 + math.cos(math.pi * progress))
        else:
            # After decay steps, keep learning rate at min_lr
            lr = self.min_lr

        return [lr for _ in self.optimizer.param_groups]
