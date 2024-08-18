# warmup_target = 2e-4
# total_steps = int(
#     len_train_data/self.config['batch_size'] * self.config['epochs'])
# warmup_steps = int(total_steps * 0.15)
# decay_steps = int(total_steps * 0.3)
# alpha = 0.1

# self.scheduler = WarmupCosineDecay(
#     self.optimizer,
#     warmup_steps=warmup_steps,
#     decay_steps=decay_steps,
#     initial_lr=1e-4,
#     warmup_target=warmup_target,
#     alpha=alpha
# )
