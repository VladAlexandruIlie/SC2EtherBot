import torch


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space_shape):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

    def cuda(self):
        self.observations = self.observations.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, current_obs, action, value_pred, reward, mask):
        self.observations[step + 1].copy_(current_obs)
        self.actions[step].copy_(action)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)

    def compute_returns(self, next_value, use_gae, gamma, tau=0.1):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step] + self.rewards[step]
