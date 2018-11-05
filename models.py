import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, num_inputs=8, actions_dim=2, hidden_size=300):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, actions_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.dense(x)
        return x

    def get_action(self, state, exploration=True, exploration_noise_sigma=0.5):
        if len(state.shape) == 1 or len(state.shape) == 3:
            state = state.unsqueeze(0)
        mean = self.__call__(state)
        if not exploration:
            return mean.detach().squeeze().cpu().numpy()
        noise = torch.zeros_like(mean).data.normal_()
        action = noise * exploration_noise_sigma + mean
        return action.detach().squeeze().cpu().numpy()


class Critic(nn.Module):
    def __init__(self, num_inputs=8, actions_dim=2, num_outputs=1, hidden_size=300):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(num_inputs + actions_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, num_outputs),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.dense(x)

