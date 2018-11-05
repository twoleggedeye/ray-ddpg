import copy
import gym
import torch
import torch.nn.functional as F
import numpy as np
import ray

import models
import core.storage as storage
import core.utils as utils
from core.utils import ewma
from core import workers
from core.wrappers import StorageWrapper
from core.wrappers import ParamServer


@ray.remote(num_gpus=1)
class DDPG(workers.OffPolicyTrainer):
    def __init__(self, config, out_dir):
        super().__init__(config)

        def env_make_fn():
            return gym.make(config['env'])

        self.env = env_make_fn()
        self.device = config['device']
        self.storage = StorageWrapper.remote(storage.ReplayBuffer, [config['replay_buffer_size']], {})
        critic_kwargs = {
            'num_inputs': self.env.observation_space.shape[0],
            'actions_dim': self.env.action_space.shape[0]
        }
        policy_kwargs = critic_kwargs
        self.critic = models.Critic(**critic_kwargs).to(self.device)
        self.policy = models.Policy(**policy_kwargs).to(self.device)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_critic = copy.deepcopy(self.critic)

        self.params_server = ParamServer.remote(utils.get_cpu_state_dict(self.policy))
        self.evaluator = workers.Evaluator.as_remote(num_gpus=config['gpu_per_runner'], num_cpus=config['cpu_per_runner'])
        self.evaluator = self.evaluator.remote(models.Policy,
                                               policy_kwargs,
                                               env_make_fn,
                                               self.params_server,
                                               self.config)

        self.runners = [workers.Runner.as_remote(num_gpus=config['gpu_per_runner'], 
                                                 num_cpus=config['cpu_per_runner']).remote(models.Policy,
                                                                                           policy_kwargs,
                                                                                           env_make_fn,
                                                                                           self.params_server,
                                                                                           self.storage,
                                                                                           self.config)
                        for _ in range(self.config['n_runners'])]

        self.critic.train()
        self.policy.train()
        self.target_policy.eval()
        self.target_critic.eval()
        self.opt_policy = torch.optim.Adam([{'params': self.policy.parameters(), 'lr': self.config['policy_lr']}])
        self.opt_critic = torch.optim.Adam([{'params': self.critic.parameters(), 'lr': self.config['critic_lr']}])
        self.critic_loss = None
        self.policy_loss = None

    def _adjust_schedule(self):
        pass

    def _make_updates(self):
        state, action, reward, next_state, done = self._sample_batch(self.config['batch_size'])
        for _ in range(self.config["critic_steps"]):
            cl = self.update_critic(state, action, next_state, reward, done)
            self.critic_loss = ewma(self.critic_loss, cl)
        pl = self.update_policy(state, action, next_state, reward, done)
        self.policy_loss = ewma(self.policy_loss, pl)
        utils.interpolate_params(self.policy, self.target_policy, tau=self.config['tau'])
        utils.interpolate_params(self.critic, self.target_critic, tau=self.config['tau'])
        return self.policy_loss, self.critic_loss

    def update_critic(self, state, action, reward, next_state, done):
        self.critic.train()
        rollout_target = self._compute_rollout_target(state, action, reward, next_state, done)
        critic_loss = F.mse_loss(self.critic(state[:, 0, :], action[:, 0, :]).squeeze(), rollout_target).mean()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        return critic_loss.data.cpu().numpy()

    def update_policy(self, state, action, reward, next_state, done):
        self.policy.train()
        self.critic.train()
        state_batch = state.view(-1, *state.shape[2:])
        q_value = self.critic(state_batch, self.policy(state_batch))
        loss = -q_value.mean()
        self.opt_policy.zero_grad()
        loss.backward()
        self.opt_policy.step()
        return loss.data.cpu().numpy()

    def _compute_rollout_target(self, state, action, next_state, reward, done):
        with torch.no_grad():
            target_action = self.target_policy(next_state[:, -1, :])
            target_q = self.target_critic(next_state[:, -1, :], target_action)
        gamma_last = pow(self.config['gamma'], self.config['rollout_steps'])
        gammas = np.ones(self.config['rollout_steps'])
        gammas[1:] *= self.config['gamma']
        gammas = torch.FloatTensor(np.cumprod(gammas)).to(self.device).unsqueeze(0)
        # shift mask to protect final reward from zeroing
        shifted_done = torch.cat([torch.zeros_like(done[:, :1]), done[:, :-1]], dim=-1)
        rollout_reward = ((reward * gammas) * (1 - shifted_done)).sum(dim=1).squeeze()
        return (rollout_reward + gamma_last * target_q.squeeze() * (1 - done[:, -1])).detach()

    def _log_info(self, info, step):
        policy_loss, critic_loss = info
        print(f"Step {step}. Policy loss {policy_loss}, Critic loss {critic_loss}")
