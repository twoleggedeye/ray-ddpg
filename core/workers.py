from collections import deque
import time
import numpy as np
import ray
import copy
import torch
from . import utils


class RemoteAgent(object):
    def __init__(self,
                 policy_prototype,
                 policy_kwargs,
                 make_env_fn,
                 params_server_handle,
                 device='cuda'):
        self.device = device
        self.env = make_env_fn()
        self.params_server = params_server_handle
        self.policy = policy_prototype(**policy_kwargs).to(self.device)
        self.param_noise = False

    def _update_params(self):
        state_dict = copy.deepcopy(ray.get(self.params_server.pull.remote()))
        if not self.param_noise:
            state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
        else:
            new_dict = {}
            for k, v in state_dict.items():
                if not isinstance(v, torch.LongTensor):
                    v += torch.zeros_like(v).data.normal_() * self.param_noise_sigma
                new_dict[k] = v
            state_dict = new_dict
        self.policy.load_state_dict(state_dict)

    @classmethod
    def as_remote(cls, num_gpus, num_cpus):
        return ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)(cls)


class Evaluator(RemoteAgent):
    def __init__(self,
                 policy_prototype,
                 policy_kwargs,
                 make_env_fn,
                 params_server_handle,
                 config,
                 n_runs=10,
                 device='cuda'):
        super().__init__(policy_prototype, policy_kwargs, make_env_fn,
                         params_server_handle, device)
        self.config = config
        self.n_runs = n_runs
        self._evals_count = 0

    def _eval_once(self):
        self._update_params()
        self.policy.eval()
        total_reward = 0
        done = False
        state = self.env.reset()
        while not done:
            with torch.no_grad():
                if self.config['render']:
                    self.env.render()
                state_torch = torch.FloatTensor(state).to(self.device)
                action = self.policy.get_action(state_torch, exploration=False)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
        self._evals_count += 1
        return total_reward

    def evaluate(self):
        result = np.mean([self._eval_once() for _ in range(self.n_runs)])
        self._log_result(result)

    def _log_result(self, result):
        print(f"Validation. Mean reward {result}")


class Runner(RemoteAgent):
    def __init__(self,
                 policy_prototype,
                 policy_kwargs,
                 make_env_fn,
                 params_server_handle,
                 storage_handle,
                 config,
                 device='cuda',
                 param_noise=True,
                 param_noise_sigma=0.2):
        super().__init__(policy_prototype, policy_kwargs, make_env_fn,
                         params_server_handle, device)
        self.config = config
        self.storage = storage_handle
        self.param_noise = param_noise
        self.param_noise_sigma = param_noise_sigma

    def run_once(self):
        self._update_params()
        self.policy.eval()
        self._run_once()

    def run_forever(self):
        self.policy.eval()
        while True:
            self._update_params()
            self._run_once()

    def _run_once(self):
        states = deque()
        actions = deque()
        next_states = deque()
        rewards = deque()
        dones = deque()
        deqs = (states, actions, rewards, next_states, dones)
        done = False
        state = self.env.reset()
        while not done:
            with torch.no_grad():
                state_torch = torch.FloatTensor(state).to(self.device)
                action = self.policy.get_action(state_torch)
                next_state, reward, done, _ = self.env.step(action)
                transition = (state, action, reward, next_state, done)
                if len(states) < self.config['rollout_steps']:
                    for store, el in zip(deqs, transition):
                        store.append(el)
                else:
                    ray.wait([self.storage.add.remote(*self._make_np_transitions(*deqs))])
                    for store, el in zip(deqs, transition):
                        store.popleft()
                        store.append(el)
                state = next_state

    def _make_np_transitions(self, states, actions, rewards, next_states, dones):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones


class OffPolicyTrainer(object):
    def __init__(self, config):
        self.config = config

    def train(self):
        step = 0
        current_buffer_sz = ray.get(self.storage.get_len.remote())
        target_size = self.config['batch_size'] * 10
        for runner in self.runners:
            runner.run_forever.remote()
        while current_buffer_sz < target_size:
            print(f'Waiting buffer to fill. Current size {current_buffer_sz}',
                  f'Target sz {target_size}')
            time.sleep(1)
            current_buffer_sz = ray.get(self.storage.get_len.remote())

        while step < self.config['max_steps']:
            info = self._make_updates()
            if step % self.config['param_update_steps'] == 0 and step != 0:
                weight_id = ray.put(utils.get_cpu_state_dict(self.policy))
                ray.wait([self.params_server.push.remote(weight_id)])
                self.evaluator.evaluate.remote()
                self._log_info(info, step)
                self._adjust_schedule()
            step += 1

    def _make_updates(self):
        raise NotImplementedError()

    def _log_info(self, info, step):
        pass

    def _adjust_schedule(self):
        pass

    def _sample_batch(self, batch_size):
        transition = ray.get(self.storage.sample.remote(self.config['batch_size']))
        transition = [torch.from_numpy(t.astype(np.float32)).to(self.device) for t in transition]
        return transition

















