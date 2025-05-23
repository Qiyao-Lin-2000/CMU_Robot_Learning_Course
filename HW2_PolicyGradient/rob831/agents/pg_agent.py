import numpy as np

from rob831.agents.base_agent import BaseAgent
from rob831.policies.MLP_policy import MLPPolicyPG
from rob831.infrastructure.replay_buffer import ReplayBuffer

from rob831.infrastructure.utils import normalize, unnormalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):
        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)
    
        train_log = self.actor.update(
            observations, 
            actions, 
            advantages,
            q_values=q_values if self.nn_baseline else None
        )
        return train_log


    def calculate_q_vals(self, rewards_list):
        all_q_values = []
        for rewards in rewards_list:
          if not self.reward_to_go:
            total_return = self._discounted_return(rewards)
            q_values = np.full(len(rewards), total_return)
          else:
            q_values = self._discounted_cumsum(rewards)
          all_q_values.append(q_values)
        return np.concatenate(all_q_values)


    def estimate_advantage(self, obs, rewards_list, q_values, terminals):
      if self.nn_baseline:
        values = self.actor.run_baseline_prediction(obs)

        if self.gae_lambda is not None:
          advantages = np.zeros_like(q_values)
          start_idx = 0

          for traj_idx in range(len(rewards_list)):
            traj_rewards = rewards_list[traj_idx]
            traj_length = len(traj_rewards)
            end_idx = start_idx + traj_length

            traj_values = values[start_idx:end_idx]
            traj_terminals = terminals[start_idx:end_idx]

            extended_values = np.append(traj_values, 0)
            last_advantage = 0

            for t in reversed(range(traj_length)):
              if traj_terminals[t]:
                delta = traj_rewards[t] - extended_values[t]
                last_advantage = delta
              else:
                delta = traj_rewards[t] + self.gamma * extended_values[t+1] - extended_values[t]
                last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
              advantages[start_idx + t] = last_advantage

            start_idx = end_idx
          assert start_idx == len(obs), "NotMatch"
        else:
          advantages = q_values - values
      else:
        if self.gae_lambda is not None:
          raise ValueError("GAE baseline (nn_baseline=True)")
        advantages = q_values.copy()

      if self.standardize_advantages:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

      return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        discounted_sum = 0
        for t, r in enumerate(rewards):
            discounted_sum += (self.gamma ** t) * r
        return np.full(len(rewards), discounted_sum)

    def _discounted_cumsum(self, rewards):
        discounted_cumsums = []
        running_sum = 0
        for r in reversed(rewards):
            running_sum = r + self.gamma * running_sum
            discounted_cumsums.insert(0, running_sum)
        return np.array(discounted_cumsums)