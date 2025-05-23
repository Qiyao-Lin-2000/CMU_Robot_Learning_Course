import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        observation = ptu.from_numpy(observation.astype(np.float32))
        dist = self(observation)
        action = dist.sample()
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    def forward(self, observation: torch.FloatTensor) -> Any:
        # TODO: Construct and return the appropriate distribution
        if self.discrete:
            logits = self.logits_na(observation)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(observation)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)

    def update(self, observations, actions, adv_n=None, acs_labels_na=None, qvals=None):
        obs_tensor = ptu.from_numpy(observations).float()
        acts_tensor = ptu.from_numpy(actions).float()

        self.optimizer.zero_grad()
        dist = self(obs_tensor)

        if self.discrete:
            acts_tensor = acts_tensor.long()
            log_probs = dist.log_prob(acts_tensor.squeeze(-1))
        else:
            log_probs = dist.log_prob(acts_tensor)
            if len(log_probs.shape) > 1:
                log_probs = log_probs.sum(dim=1)
        
        loss = -log_probs.mean()
        loss.backward()
        self.optimizer.step()

        return {'Training Loss': ptu.to_numpy(loss)}