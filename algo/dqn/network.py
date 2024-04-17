import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Q_net(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Q_net, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = self.dim_observation
        act_dim = self.dim_action

        self.FC1 = nn.Linear(obs_dim, 64)
        self.FC2 = nn.Linear(64 + act_dim, 128)
        self.FC3 = nn.Linear(128, 32)
        self.FC4 = nn.Linear(32, 32)
        self.FC5 = nn.Linear(32, 5)

    # obs:batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], dim=1)
        result = F.relu(self.FC2(combined))
        result = self.FC4(F.relu(self.FC3(result)))
        return self.FC5(result)
