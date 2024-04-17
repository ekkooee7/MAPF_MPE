import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = self.dim_observation
        act_dim = self.dim_action

        self.FC1 = nn.Linear(obs_dim + act_dim, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 1)
        # self.FC4 = nn.Linear(32, 32)
        # self.FC5 = nn.Linear(32, 1)

    # obs:batch_size * obs_dim
    def forward(self, obs, acts):
        combined = th.cat([obs, acts], dim=1)
        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        return result


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        # print('model.dim_action',dim_action)
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 64)
        self.FC2 = nn.Linear(64, 32)
        # self.FC3 = nn.Linear(64, 32)
        self.FC4 = nn.Linear(32, dim_action)
        self.sfm = nn.Softmax(dim=-1)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        # result = F.relu(self.FC3(result))
        result = self.FC4(result)
        result = F.tanh(result)

        # 将输出归一化到0-1

        return result
