from algo.dqn.network import Q_net
import torch
from copy import deepcopy
from torch.optim import Adam
from algo.memory import ReplayMemory, Experience
from algo.random_process import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import torch.nn.functional as F
import os
import torch.nn as nn
import numpy as np
from algo.utils import device
scale_reward = 0.01


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class DQN:
    def __init__(self, dim_obs, dim_act, n_agents, args):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics = []
        # self.Q = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        self.Q = Q_net(dim_obs, dim_act)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act

        # self.actors_target = deepcopy(self.actors)
        self.Q_target = deepcopy(self.Q)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = args.episode_before_train

        self.GAMMA = args.gamma
        self.tau = args.tau

        self.epsilon = 0.9

        self.var = [args.ou_sigma for i in range(n_agents)]

        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=args.c_lr) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=args.a_lr) for x in self.actors]

        self.soft_update_interval = args.soft_update_interval

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def load_model(self):
        if self.args.model_episode:
            path_flag = True
            for idx in range(self.n_agents):
                path_flag = path_flag \
                            and (os.path.exists("trained_model/maddpg/actor["+ str(idx) + "]_"
                                                +str(self.args.model_episode)+".pth")) \
                            and (os.path.exists("trained_model/maddpg/critic["+ str(idx) + "]_"
                                                +str(self.args.model_episode)+".pth"))

            if path_flag:
                print("load model!")
                for idx in range(self.n_agents):
                    actor = torch.load("trained_model/maddpg/actor["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                    critic = torch.load("trained_model/maddpg/critic["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                    self.actors[idx].load_state_dict(actor.state_dict())
                    self.critics[idx].load_state_dict(critic.state_dict())

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    def save_model(self, episode):
        if not os.path.exists("./trained_model/" + str(self.args.algo) + "/"):
            os.mkdir("./trained_model/" + str(self.args.algo) + "/")
        for i in range(self.n_agents):
            torch.save(self.actors[i],
                       'trained_model/ddpg/actor[' + str(i) + ']' + '_' + str(episode) + '.pth')
            torch.save(self.critics[i],
                       'trained_model/ddpg/critic[' + str(i) + ']' + '_' + str(episode) + '.pth')

    def update_single(self, i_episode):
        self.train_num = i_episode
        if self.train_num <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        non_final_mask = BoolTensor(list(map(lambda s: s is not None, batch.next_states)))
        # state_batch: batch_size x n_agents x dim_obs
        states = torch.stack(batch.states).type(FloatTensor)
        actions = torch.stack(batch.actions).type(FloatTensor)
        rewards = torch.stack(batch.rewards).type(FloatTensor)
        next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
        dones = 0

        states = states.view(self.batch_size, -1)
        actions = actions.view(self.batch_size, -1)
        rewards = rewards.view(self.batch_size, -1)
        next_states = next_states.view(self.batch_size, -1)

        # states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        # rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        self.target_critic = self.critics_target[0]
        self.target_actor = self.actors_target[0]
        self.critic = self.critics[0]
        self.actor = self.actors[0]

        # update critic
        self.critic.zero_grad()
        self.actor.zero_grad()
        self.critic_optimizer[0].zero_grad()
        self.actor_optimizer[0].zero_grad()

        action_miu_target = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, action_miu_target)
        q_targets = rewards + self.GAMMA * next_q_values * (1 - dones)
        q_current = self.critic(states, actions)
        if self.train_num % 100 == 0:
            print("next_q_values", next_q_values[0:5])
            print("current_q_values", q_current[0:5])
            print("rewards", rewards[0:5])
            print("q_targets", q_targets[0:5])

        critic_loss = torch.mean(F.mse_loss(q_current, q_targets))
        self.critic_optimizer[0].zero_grad()
        critic_loss.backward()
        self.critic_optimizer[0].step()

        # update actor
        self.critic.zero_grad()
        self.actor.zero_grad()
        self.critic_optimizer[0].zero_grad()
        self.actor_optimizer[0].zero_grad()

        action_miu = self.actor(states)
        gradient = self.critic(states, action_miu)
        actor_loss = -torch.mean(gradient)
        self.actor_optimizer[0].zero_grad()
        actor_loss.backward()
        self.actor_optimizer[0].step()
        # for name, parms in self.actor.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #           ' -->grad_value:', parms.grad)

        soft_update(self.target_actor, self.actor, self.tau)  # 软更新策略网络
        soft_update(self.target_critic, self.critic, self.tau)  # 软更新价值网络

        return critic_loss, actor_loss

    def choose_action(self, state, noisy=True):
        obs = torch.from_numpy(np.stack(state)).float().to(device)
        actions = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        for i in range(self.n_agents):
            sb = obs[i].detach()
            self.epsilon = self.epsilon * 0.999
            if np.random.random() < self.epsilon:
                act = np.random.randint(self.action_dim)
            else:
                act = self.Q(sb).argmax().item()

            actions[i, :] = act
        self.steps_done += 1
        return actions.data.cpu().numpy()
