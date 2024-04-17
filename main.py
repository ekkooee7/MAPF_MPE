import argparse
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np
import torch,os

from algo.bicnet.bicnet_agent import BiCNet
from algo.commnet.commnet_agent import CommNet
from algo.maddpg.maddpg_agent import MADDPG
from algo.ddpg.ddpg_agent import DDPG

from algo.normalized_env import ActionNormalizedEnv, ObsEnv, reward_from_state

from algo.utils import *
from copy import deepcopy

import pettingzoo.mpe.simpla_pathfinding.simple_pathfinding as mpe




# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#
#
#     if termination or truncation:
#         action = None
#     else:
#         action = env.action_space(agent).sample() # this is where you would insert your policy
#
#     env.step(action)
# env.close()



def main(args):
    if args.render_mode=="human":
        env = mpe.env(render_mode="human")
    else:
        env = mpe.env(render_mode="rgb_array")
    env.reset()

    n_agents = env.env.num_agents
    n_actions = env.world.dim_p
    n_actions = 2
    # env = ActionNormalizedEnv(env)
    # env = ObsEnv(env)
    n_states = 4

    torch.manual_seed(args.seed)

    # if args.tensorboard and args.mode == "train":
    #     writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir)

    model = DDPG(n_states, n_actions, n_agents, args)
    # print(model.n_agents)



    if args.tensorboard and args.mode == "train":
        writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir)

    if args.mode == "train":
        episode = 0
        total_step = 0
        while episode < args.max_episodes:
            episode += 1
            step = 0
            accum_reward = 0
            rewardA = 0
            observation_all = []
            env.reset()
            for agent in env.agent_iter():
                step += 1
                observation, reward, termination, truncation, info = env.last()

                observation_all.append(observation)
                state = [observation]

                if termination or truncation:
                    action = None
                    env.step(action)
                else:
                    # take action
                    action, act = model.choose_action(state, noisy=True)
                    # print(action)
                    action = np.squeeze(action)

                    # step
                    a = np.float32(0)
                    action = np.append(a, action)
                    env.step(action)
                    # get next state info
                    next_state, reward, termination, truncation, info = env.last()

                    next_state = [next_state]

                    reward = np.array([reward])
                    rewardA += reward

                    obs = torch.from_numpy(np.stack(state)).float().to(device)
                    obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)
                    next_obs = obs_

                    # if step != args.episode_length - 1:
                    #     next_obs = obs_
                    # else:
                    #     next_obs = None
                    rw_tensor = torch.FloatTensor(reward).to(device)
                    # ac_tensor = torch.FloatTensor(act).to(device)
                    ac_tensor = act

                    # maddpg memory
                    model.memory.push(obs, ac_tensor, next_obs, rw_tensor)

                    state = next_state

                    if termination or truncation:
                        print(action)
                        c_loss, a_loss = model.update_single(episode)
                        if args.tensorboard:
                            # writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward)
                            writer.add_scalar(tag='agent/reward_0', global_step=episode, scalar_value=rewardA / step)
                            if c_loss and a_loss:
                                writer.add_scalars('agent/loss', global_step=episode,
                                                   tag_scalar_dict={'actor': a_loss, 'critic': c_loss})

                        if c_loss and a_loss:
                            print("episode: ", episode, " a_loss %3.2f c_loss %3.2f reward %3.2f" % (a_loss, c_loss, rewardA / step), end='')
                            print()

                        if episode % args.save_interval == 0 and args.mode == "train":
                            model.save_model(episode)

    if args.mode == "eval":
        model.load_model()
        episode = 0
        total_step = 0
        while episode < args.max_episodes:
            episode += 1
            step = 0
            accum_reward = 0
            rewardA = 0
            # action for every agent
            observation_all = []
            env.reset()
            for agent in env.agent_iter():
                step += 1
                observation, reward, termination, truncation, info = env.last()
                observation_all.append(observation)
                if termination or truncation:
                    action = None
                    env.step(action)
                else:
                    # this is where you would insert your policy
                    state = [observation]
                    action, act = model.choose_action(state, noisy=False)
                    # print(action, act)
                    action = np.squeeze(action)
                    a = np.float32(0)
                    action = np.append(a, action)
                    # print(action)
                    env.step(action)
                    next_state, reward, termination, truncation, info = env.last()
                    state = [next_state]
                    reward = np.array(reward)
                    # env.render()

                    rewardA += reward

                # env.step(action)
                print(rewardA)

            episode += 1
            step = 0
            accum_reward = 0
            rewardA = 0
            rewardB = 0
            rewardC = 0

        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="", type=str)
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--algo', default="ddpg", type=str, help="commnet/bicnet/maddpg/ddpg")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--memory_length', default=int(2e6), type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    # parser.add_argument('--ou_theta', default=0.15, type=float)
    # parser.add_argument('--ou_mu', default=0.0, type=float)
    # parser.add_argument('--ou_sigma', default=1, type=float)
    # parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=5000, type=int)
    parser.add_argument("--model_episode", default=15000, type=int)
    parser.add_argument('--episode_before_train', default=256, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d'))

    parser.add_argument('--render_mode', default="rgb_array", type=str, help="rgb_array/human")
    parser.add_argument('--soft_update_interval', default=50, type=int)


    args = parser.parse_args()
    main(args)