import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win')
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--training-num', type=int, default=64)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--win-rate', type=float, default=-30, help='the expected winning rate: Optimal policy can get 0.7')
    parser.add_argument('--watch', default=False, action='store_true', help='no training, watch the play of pre-trained models')
    parser.add_argument('--agent-id', type=int, default=2)
    parser.add_argument('--resume-path-0', type=str, default='', help='the path of agent pth file for resuming from a pre-trained agent 0')
    parser.add_argument('--resume-path-1', type=str, default='', help='the path of agent pth file for resuming from a pre-trained agent 1')
    parser.add_argument('--resume-path-2', type=str, default='', help='the path of agent pth file for resuming from a pre-trained agent 2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

num_agents = 3

def get_env(render_mode=None):
    """This functions is needed to provide callables for DummyVectorEnv"""
    return PettingZooEnv(simple_spread_v3.env(N=num_agents, max_cycles=25, local_ratio=0.5, continuous_actions=False,
                                              render_mode="rgb_array"))

def get_agents(
        args: argparse.Namespace = get_args(),
        agent_0: Optional[BasePolicy] = None,
        agent_1: Optional[BasePolicy] = None,
        agent_2: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = env.observation_space['observation'] if isinstance(env.observation_space, gym.spaces.Dict) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    if agent_0 is None:
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device
        ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_0 = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=args.gamma,
            action_space=env.action_space,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq
        )
        if args.resume_path_0:
            agent_0.load_state_dict(torch.load(args.resume_path_0))

    if agent_1 is None:
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device
        ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_1 = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=args.gamma,
            action_space=env.action_space,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq
        )
        if args.resume_path_1:
            agent_1.load_state_dict(torch.load(args.resume_path_1))
    
    if agent_2 is None:
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device
        ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_2 = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=args.gamma,
            action_space=env.action_space,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq
        )
        if args.resume_path_2:
            agent_2.load_state_dict(torch.load(args.resume_path_2))
    
    agents = [agent_0, agent_1, agent_2]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

def train_agent(
        args: argparse.Namespace = get_args(),
        agent_0: Optional[BasePolicy] = None,
        agent_1: Optional[BasePolicy] = None,
        agent_2: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None
) -> Tuple[dict, BasePolicy]:
    
    # ======== environment setup ========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup ========
    policy, optim, agents = get_agents(args, agent_0=agent_0, agent_1=agent_1, agent_2=agent_2, optim=optim)

    # ======== collector setup ========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup ========
    log_path = os.path.join(args.logdir, 'simple_spread', 'dqn')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training ========
    def save_best_fn(policy):
        if hasattr(args, 'model_save_path'):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(args.logdir, 'simple_spread', 'dqn', 'policy.pth')
            torch.save(policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path)
    
    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate
    
    def train_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)
    
    def test_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    
    def reward_metric(rews):
        return rews[:, args.agent_id - 1]
    
    # trainer
    result = OffpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric
    ).run()

    return result, policy.policies[agents[args.agent_id - 2]], policy.policies[agents[args.agent_id - 1]], policy.policies[agents[args.agent_id]]

# ======== a test function that tests a pre-trained agent ========
def watch(
        args: argparse.Namespace = get_args(),
        agent_0: Optional[BasePolicy] = None,
        agent_1: Optional[BasePolicy] = None,
        agent_2: Optional[BasePolicy] = None
) -> None:
    env = get_env(render_mode="human")
    env = DummyVectorEnv([lambda: env])
    policy, optim, agents = get_agents(args, agent_0=agent_0, agent_1=agent_1, agent_2=agent_2)
    policy.eval()
    policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")

# train the agent and watch its performance
args = get_args()
result, agent_0, agent_1, agent_2 = train_agent(args)
watch(args, agent_0, agent_1, agent_2)