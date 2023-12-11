from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

from pettingzoo.classic import rps_v2
from pettingzoo.mpe import simple_spread_v3

if __name__=="__main__":
    # step 1: load the PettingZoo environment
    env = rps_v2.env(render_mode="human")

    # step 2: wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # step 3: define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

    # step 4: convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # step 5: construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # step 6: execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=0.2)


# if __name__=="__main__":
#     # step 1: load the PettingZoo environment
#     env = simple_spread_v3.env(render_mode="human")

#     # step 2: wrap the environment for Tianshou interfacing
#     env = PettingZooEnv(env)

#     # step 3: define policies for each agent
#     policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy(), RandomPolicy()], env)

#     # step 4: convert the env to vector format
#     env = DummyVectorEnv([lambda: env])

#     # step 5: construct the Collector, which interfaces the policies with the vectorised environment
#     collector = Collector(policies, env)

#     # step 6: execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
#     result = collector.collect(n_episode=1, render=0.1)