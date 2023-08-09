import gymnasium as gym


# creating the blackjack environment
env = gym.make("Blackjack-v1", sab=True)
# reset the environment to get the first observation
done = False
observation, info = env.reset()

# sample a random action from all valid actions
action = env.action_space.sample()

# execute the action in our environment and receive infos from the environment
observation, reward, terminated, truncated, info = env.step(action)
