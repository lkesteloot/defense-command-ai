import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()
print(observation, info)
print(env.action_space.shape)
print(env.observation_space.shape)

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    # print("action", action)
    observation, reward, terminated, truncated, info = env.step(action)
    # print("observation", observation, "reward", reward)

    episode_over = terminated or truncated

env.close()
