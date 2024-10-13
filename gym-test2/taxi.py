
import gymnasium as gym
import numpy as np
import random

env = gym.make("Taxi-v3", render_mode="ansi")
#print(env.metadata)
#print(dir(env))

observation_count = env.observation_space.n
action_count = env.action_space.n

def play(learn, Q, alpha, gamma, epsilon):
    observation, info = env.reset()
    #print(observation, info)

    episode_over = False
    total_reward = 0
    while not episode_over:
        #print("----------------------------------------------------------------------------------")

        action_mask = info["action_mask"]
        if learn and random.random() < epsilon:
            # Explore.
            action = env.action_space.sample(action_mask)
        else:
            # Exploit.
            action = max((Q[observation][potential_action], potential_action)
                         for potential_action in range(action_count)
                         if action_mask[potential_action])[1]

        new_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if False and episode == episode_count - 1:
            print("action", action,
                  "new observation", new_observation,
                  "reward", reward,
                  "total reward", total_reward)

            print(env.render())

        if learn:
            best_future = max(Q[new_observation][next_action] for next_action in range(action_count))
            target = reward + gamma*best_future
            Q[observation][action] = (1 - alpha)*Q[observation][action] + alpha*target

        episode_over = terminated or truncated
        observation = new_observation

    return total_reward

def main():
    print("Episode count [domain]\tTable")
    for episode_count in range(100, 1001, 100):
        Q = np.zeros([observation_count, action_count])
        alpha_range = (0.9, 0.1)  # Learning rate.
        gamma = 0.99 # Discount factor.
        epsilon_range = (0.9, 0.0) # Explore (exploit). 

        for episode in range(episode_count):
            alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0])*episode/episode_count
            epsilon = epsilon_range[0] + (epsilon_range[1] - epsilon_range[0])*episode/episode_count

            total_reward = play(True, Q, alpha, gamma, epsilon)
            # print("episode", episode, "total reward", total_reward)

        play_count = 1000
        total_reward_total = 0
        for episode in range(play_count):
            total_reward = play(False, Q, None, None, None)
            total_reward_total += total_reward
        print(episode_count, total_reward_total / play_count)

    env.close()

main()
