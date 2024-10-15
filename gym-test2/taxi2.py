
import sys
import os
os.environ['KERAS_BACKEND'] = 'torch'

import random
import numpy as np
import gymnasium as gym
from keras.models import Sequential
from keras.layers import InputLayer, Dense
import tensorflow as tf
import time

env = gym.make("Taxi-v3", render_mode="ansi")
#print(env.metadata)
#print(dir(env))

observation_count = env.observation_space.n
action_count = env.action_space.n

class Step:
    def __init__(self, observation, action, reward, total_reward, episode_over, new_observation):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.total_reward = total_reward
        self.episode_over = episode_over
        self.new_observation = new_observation

class Playthrough:
    def __init__(self, steps, total_reward):
        self.steps = steps
        self.total_reward = total_reward

def make_one_hot(n, i):
    return np.array([int(i == j) for j in range(n)])

def make_input_tensor(observation):
    # For Embedding:
    ## return np.array([[observation]])

    # https://gymnasium.farama.org/environments/toy_text/taxi/
    # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    destination = observation % 4
    observation //= 4
    passenger_location = observation % 5
    observation //= 5
    taxi_col = observation % 5
    taxi_row = observation // 5

    # Integer values:
    ## return np.array([[taxi_row, taxi_rol, passenger_location, destination]])

    # Several one-hot arrays:
    return np.concatenate( (
        make_one_hot(5, taxi_row),
        make_one_hot(5, taxi_col),
        make_one_hot(5, passenger_location),
        make_one_hot(4, destination)
    ) )

def play(model, epsilon):
    steps = []

    observation, info = env.reset()
    #print(observation, info)

    episode_over = False
    total_reward = 0
    while not episode_over:
        old_input_tensor = make_input_tensor(observation)

        action_mask = info["action_mask"]
        if random.random() < epsilon:
            # Explore.
            action = env.action_space.sample(action_mask)
        else:
            # Exploit.
            actions = model.predict(np.array([old_input_tensor]), verbose=0)[0]
            #print("Choosing from", actions)
            #print("Masks        ", action_mask)
            action = max((actions[potential_action], potential_action)
                         for potential_action in range(action_count)
                         if action_mask[potential_action])[1]

        new_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if False:
            print("action", action,
                  "new observation", new_observation,
                  "reward", reward,
                  "total reward", total_reward)

            print(env.render())

        episode_over = terminated or truncated
        steps.append(Step(observation, action, reward, total_reward, episode_over, new_observation))

        observation = new_observation

    return Playthrough(steps, total_reward)

def gather_steps(model, epsilon, min_steps, min_reward):
    steps = []

    print("Gathering steps", "epsilon =", epsilon)

    game_count = 0
    used_count = 0
    total_reward = 0
    total_used_reward = 0
    while len(steps) < min_steps:
        playthrough = play(model, epsilon)
        game_count += 1
        total_reward += playthrough.total_reward
        if playthrough.total_reward >= min_reward:
            steps.extend(playthrough.steps)
            used_count += 1
            total_used_reward += playthrough.total_reward

    print("    Used %d of %d games, %d reward, %d used reward" % (used_count, game_count,
                                                                  total_reward/game_count,
                                                                  total_used_reward/used_count))

    return steps

def learn(model, steps, gamma):
    print("Learning", "gamma =", gamma)

    old_input_tensor = np.array([make_input_tensor(step.observation) for step in steps])
    new_input_tensor = np.array([make_input_tensor(step.new_observation) for step in steps])
    action = np.array([step.action for step in steps])
    reward = np.array([step.reward for step in steps])

    before = time.perf_counter()
    futures = model.predict(new_input_tensor, verbose=0)
    print("    timing 1", int((time.perf_counter() - before)*1000), "futures", futures.shape)

    # Figure out target.
    best_future = np.max(futures, axis=1)
    target = reward + gamma*best_future

    before = time.perf_counter()
    target_vector = model.predict(old_input_tensor, verbose=0)
    print("    timing 2", int((time.perf_counter() - before)*1000))

    target_vector[np.arange(target_vector.shape[0]),action] = target

    before = time.perf_counter()
    model.fit(old_input_tensor, target_vector, epochs=10, verbose=0)
    print("    timing 3", int((time.perf_counter() - before)*1000))

def main():
    print("Episode count [domain]\tTable")
    input_tensor_size = make_input_tensor(0).shape[-1]
    for episode_count in range(10, 101, 10):
        model = Sequential([
            InputLayer( (input_tensor_size,) ),
            Dense(50, activation='relu'),
            #Dense(50, activation='relu'),
            #Dense(50, activation='relu'),
            Dense(action_count, activation='linear'),
        ])
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        #model.compile(Adam(lr=1e-2), loss='mae')
        #model.build(make_input_tensor(0).shape)
        model.summary()

        alpha_range = (0.9, 0.1)  # Learning rate.
        gamma = 0.99 # Discount factor.
        epsilon_range = (1.0, 0.0) # Explore (exploit). 

        for episode in range(episode_count):
            alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0])*episode/episode_count
            epsilon = epsilon_range[0] + (epsilon_range[1] - epsilon_range[0])*episode/episode_count

            steps = gather_steps(model, epsilon, 1000, -199)
            learn(model, steps, gamma)

            #total_reward = play(True, model, alpha, gamma, epsilon)
            #print("learn", episode, "/", episode_count, "=", total_reward)
            #return

        play_count = 10
        total_reward_total = 0
        for episode in range(play_count):
            playthrough = play(model, 0)
            total_reward = playthrough.total_reward
            print("play", episode, "/", play_count, "=", total_reward)
            total_reward_total += total_reward
        print(episode_count, total_reward_total / play_count)

    env.close()

try:
    main()
except KeyboardInterrupt:
    pass
