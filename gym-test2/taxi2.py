
import os
os.environ['KERAS_BACKEND'] = 'torch'

import random
import numpy as np
import gymnasium as gym
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Embedding, Reshape
import tensorflow as tf
import time

env = gym.make("Taxi-v3", render_mode="ansi")
#print(env.metadata)
#print(dir(env))

observation_count = env.observation_space.n
action_count = env.action_space.n

class Step:
    def __init__(self, ):

class Playthrough:
    def __init__(self, steps, total_reward):
        self.steps = steps
        self.total_reward = total_reward

def make_one_hot(n, i):
    return [int(i == j) for j in range(n)]

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
    return np.array([
        make_one_hot(5, taxi_row) +
        make_one_hot(5, taxi_col) +
        make_one_hot(5, passenger_location) +
        make_one_hot(4, destination)])

def play(model, epsilon):
    steps = []

    observation, info = env.reset()
    #print(observation, info)

    episode_over = False
    total_reward = 0
    step = 0
    while not episode_over:
        with tf.profiler.experimental.Trace("train", step_num=step, _r=1):
            step += 1
            old_input_tensor = make_input_tensor(observation)

            action_mask = info["action_mask"]
            model_prediction = None
            if learn and random.random() < epsilon:
                # Explore.
                action = env.action_space.sample(action_mask)
            else:
                # Exploit.
                actions = model.predict(old_input_tensor, verbose=0)[0]
                model_prediction = actions
                print("Choosing from", actions)
                # print("Masks        ", action_mask)
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
            observation = new_observation

    return Playthrough(steps, total_reward)

def play(learn, model, alpha, gamma, epsilon):
    observation, info = env.reset()
    #print(observation, info)

    episode_over = False
    total_reward = 0
    step = 0
    while not episode_over:
        with tf.profiler.experimental.Trace("train", step_num=step, _r=1):
            step += 1
            old_input_tensor = make_input_tensor(observation)

            action_mask = info["action_mask"]
            model_prediction = None
            if learn and random.random() < epsilon:
                # Explore.
                action = env.action_space.sample(action_mask)
            else:
                # Exploit.
                actions = model.predict(old_input_tensor, verbose=0)[0]
                model_prediction = actions
                print("Choosing from", actions)
                # print("Masks        ", action_mask)
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

            if learn:
                if False:
                    W = model.get_weights()
                    for i in range(len(W)):
                        print(i, W[i].shape)
                before = time.perf_counter()
                new_input_tensor = make_input_tensor(new_observation)
                futures = model.predict(new_input_tensor, verbose=0)#.numpy()
                #print("timing 1", int((time.perf_counter() - before)*1000))
                best_future = np.max(futures)
                target = reward + gamma*best_future
                before = time.perf_counter()
                target_vector = model.predict(old_input_tensor, verbose=0)[0]
                #print("timing 2", int((time.perf_counter() - before)*1000))
                target_vector[action] = target
                target_vector = target_vector.reshape(-1, env.action_space.n)
                # print("target_vector", target_vector)
                before = time.perf_counter()
                model.fit(old_input_tensor, target_vector, epochs=1, verbose=0)
                #print("timing 3", int((time.perf_counter() - before)*1000))
                best_future = np.max(futures)
                target = reward + gamma*best_future

            episode_over = terminated or truncated
            observation = new_observation

    return total_reward

def main():
    print("Episode count [domain]\tTable")
    input_tensor_size = make_input_tensor(0).shape[1]
    for episode_count in range(1000, 2001, 100):
        model = Sequential([
            InputLayer( (input_tensor_size,) ),
            #Embedding(observation_count, 4),
            #Reshape((4,)),
            Dense(50, activation='relu'),
            Dense(50, activation='relu'),
            Dense(50, activation='relu'),
            Dense(action_count, activation='linear'),
        ])
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        #model.build(make_input_tensor(0).shape)
        print(model.summary())

        alpha_range = (0.9, 0.1)  # Learning rate.
        gamma = 0.99 # Discount factor.
        epsilon_range = (0.9, 0.0) # Explore (exploit). 

        for episode in range(episode_count):
            alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0])*episode/episode_count
            epsilon = epsilon_range[0] + (epsilon_range[1] - epsilon_range[0])*episode/episode_count

            tf.profiler.experimental.start('logdir')
            total_reward = play(True, model, alpha, gamma, epsilon)
            tf.profiler.experimental.stop()
            print("learn", episode, "/", episode_count, "=", total_reward)
            return

        play_count = 1
        total_reward_total = 0
        for episode in range(play_count):
            total_reward = play(False, model, None, None, None)
            print("play", episode, "/", play_count, "=", total_reward)
            total_reward_total += total_reward
        print(episode_count, total_reward_total / play_count)

    env.close()

try:
    main()
except KeyboardInterrupt:
    pass
