
import sys
import os
os.environ['KERAS_BACKEND'] = 'torch'
import pickle

import random
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam
import time
from collections import deque
from boxaverage import BoxAverage
from ml import DQN

LEARNING_RATE=0.001 # Default is 0.001.
GAMMA = 0.99 # Discount factor.
GAMMA = 0.8
MINIBATCH_SIZE = 32 # How many replays to sample when back-propagating.
DOUBLE_DQN = True # Use Double DQN.
TARGET_MODEL_UPDATE_PERIOD = 1000 # How often to update the target model.
PREDICTIONS_FILENAME = "predictions.txt"
LOSS_FILENAME = "loss.txt"
MODEL_WEIGHTS_FILENAME = "model.weights.h5"
TARGET_MODEL_WEIGHTS_FILENAME = "target_model.weights.h5"

# Model:
# The player has a position 0 to 10, initially random.
# If the player gets to position 5 to 6 inclusive, they get a point
# and the game restarts.
# Each move they can move left or right.

ACTIONS = [
    "<",
    ">",
]

class GameState:
    def __init__(self, position):
        self.position = float(position)

    def score(self):
        return int(self.win())

    def game_over(self):
        return self.win() or self.lose()

    def win(self):
        return self.position >= 5 and self.position <= 6

    def lose(self):
        return self.position < 0 or self.position >= 10

    def input_tensor(self):
        return np.array([
            (self.position - 5.5)/5,
        ])

    def __repr__(self):
        return "%.1f" % (self.position,)

class Transition:
    def __init__(self, previous_state, action, next_state):
        self.previous_state = previous_state
        self.action = action
        self.next_state = next_state

    def compute_reward(self):
        score_diff = self.next_state.score() - self.previous_state.score()
        return max(score_diff, 0)

    def __repr__(self):
        return "%s %s %s (%.1f)" % (self.previous_state, ACTIONS[self.action],
                                    self.next_state, self.compute_reward())

def dump_model_profile(model, filename):
    with open(filename, "w") as f:
        f.write("Pos [domain]\t<\t>\n")
        pos = 0
        while pos < 10:
            game_state = GameState(pos)
            input_tensor = game_state.input_tensor()
            actions = model.predict(input_tensor, verbose=0)
            f.write("%g %g %g\n" % (game_state.position, actions[0][0], actions[0][1]))
            pos += 0.1

class DQN:
    def __init__(self, input_size, output_size, replay_buffer, learning_rate,
                 minibatch_size, double_dqn, gamma):

        self.input_size = input_size
        self.output_size = output_size
        self.replay_buffer = replay_buffer
        self.minibatch_size = minibatch_size
        self.double_dqn = double_dqn
        self.gamma = gamma

        # Average of input and output.
        middle_layer_size = (input_size + output_size) // 2
        middle_layer_size = input_size # Guaranteed not lossy.
        middle_layer_size = 8

        # relu = rectified linear unit = max(0, x)
        ACTIVATION = "relu"
        #ACTIVATION = "sigmoid"

        # Create neural net.
        self.model = Sequential([
            InputLayer( (input_size,) ),
            Dense(middle_layer_size, activation=ACTIVATION),
            Dense(middle_layer_size, activation=ACTIVATION),
            Dense(middle_layer_size, activation=ACTIVATION),
            Dense(middle_layer_size, activation=ACTIVATION),
            Dense(middle_layer_size, activation=ACTIVATION),
            Dense(middle_layer_size, activation=ACTIVATION),
            #Dense(middle_layer_size, activation=ACTIVATION, kernel_initializer="zeros", bias_initializer="zeros"),
            Dense(output_size, activation='linear'),
            #Dense(output_size, activation='linear', kernel_initializer="zeros", bias_initializer="zeros"),
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])
        self.model.summary()

        # Make a copy of the model to use as the target model.
        self.target_model = clone_model(self.model)
        self.take_target_model_snapshot()

    def save_weights(self):
        self.model.save_weights(MODEL_WEIGHTS_FILENAME)
        self.target_model.save_weights(TARGET_MODEL_WEIGHTS_FILENAME)

    def load_weights(self):
        self.model.load_weights(MODEL_WEIGHTS_FILENAME)
        self.target_model.load_weights(TARGET_MODEL_WEIGHTS_FILENAME)

    def update(self):
        # Sample minibatch of transitions from replay buffer.
        # TODO: Sample proportionally to loss.
        ## transitions = self.replay_buffer.sample(self.minibatch_size)
        transitions = random.sample(self.replay_buffer, min(self.minibatch_size, len(self.replay_buffer)))

        # Compute predictions of batch.
        previous_input_tensors = np.array([transition.previous_state.input_tensor()
                                           for transition in transitions])
        predictions = self.model.predict(previous_input_tensors, verbose=0)

        # Compute targets of batch.
        rewards = np.array([transition.compute_reward() for transition in transitions])
        # If end of game, then it's just the reward. Otherwise it's the reward
        # plus expected value of best action using target_model.
        # TODO this complicates things a lot and nearly all transitions aren't
        # game over, so just do the full thing and zero out the "best_futures"
        # indices for the game over ones.
        live_game_indices = np.array([i for i in range(len(transitions))
                                      if not transitions[i].next_state.game_over()])
        #print("live_game_indices", live_game_indices)
        next_input_tensors = np.array([transitions[i].next_state.input_tensor() for i in live_game_indices])
        #print("next_input_tensors", next_input_tensors[0])
        if self.double_dqn:
            # Double DQN
            live_futures = self.model.predict(next_input_tensors, verbose=0)
            #print("live_futures", live_futures[0])
            best_live_future_indices = np.argmax(live_futures, axis=1)
            live_target_futures = self.target_model.predict(next_input_tensors, verbose=0)
            best_live_futures = live_target_futures[
                    np.arange(live_target_futures.shape[0]),best_live_future_indices]
            #print("best_live_futures", best_live_futures)
            best_futures = np.zeros(len(transitions))
            best_futures[live_game_indices] = best_live_futures
        else:
            # DQN
            live_futures = self.target_model.predict(next_input_tensors, verbose=0)
            #print("live_futures", live_futures[0])
            best_live_futures = np.max(live_futures, axis=1)
            #print("best_live_futures", best_live_futures)
            best_futures = np.zeros(len(transitions))
            best_futures[live_game_indices] = best_live_futures

            if False:
                print()
                print()
                print()
                print(live_futures)
                print(best_live_futures)
                print()
                print()
                print()
        targets = rewards + self.gamma*best_futures

        # Update targets for the actions we actually took.
        actions = np.array([transition.action for transition in transitions])
        fixed_predictions = predictions.copy()
        fixed_predictions[np.arange(fixed_predictions.shape[0]),actions] = targets

        # Perform gradient descent.
        if False:
            print()
            print("fit")
            print(np.hstack((previous_input_tensors, fixed_predictions)))
            print()
        history = self.model.fit(previous_input_tensors, fixed_predictions, epochs=1, verbose=0)
        return history.history["loss"][0], history.history["mae"][0], predictions

    def take_target_model_snapshot(self):
        self.target_model.set_weights(self.model.get_weights())

    def print_weights(self):
        for i, layer in enumerate(self.model.layers):
            print("Weights for layer %d:" % (i + 1,))
            weights, biases = layer.get_weights()
            print(weights)
            print(biases)
            print()

def make_transitions():
    print("Generating transitions...")
    transitions = []

    while len(transitions) < 100_000:
        # New game.
        position = random.random()*10
        game_state = GameState(position)
        while not game_state.game_over():
            action = random.randrange(len(ACTIONS))
            if action == 0:
                position -= 1
            elif action == 1:
                position += 1
            next_game_state = GameState(position)
            transition = Transition(game_state, action, next_game_state)
            transitions.append(transition)
            game_state = next_game_state

    if False:
        for t in random.sample(transitions, 100):
            print(t)

    return transitions

def learn(transitions):
    print("Learning...")
    input_size = GameState(0).input_tensor().shape[0]
    action_count = len(ACTIONS)

    dqn = DQN(input_size, action_count, transitions, LEARNING_RATE,
              MINIBATCH_SIZE, DOUBLE_DQN, GAMMA)

    loss_average = BoxAverage(1000)
    learn_step = 0
    print("Writing loss data to \"%s\"" % (LOSS_FILENAME,))
    with open(PREDICTIONS_FILENAME, "w") as predictions_file, \
            open(LOSS_FILENAME, "w") as loss_file:

        predictions_file.write("Step [domain]\t" +
                               "\t".join(ACTIONS) + "\n")
        loss_file.write("Step [domain]\tLoss [zero]\n")
        while True:
            loss, mae, predictions = dqn.update()
            if learn_step % 1000 == 0:
                predictions_file.write(str(learn_step) + " " +
                                       " ".join(predictions.mean(axis=0).astype(str)) + "\n")
                predictions_file.flush()
            loss_average.append(loss)
            learn_step += 1
            if learn_step % 100 == 0:
                loss_file.write("%5d %10.2f\n" % (learn_step, loss_average.average()))
                loss_file.flush()
            if learn_step % TARGET_MODEL_UPDATE_PERIOD == 0:
                dqn.take_target_model_snapshot()
            if learn_step % 100 == 0:
                print(learn_step)
                #dqn.print_weights()
                for i in range(0, 10):
                    input_tensor = GameState(i).input_tensor()
                    actions = dqn.model.predict(input_tensor, verbose=0)
                    action = np.argmax(actions)
                    print(i, actions, " "*20 if action == 1 else "", ACTIONS[action])
                dump_model_profile(dqn.model, "model.txt")
                dump_model_profile(dqn.target_model, "target_model.txt")

            if learn_step % 100 == 0:
                dqn.save_weights()

def main():
    transitions = make_transitions()
    learn(transitions)

def fuck():
    input_size = 1
    middle_layer_size = 4
    output_size = 2
    ACTIVATION = "sigmoid"
    learning_rate = 0.0001

    model = Sequential([
        InputLayer( (input_size,) ),
        Dense(middle_layer_size, activation=ACTIVATION),
        Dense(middle_layer_size, activation=ACTIVATION),
        Dense(middle_layer_size, activation=ACTIVATION),
        Dense(middle_layer_size, activation=ACTIVATION),
        Dense(middle_layer_size, activation=ACTIVATION),
        #Dense(middle_layer_size, activation=ACTIVATION),
        #Dense(middle_layer_size, activation=ACTIVATION, kernel_initializer="zeros", bias_initializer="zeros"),
        Dense(output_size, activation='linear'),
        #Dense(output_size, activation='linear', kernel_initializer="zeros", bias_initializer="zeros"),
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])
    model.summary()

    for i, layer in enumerate(model.layers):
        print("Weights for layer %d:" % (i + 1,))
        weights, biases = layer.get_weights()
        print(weights)
        print(biases)
        print()

    xx = clone_model(model)
    xx.set_weights(model.get_weights())
    for i, layer in enumerate(xx.layers):
        print("Weights for layer %d:" % (i + 1,))
        weights, biases = layer.get_weights()
        print(weights)
        print(biases)
        print()

try:
    main()
except KeyboardInterrupt:
    pass

