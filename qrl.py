
import sys
import os
os.environ['KERAS_BACKEND'] = 'torch'

import random
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import InputLayer, Dense
import tensorflow as tf
import time
from collections import deque
import game

# Hyperparameters.
EPISODE_COUNT = 100 # Number of games to play.
EPISODE_EXPLORE_COUNT = 50 # How many episodes we explore for (after that all exploit).
GAMMA = 0.99 # Discount factor.
EPSILON_RANGE = (1.0, 0.1) # Explore (exploit). 
REPLAY_BUFFER_SIZE = 10000
MINIBATCH_SIZE = 32 # How many replays to sample when back-propagating.
TARGET_MODEL_UPDATE_PERIOD = 1000 # How often to update the target model.
HISTORY_SIZE = 4 # Number of game states to include in NN inputs.
ACTION_REPEAT = 4 # Number of times to repeat each action.
UPDATE_PERIOD = 4 # Number of actions between NN update.

# Preprocessed sequence of game states.
class Phi:
    def __init__(self, previous_phi, game_state):
        if previous_phi is None:
            self.game_states = [game_state]
        else:
            self.game_states = previous_phi.game_states[-(HISTORY_SIZE - 1):]
            self.game_states.append(game_state)

    def input_tensor(self):
        # TODO we could cache this, but we'd have to make sure it's not modified
        # by the caller.
        game_state = self.game_states[-1]
        players = game_state.get_entities_by_types([game.TYPE_PLAYER_SHIP])
        enemies = game_state.get_entities_by_types(game.TYPE_ENEMIES)

        return np.array([
            players[0].x if players else 0,
            enemies[0].x if enemies else 0,
        ])

    def __repr__(self):
        # Only include most recent game state.
        return repr(self.game_states[-1])

# What we store in the replay buffer.
class Transition:
    def __init__(self, previous_phi, action, next_phi):
        self.previous_phi = previous_phi
        self.action = action
        # TODO take into account number of gas cans.
        self.reward = next_phi.game_states[-1].score - previous_phi.game_states[-1].score
        self.next_phi = next_phi

    def __repr__(self):
        return "Transition[%s, \"%s\", +%d, %s]" % \
                (self.previous_phi, game.ACTIONS[self.action], self.reward, self.next_phi)

def main():
    # Replay buffer of Transition objects.
    replay_buffer = deque([], REPLAY_BUFFER_SIZE)

    # Parameters of the neural network.
    input_tensor_shape = Phi(None, game.GameState()).input_tensor().shape
    print("input_tensor_shape", input_tensor_shape)
    action_count = len(game.ACTIONS)

    # Create neural net.
    model = Sequential([
        InputLayer(input_tensor_shape),
        Dense(50, activation='relu'), # relu = rectified linear unit = max(0, x)
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(action_count, activation='linear'),
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()

    # Make a copy of the model to use as the target model.
    target_model = clone_model(model)

    # TODO I think the Lua version has a single loop of steps and they
    # restart the game when it finishes. Might make it easier to re-use the
    # same trs80gp instance.
    step = 0

    live_game = game.LiveGame()

    # An episode is a full play of the game.
    for episode in range(EPISODE_COUNT):
        epsilon = EPSILON_RANGE[0] + (EPSILON_RANGE[1] - EPSILON_RANGE[0])*min(episode/EPISODE_EXPLORE_COUNT, 1)
        print("episode", episode, "step", step, "epsilon", epsilon, "replay buffer", len(replay_buffer))

        # Run the game.
        seed = episode
        live_game.start_new_game(seed)

        # Initialize our game state.
        game_state = live_game.read_state()
        phi = Phi(None, game_state)

        # Each iteration is an action chosen in the game.
        while True:
            # Decide whether to expore or exploit.
            if random.random() < epsilon:
                # Explore.
                action = random.randrange(0, action_count)
            else:
                # Exploit.
                actions = model.predict(phi.input_tensor()[np.newaxis,:], verbose=0)[0]
                action = np.argmax(actions)

            # Execute action in game, get next state.
            for i in range(ACTION_REPEAT):
                live_game.perform_action(action)
                game_state = live_game.read_state()
                if game_state.game_over:
                    break
            if game_state.game_over:
                print("Game over, score =", game_state.score)
                break
            next_phi = Phi(phi, game_state)

            # Make transition to capture what we experienced.
            transition = Transition(phi, action, next_phi)

            # Append transition to replay buffer.
            replay_buffer.append(transition)

            if step % UPDATE_PERIOD == 0:
                # Sample minibatch of transitions from replay buffer.
                # TODO: Sample proportionally to loss.
                transitions = random.sample(replay_buffer, min(MINIBATCH_SIZE, len(replay_buffer)))

                # Compute predictions of batch.
                previous_input_tensors = np.array([transition.previous_phi.input_tensor() for transition in transitions])
                predictions = model.predict(previous_input_tensors, verbose=0)

                # Compute targets of batch.
                rewards = np.array([transition.reward for transition in transitions])
                # TODO take into account game over.
                # If end of game, then it's just the reward. Otherwise it's the reward
                # plus expected value of best action using target_model.
                ##game_overs = np.array([transition.next_phi.game_states[-1].game_over for transition in transitions])
                ##game_over_indices = np.where(game_overs)[0]
                ##live_game_indices = np.where(~game_overs)[0]
                next_input_tensors = np.array([transition.next_phi.input_tensor() for transition in transitions])
                futures = target_model.predict(next_input_tensors, verbose=0)
                best_futures = np.max(futures, axis=1)
                targets = rewards + GAMMA*best_futures

                # Update targets for the actions we actually took.
                actions = np.array([transition.action for transition in transitions])
                predictions[np.arange(predictions.shape[0]),actions] = targets

                # Perform gradient descent.
                model.fit(previous_input_tensors, predictions, epochs=1, verbose=0)

            # Periodically update our target model to our current model.
            step += 1
            if step % TARGET_MODEL_UPDATE_PERIOD == 0:
                target_model = clone_model(model)

    live_game.kill()

try:
    main()
except KeyboardInterrupt:
    pass
