
import sys
import os
os.environ['KERAS_BACKEND'] = 'torch'
import pickle

import random
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam
import tensorflow as tf
import time
from collections import deque
import game

# Hyperparameters.
EPISODE_COUNT = 500 # Number of games to play.
GAMMA = 0.99 # Discount factor.
EPSILON_RANGE = (1.0, 0.1) # Explore (exploit). 
EPISODE_EXPLORE_COUNT = int(EPISODE_COUNT*0.10) # How many episodes to interpolate over EPSILON_RANGE.
REPLAY_BUFFER_SIZE = 100000
MINIBATCH_SIZE = 32 # How many replays to sample when back-propagating.
TARGET_MODEL_UPDATE_PERIOD = 1000 # How often to update the target model.
HISTORY_SIZE = 4 # Number of game states to include in NN inputs.
ACTION_REPEAT = 4 # Number of times to repeat each action.
UPDATE_PERIOD = 4 # Number of actions between NN update.
DDQN = True # Use Double DQN.
REPLAY_BUFFER_FILENAME = "games.pkl"

# Just random play, no learning:
if True:
    EPSILON_RANGE = (1.0, 1.0)
    UPDATE_PERIOD = None

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

        if False:
            game_state = self.game_states[-1]
            players = game_state.get_entities_by_types([game.TYPE_PLAYER_SHIP])
            enemies = game_state.get_entities_by_types(game.TYPE_ENEMIES)

            return np.array([
                players[0].x if players else 0,
                enemies[0].x if enemies else 0,
            ])

        if True:
            # TODO provide a few game states, at least 2 but maybe 4.
            game_state = self.game_states[-1]
            v = []
            for i in range(game.NUM_ENTITIES):
                entity = game_state.entities[i]
                if entity is None:
                    v.extend([0, 0, 0])
                else:
                    v.extend([1, entity.x, entity.y])
            v.extend([
                game_state.score,
                game_state.ships_left,
                game_state.abms_left,
            ])
            return np.array(v)

    def __repr__(self):
        # Only include most recent game state.
        return repr(self.game_states[-1])

# What we store in the replay buffer.
class Transition:
    def __init__(self, previous_phi, action, next_phi):
        self.previous_phi = previous_phi
        self.action = action
        if False:
            # Score.
            # TODO take into account number of gas cans.
            # TODO take into account 16-bit score wrapping.
            self.reward = next_phi.game_states[-1].score - previous_phi.game_states[-1].score
        if True:
            # Move to the right.
            self.reward = next_phi.game_states[-1].entities[0].x - previous_phi.game_states[-1].entities[0].x
            # Peg to the right.
            game_state = next_phi.game_states[-1]
            self.reward += int(game_state.entities[0].x == 121)
        if False:
            # Move toward the bad guy.
            game_state = previous_phi.game_states[-1]
            bad_guy = game_state.entities[36]
            target = bad_guy.x if bad_guy is not None else 0
            previous_diff = abs(game_state.entities[0].x - target)
            game_state = next_phi.game_states[-1]
            bad_guy = game_state.entities[36]
            target = bad_guy.x if bad_guy is not None else 0
            next_diff = abs(game_state.entities[0].x - target)
            self.reward = previous_diff - next_diff
        #print(self.reward, next_phi.game_states[-1].score, previous_phi.game_states[-1].score)
        # Clip to -1 to 1 like they do in the paper.
        # self.reward = min(max(self.reward, -1), 1)
        self.next_phi = next_phi

    def __repr__(self):
        return "Transition[%s, \"%s\", %d, %s]" % \
                (self.previous_phi, game.ACTIONS[self.action], self.reward, self.next_phi)

def main():
    # Replay buffer of Transition objects.
    initial_replay_buffer = []
    if os.path.exists(REPLAY_BUFFER_FILENAME):
        with open(REPLAY_BUFFER_FILENAME, "rb") as f:
            initial_replay_buffer = pickle.load(f)
        print("Loaded replay buffer:", len(initial_replay_buffer))
    replay_buffer = deque(initial_replay_buffer, REPLAY_BUFFER_SIZE)

    # Parameters of the neural network.
    input_tensor_shape = Phi(None, game.GameState()).input_tensor().shape
    print("input_tensor_shape", input_tensor_shape)
    action_count = len(game.ACTIONS)

    # Create neural net.
    model = Sequential([
        InputLayer(input_tensor_shape),
        Dense(1, activation='relu'), # relu = rectified linear unit = max(0, x)
        #Dense(50, activation='relu'),
        #Dense(50, activation='relu'),
        Dense(action_count, activation='linear'),
    ])
    model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])
    model.summary()

    if False: # TODO remove
        # See what noise we have in the blank NN.
        x = np.ones( (1,input_tensor_shape[0]) )*100
        print(x)
        y = model.predict(x, verbose=2)
        print(y)
        return

    # Make a copy of the model to use as the target model.
    target_model = clone_model(model)

    # TODO I think the Lua version has a single loop of steps and they
    # restart the game when it finishes. Might make it easier to re-use the
    # same trs80gp instance.
    step = 0

    live_game = game.LiveGame(False)
    start_time = time.perf_counter()
    print("start time", start_time)

    # An episode is a full play of the game.
    for episode in range(EPISODE_COUNT):
        now = time.perf_counter()
        epsilon = EPSILON_RANGE[0] + (EPSILON_RANGE[1] - EPSILON_RANGE[0])*min(episode/EPISODE_EXPLORE_COUNT, 1)
        estimated_minutes_left = 0 if episode == 0 else EPISODE_COUNT/episode*(now - start_time)/60
        print("episode", episode, "step", step, "epsilon", epsilon, "replay buffer", len(replay_buffer), "time", now, "minutes left", int(estimated_minutes_left))

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
                print(actions, action, game.ACTIONS[action])

            # Execute action in game, get next state.
            before = time.perf_counter()
            for i in range(ACTION_REPEAT):
                live_game.perform_action(action)
                game_state = live_game.read_state()
                #print("player x", game_state.entities[0].x)
                if game_state.game_over:
                    break
            if game_state.game_over:
                print("Game over, score =", game_state.score)
                break
            after = time.perf_counter()
            #print("update time", int((after - before)*1000))
            #print(" ".join("%x" % e.entity_type if e is not None else "." for e in game_state.entities))
            next_phi = Phi(phi, game_state)

            # Make transition to capture what we experienced.
            transition = Transition(phi, action, next_phi)
            phi = next_phi

            # Append transition to replay buffer.
            replay_buffer.append(transition)
            #print(transition, phi.game_states[-1].entities[0].x)

            if UPDATE_PERIOD is not None and step % UPDATE_PERIOD == 0:
                # Sample minibatch of transitions from replay buffer.
                # TODO: Sample proportionally to loss.
                transitions = random.sample(replay_buffer, min(MINIBATCH_SIZE, len(replay_buffer)))

                # Compute predictions of batch.
                previous_input_tensors = np.array([transition.previous_phi.input_tensor() for transition in transitions])
                predictions = model.predict(previous_input_tensors, verbose=0)

                # Compute targets of batch.
                rewards = np.array([transition.reward for transition in transitions])
                # If end of game, then it's just the reward. Otherwise it's the reward
                # plus expected value of best action using target_model.
                live_game_indices = np.array([i for i in range(len(transitions))
                                              if not transitions[i].next_phi.game_states[-1].game_over])
                #print("live_game_indices", live_game_indices)
                next_input_tensors = np.array([transitions[i].next_phi.input_tensor() for i in live_game_indices])
                #print("next_input_tensors", next_input_tensors[0])
                if DDQN:
                    # DDQN
                    live_futures = model.predict(next_input_tensors, verbose=0)
                    #print("live_futures", live_futures[0])
                    best_live_future_indices = np.argmax(live_futures, axis=1)
                    live_target_futures = target_model.predict(next_input_tensors, verbose=0)
                    best_live_futures = live_target_futures[
                            np.arange(live_target_futures.shape[0]),best_live_future_indices]
                    #print("best_live_futures", best_live_futures)
                    best_futures = np.zeros(len(transitions))
                    best_futures[live_game_indices] = best_live_futures
                else:
                    # DQN
                    live_futures = target_model.predict(next_input_tensors, verbose=0)
                    #print("live_futures", live_futures[0])
                    best_live_futures = np.max(live_futures, axis=1)
                    #print("best_live_futures", best_live_futures)
                    best_futures = np.zeros(len(transitions))
                    best_futures[live_game_indices] = best_live_futures
                targets = rewards + GAMMA*best_futures

                # Update targets for the actions we actually took.
                actions = np.array([transition.action for transition in transitions])
                predictions[np.arange(predictions.shape[0]),actions] = targets

                # Perform gradient descent.
                history = model.fit(previous_input_tensors, predictions, epochs=1, verbose=0)
                print("metric", history.history["loss"], history.history["mae"])

            # Periodically update our target model to our current model.
            step += 1
            if step % TARGET_MODEL_UPDATE_PERIOD == 0:
                target_model = clone_model(model)

        if False:
            before = time.perf_counter()
            with open(REPLAY_BUFFER_FILENAME, "wb") as f:
                pickle.dump(replay_buffer, f)
            after = time.perf_counter()
            print("pickling time", int((after - before)*1000), "ms")

    live_game.kill()

try:
    main()
except KeyboardInterrupt:
    pass
