
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
import game

# Hyperparameters.
EPISODE_COUNT = 500 # Number of games to play.
GAMMA = 0.99 # Discount factor.
## GAMMA = 0.00 # Myopic.
EPSILON_RANGE = (1.0, 0.1) # Explore (exploit). 
EPSILON_RANGE = (0.0, 0.0) # Only exploit.
EPISODE_EXPLORE_COUNT = int(EPISODE_COUNT*0.10) # How many episodes to interpolate over EPSILON_RANGE.
REPLAY_BUFFER_SIZE = 100000
MINIBATCH_SIZE = 32 # How many replays to sample when back-propagating.
TARGET_MODEL_UPDATE_PERIOD = 1000 # How often to update the target model.
HISTORY_SIZE = 4 # Number of game states to include in NN inputs.
ACTION_REPEAT = 4 # Number of times to repeat each action.
UPDATE_PERIOD = 4 # Number of actions between NN update.
DDQN = True # Use Double DQN.
REPLAY_BUFFER_FILENAME = "games.pkl"
WEIGHTS_FILENAME = "trained.weights.h5"
PREDICTIONS_FILENAME = "predictions.txt"
LOSS_FILENAME = "loss.txt"

FOR_NOW = True

def move_recommendation(game_state):
    # Just whether to move left or right.
    players = game_state.get_entities_by_types([game.TYPE_PLAYER_SHIP])
    enemies = game_state.get_entities_by_types(game.TYPE_ENEMIES)

    move = 0
    if players:
        if enemies:
            lowest_enemy = enemies[0]
            for enemy in enemies:
                if enemy.y > lowest_enemy.y:
                    lowest_enemy = enemy
            target_x = lowest_enemy.x
        else:
            target_x = 64

        px = players[0].x
        if px < target_x:
            move = 1
        elif px > target_x:
            move = -1

    return move

def moving_right(previous_game_state, next_game_state):
    if next_game_state.entities[0].x > previous_game_state.entities[0].x:
        return True

    # Peg to the right.
    return next_game_state.entities[0].x == 121

def moving_left(previous_game_state, next_game_state):
    if next_game_state.entities[0].x < previous_game_state.entities[0].x:
        return True

    # Peg to the left.
    return next_game_state.entities[0].x == 0

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

        if True:
            return np.array([
                move_recommendation(game_state),
            ])

        if True:
            # Just X position of player and one bad guy.
            players = game_state.get_entities_by_types([game.TYPE_PLAYER_SHIP])
            enemies = game_state.get_entities_by_types(game.TYPE_ENEMIES)

            return np.array([
                players[0].x if players else 0,
                enemies[0].x if enemies else 0,
            ])

        if False:
            # Something basic.
            return np.array([
                # game_state.ships_left,
                game_state.get_fuel_can_count(),
                # game_state.score,
            ])

        if False:
            # TODO provide a few game states, at least 2 but maybe 4.
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
        self.next_phi = next_phi

    def compute_reward(self):
        next_game_state = self.next_phi.game_states[-1]
        previous_game_state = self.previous_phi.game_states[-1]

        if False:
            # Score.
            # TODO take into account number of gas cans.
            # TODO take into account 16-bit score wrapping.
            reward = next_game_state.score - previous_game_state.score

        if True:
            rec = move_recommendation(previous_game_state)
            if rec < 0:
                reward = int(moving_left(previous_game_state, next_game_state))
            elif rec > 0:
                reward = int(moving_right(previous_game_state, next_game_state))
            else:
                reward = 0

        if False:
            if previous_game_state.get_fuel_can_count() > 8:
                reward = int(moving_right(previous_game_state, next_game_state))
            else:
                reward = int(moving_left(previous_game_state, next_game_state))

        if False:
            # Move toward the bad guy.
            bad_guy = previous_game_state.entities[36]
            target = bad_guy.x if bad_guy is not None else 0
            previous_diff = abs(previous_game_state.entities[0].x - target)

            bad_guy = next_game_state.entities[36]
            target = bad_guy.x if bad_guy is not None else 0
            next_diff = abs(next_game_state.entities[0].x - target)

            reward = previous_diff - next_diff

        # Clip to -1 to 1 like they do in the paper.
        # reward = min(max(reward, -1), 1)

        return reward

    def __repr__(self):
        return "Transition[%s, \"%s\", %d, %s]" % \
                (self.previous_phi, game.ACTIONS[self.action], self.compute_reward(), self.next_phi)

class DQN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.replay_buffer = deque([], REPLAY_BUFFER_SIZE)

        # Average of input and output.
        middle_layer_size = (input_size + output_size) // 2
        #middle_layer_size = 1

        # Create neural net.
        self.model = Sequential([
            InputLayer( (input_size,) ),
            # relu = rectified linear unit = max(0, x)
            #Dense(middle_layer_size, activation='relu'),
            #Dense(middle_layer_size, activation='relu', kernel_initializer="zeros", bias_initializer="zeros"),
            #Dense(output_size, activation='linear'),
            Dense(output_size, activation='linear', kernel_initializer="zeros", bias_initializer="zeros"),
        ])
        self.model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])
        self.model.summary()

        if False: # TODO remove
            # See what noise we have in the blank NN.
            x = np.ones( (1,input_size) )*100
            print(x)
            y = self.model.predict(x, verbose=2)
            print(y)

        # Make a copy of the model to use as the target model.
        self.target_model = clone_model(self.model)

    def update(self):
        # Sample minibatch of transitions from replay buffer.
        # TODO: Sample proportionally to loss.
        transitions = random.sample(self.replay_buffer, min(MINIBATCH_SIZE, len(self.replay_buffer)))

        # Compute predictions of batch.
        previous_input_tensors = np.array([transition.previous_phi.input_tensor()
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
                                      if not transitions[i].next_phi.game_states[-1].game_over])
        #print("live_game_indices", live_game_indices)
        next_input_tensors = np.array([transitions[i].next_phi.input_tensor() for i in live_game_indices])
        #print("next_input_tensors", next_input_tensors[0])
        if DDQN:
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
        targets = rewards + GAMMA*best_futures

        # Update targets for the actions we actually took.
        actions = np.array([transition.action for transition in transitions])
        fixed_predictions = predictions.copy()
        fixed_predictions[np.arange(fixed_predictions.shape[0]),actions] = targets

        # Perform gradient descent.
        history = self.model.fit(previous_input_tensors, fixed_predictions, epochs=1, verbose=0)
        return history.history["loss"][0], history.history["mae"][0], predictions

    def take_target_model_snapshot(self):
        self.target_model = clone_model(self.model)

    def load_replay_buffer(self):
        # Replay buffer of Transition objects.
        if os.path.exists(REPLAY_BUFFER_FILENAME):
            with open(REPLAY_BUFFER_FILENAME, "rb") as f:
                new_transitions = pickle.load(f)
                self.replay_buffer.extend(new_transitions)
            print("Loaded replay buffer:", len(new_transitions))

    def dump_replay_buffer(self):
        before = time.perf_counter()
        with open(REPLAY_BUFFER_FILENAME, "wb") as f:
            pickle.dump(self.replay_buffer, f)
        after = time.perf_counter()
        print("dqn dump time", int((after - before)*1000), "ms")

# Compute the box (straight) average of the previous "size" numbers.
class BoxAverage:
    def __init__(self, size):
        self.buffer = deque([], size)
        self.sum = 0

    def append(self, value):
        if len(self.buffer) == self.buffer.maxlen:
            self.sum -= self.buffer[0]
        self.sum += value
        self.buffer.append(value)

    def average(self):
        return self.sum / len(self.buffer) if len(self.buffer) != 0 else 0

def learn():
    # Parameters of the neural network.
    input_tensor_shape = Phi(None, game.GameState()).input_tensor().shape
    print("input_tensor_shape", input_tensor_shape)
    action_count = len(game.ACTIONS)

    dqn = DQN(input_tensor_shape[0], action_count)
    dqn.load_replay_buffer()

    if False and os.path.exists(WEIGHTS_FILENAME):
        print("Loading weights from " + WEIGHTS_FILENAME)
        dqn.model.load_weights(WEIGHTS_FILENAME)

    # Train on loaded replay buffer.
    if True:
        loss_average = BoxAverage(1000)
        mae_average = BoxAverage(1000)
        learn_step = 0
        print("Writing loss data to \"%s\"" % (LOSS_FILENAME,))
        with open(PREDICTIONS_FILENAME, "w") as predictions_file, \
                open(LOSS_FILENAME, "w") as loss_file:

            predictions_file.write("Step [domain]\t" +
                                   "\t".join(action if action != "" else "None"
                                             for action in game.ACTIONS) + "\n")
            loss_file.write("Step [domain]\tLoss [zero]\tMAE [right,zero]\n")
            while learn_step < 1_000_000:
                loss, mae, predictions = dqn.update()
                if learn_step % 1000 == 0:
                    predictions_file.write(str(learn_step) + " " +
                                           " ".join(predictions.mean(axis=0).astype(str)) + "\n")
                    predictions_file.flush()
                loss_average.append(loss)
                mae_average.append(mae)
                learn_step += 1
                if learn_step % 1000 == 0:
                    loss_file.write("%5d %10.2f %10.2f\n" % (learn_step,
                                                     loss_average.average(),
                                                     mae_average.average()))
                    loss_file.flush()
                if learn_step % TARGET_MODEL_UPDATE_PERIOD == 0:
                    dqn.take_target_model_snapshot()
                if learn_step % 1000 == 0:
                    print("    %d" % (learn_step,))
                if learn_step % 1000 == 0:
                    print("Saving weights to " + WEIGHTS_FILENAME)
                    dqn.model.save_weights(WEIGHTS_FILENAME)
        return

    # TODO I think the Lua version has a single loop of steps and they
    # restart the game when it finishes.
    step = 0

    live_game = game.LiveGame(False)
    start_time = time.perf_counter()

    # An episode is a full play of the game.
    for episode in range(EPISODE_COUNT):
        now = time.perf_counter()
        epsilon = EPSILON_RANGE[0] + (EPSILON_RANGE[1] - EPSILON_RANGE[0])*min(episode/EPISODE_EXPLORE_COUNT, 1)
        estimated_minutes_left = 0 if episode == 0 else EPISODE_COUNT/episode*(now - start_time)/60
        print("episode", episode, "step", step, "epsilon", epsilon, "replay buffer", len(dqn.replay_buffer), "time", now, "minutes left", int(estimated_minutes_left))

        # Run the game.
        seed = episode
        game_state = live_game.start_new_game(seed)

        if FOR_NOW:
            game_step = 0

        # Initialize our game state.
        phi = Phi(None, game_state)

        # Each iteration is an action chosen in the game.
        while True:
            # Decide whether to explore or exploit.
            if random.random() < epsilon:
                # Explore.
                action = random.randrange(0, dqn.output_size)
            else:
                # Exploit.
                input_tensor = phi.input_tensor()
                if FOR_NOW:
                    time.sleep(0.050)
                actions = dqn.model.predict(input_tensor[np.newaxis,:], verbose=0)[0]
                action = np.argmax(actions)
                if FOR_NOW and action < 3:
                    # Force fire.
                    action += 3
                if FOR_NOW and game_step < 25:
                    # Move right to avoid initial death.
                    action = 2
                print(input_tensor, actions, action, game.ACTIONS[action])

            # Execute action in game, get next state.
            before = time.perf_counter()
            next_phi = phi
            for i in range(ACTION_REPEAT):
                game_state = live_game.perform_action(action)
                next_phi = Phi(next_phi, game_state)
                #print("player x", game_state.entities[0].x)
                if game_state.game_over:
                    break
            if game_state.game_over:
                print("Game over, score =", game_state.score)
                break
            after = time.perf_counter()
            #print("game action time", int((after - before)*1000))

            # Make transition to capture what we experienced.
            transition = Transition(phi, action, next_phi)
            phi = next_phi

            # Append transition to replay buffer.
            dqn.replay_buffer.append(transition)
            #print(transition, phi.game_states[-1].entities[0].x)

            if UPDATE_PERIOD is not None and step % UPDATE_PERIOD == 0:
                dqn.update()

            # Periodically update our target model to our current model.
            step += 1
            if step % TARGET_MODEL_UPDATE_PERIOD == 0:
                dqn.take_target_model_snapshot()

            if FOR_NOW:
                game_step += 1

        if False:
            # Save the replay buffer between every game.
            dqn.dump_replay_buffer()

    live_game.kill()

def play():
    # Parameters of the neural network.
    input_tensor_shape = Phi(None, game.GameState()).input_tensor().shape
    print("input_tensor_shape", input_tensor_shape)
    action_count = len(game.ACTIONS)

    dqn = DQN(input_tensor_shape[0], action_count)
    dqn.load_replay_buffer()

    if os.path.exists(WEIGHTS_FILENAME):
        print("Loading weights from " + WEIGHTS_FILENAME)
        dqn.model.load_weights(WEIGHTS_FILENAME)

    # TODO I think the Lua version has a single loop of steps and they
    # restart the game when it finishes.
    step = 0

    live_game = game.LiveGame(False)

    # An episode is a full play of the game.
    for episode in range(EPISODE_COUNT):
        now = time.perf_counter()

        # Run the game.
        seed = episode
        game_state = live_game.start_new_game(seed)

        if FOR_NOW:
            game_step = 0

        # Initialize our game state.
        phi = Phi(None, game_state)

        # Each iteration is an action chosen in the game.
        while True:
            input_tensor = phi.input_tensor()
            if FOR_NOW:
                time.sleep(0.050)
            actions = dqn.model.predict(input_tensor[np.newaxis,:], verbose=0)[0]
            action = np.argmax(actions)
            if FOR_NOW and action < 3:
                # Force fire.
                action += 3
            if FOR_NOW and game_step < 100:
                # Move right to avoid initial death.
                action = 2
            print(input_tensor, actions, action, game.ACTIONS[action])

            # Execute action in game, get next state.
            before = time.perf_counter()
            game_state = live_game.perform_action(action)
            next_phi = Phi(phi, game_state)
            #print("player x", game_state.entities[0].x)
            if game_state.game_over:
                print("Game over, score =", game_state.score)
                break
            after = time.perf_counter()
            #print("game action time", int((after - before)*1000))

            phi = next_phi

            step += 1

            if FOR_NOW:
                game_step += 1

    live_game.kill()

try:
    if "--play" in sys.argv:
        play()
    else:
        learn()
except KeyboardInterrupt:
    pass
