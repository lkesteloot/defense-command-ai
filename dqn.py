
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
REPLAY_BUFFER_SIZE = 1_000_000
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

# Enemy with highest Y value, or None if there are no enemies.
def get_lowest_enemy(game_state):
    enemies = game_state.get_entities_by_types(game.TYPE_ENEMIES)
    if enemies:
        lowest_enemy = enemies[0]
        for enemy in enemies:
            if enemy.y > lowest_enemy.y:
                lowest_enemy = enemy
    else:
        lowest_enemy = None

    return lowest_enemy

def get_player(game_state):
    players = game_state.get_entities_by_types([game.TYPE_PLAYER_SHIP])
    return players[0] if players else None

def get_player_bullet(game_state):
    player_bullets = game_state.get_entities_by_types([game.TYPE_PLAYER_SHOT])
    return player_bullets[0] if player_bullets else None

def move_recommendation(game_state):
    move = 0

    player = get_player(game_state)
    if player:
        lowest_enemy = get_lowest_enemy(game_state)
        target_x = 64 if lowest_enemy is None else lowest_enemy.x
        move = target_x - player.x

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

def moving(previous_game_state, next_game_state):
    return next_game_state.entities[0].x != previous_game_state.entities[0].x

def normalize_x(x):
    return x/64 - 1

def normalize_y(y):
    return y/24 - 1

def unnormalize_x(x):
    return round((x + 1)*64)

def unnormalize_y(y):
    return round((y + 1)*24)

def normalized_x_of(entity):
    return normalize_x(entity.x if entity else 64)

def normalized_y_of(entity):
    return normalize_y(entity.y if entity else 24)

# Preprocessed sequence of game states.
class Phi:
    def __init__(self, previous_phi, game_state):
        if previous_phi is None:
            self.game_states = []
        else:
            self.game_states = previous_phi.game_states[-(HISTORY_SIZE - 1):]

        while len(self.game_states) < HISTORY_SIZE:
            self.game_states.append(game_state)

    def input_tensor(self):
        # TODO we could cache this, but we'd have to make sure it's not modified
        # by the caller.
        game_state = self.game_states[-1]

        # Fix data read from a file.
        while len(self.game_states) < HISTORY_SIZE:
            self.game_states.insert(0, self.game_states[0])

        if False:
            # Simple direction, left or right.
            return np.array([
                move_recommendation(game_state),
            ])

        if False:
            # Minimal info for all states.
            x = []
            for game_state in self.game_states:
                player = get_player(game_state)
                player_bullet = get_player_bullet(game_state)
                lowest_enemy = get_lowest_enemy(game_state)
                x.extend([
                    normalized_x_of(player),
                    normalized_x_of(player_bullet),
                    normalized_y_of(player_bullet),
                    normalized_x_of(lowest_enemy),
                    normalized_y_of(lowest_enemy),
                ])
            return np.array(x)

        if True:
            # Minimal info.
            player = get_player(game_state)
            player_bullet = get_player_bullet(game_state)
            lowest_enemy = get_lowest_enemy(game_state)
            return np.array([
                normalized_x_of(player),
                normalized_x_of(player_bullet),
                normalized_y_of(player_bullet),
                normalized_x_of(lowest_enemy),
                normalized_y_of(lowest_enemy),
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
        previous_game_state = self.previous_phi.game_states[-1]
        next_game_state = self.next_phi.game_states[-1]

        if True:
            # Score.
            # TODO take into account number of gas cans.
            # TODO take into account 16-bit score wrapping.
            reward = next_game_state.score - previous_game_state.score

        if False:
            rec = move_recommendation(previous_game_state)
            if rec < 0:
                reward = int(moving_left(previous_game_state, next_game_state))
            elif rec > 0:
                reward = int(moving_right(previous_game_state, next_game_state))
            else:
                reward = int(not moving(previous_game_state, next_game_state))

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

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque([], size)

    def sample(self, count):
        return random.sample(self.buffer, min(count, len(self.buffer)))

    def add(self, transition):
        self.buffer.append(transition)

    def load(self):
        # Replay buffer of Transition objects.
        if os.path.exists(REPLAY_BUFFER_FILENAME):
            with open(REPLAY_BUFFER_FILENAME, "rb") as f:
                new_transitions = pickle.load(f)
                self.buffer.extend(new_transitions)
            print("Loaded replay buffer:", len(new_transitions))
        else:
            print("Replay buffer missing")

    def save(self):
        with open(REPLAY_BUFFER_FILENAME, "wb") as f:
            pickle.dump(self.buffer, f)

class DQN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Average of input and output.
        middle_layer_size = (input_size + output_size) // 2
        middle_layer_size = input_size # Guaranteed not lossy.
        middle_layer_size = 32

        # Create neural net.
        self.model = Sequential([
            InputLayer( (input_size,) ),
            # relu = rectified linear unit = max(0, x)
            Dense(middle_layer_size, activation='relu'),
            Dense(middle_layer_size, activation='relu'),
            #Dense(middle_layer_size, activation='relu', kernel_initializer="zeros", bias_initializer="zeros"),
            Dense(output_size, activation='linear'),
            #Dense(output_size, activation='linear', kernel_initializer="zeros", bias_initializer="zeros"),
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
        transitions = self.replay_buffer.sample(MINIBATCH_SIZE)

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

    def print_weights(self):
        for i, layer in enumerate(self.model.layers):
            print("Weights for layer %d:" % (i + 1,))
            weights, biases = layer.get_weights()
            print(weights)
            print(biases)
            print()

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
    print("phi", Phi(None, game.GameState()))
    print("input_tensor", Phi(None, game.GameState()).input_tensor())
    print("input_tensor_shape", input_tensor_shape)
    action_count = len(game.ACTIONS)

    dqn = DQN(input_tensor_shape[0], action_count)
    print("Initial weights:")
    dqn.print_weights()
    dqn.replay_buffer.load()

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
            while True:
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
                    print("%d" % (learn_step,))
                    dqn.print_weights()
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
        print("episode", episode, "step", step, "epsilon", epsilon, "replay buffer", len(dqn.replay_buffer.buffer), "time", now, "minutes left", int(estimated_minutes_left))

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
            dqn.replay_buffer.add(transition)
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
            dqn.replay_buffer.save()

    live_game.kill()

def play():
    # Parameters of the neural network.
    input_tensor_shape = Phi(None, game.GameState()).input_tensor().shape
    print("input_tensor_shape", input_tensor_shape)
    action_count = len(game.ACTIONS)

    dqn = DQN(input_tensor_shape[0], action_count)
    dqn.replay_buffer.load()

    if os.path.exists(WEIGHTS_FILENAME):
        print("Loading weights from " + WEIGHTS_FILENAME)
        dqn.model.load_weights(WEIGHTS_FILENAME)

    dqn.print_weights()

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
            if FOR_NOW and game_step < 120:
                # Move left to avoid initial death.
                action = 1
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

def print_transition(transition):
    p = transition.previous_phi
    n = transition.next_phi

    def unnormalize(v):
        return np.array([
            unnormalize_x(v[0]),
            unnormalize_x(v[1]),
            unnormalize_y(v[2]),
            unnormalize_x(v[3]),
            unnormalize_y(v[4]),
        ])

    print(game.ACTIONS[transition.action], transition.compute_reward(), n.game_states[-1].game_over)
    print(np.vstack([
        unnormalize(p.input_tensor()[-5:]),
        unnormalize(n.input_tensor()[-5:]),
    ]))

# Create and save a replay buffer from random actions.
def make_replay():
    DEBUG = False

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    start_time = time.perf_counter()
    live_game = game.LiveGame(not DEBUG)
    episode = 0

    while len(replay_buffer.buffer) < REPLAY_BUFFER_SIZE:
        now = time.perf_counter()
        elapsed = now - start_time
        size = len(replay_buffer.buffer)
        num_left = max(0, REPLAY_BUFFER_SIZE - size)
        estimated_minutes_left = 0 if size == 0 else elapsed/size*num_left/60

        print("replay buffer", len(replay_buffer.buffer), "minutes left", int(estimated_minutes_left))

        # Run the game.
        seed = episode
        game_state = live_game.start_new_game(seed)
        game_step = 0

        # Initialize our game state.
        phi = Phi(None, game_state)

        # Each iteration is an action chosen in the game.
        while True:
            if DEBUG:
                time.sleep(0.050)

            # Move randomly.
            action = random.randrange(0, len(game.ACTIONS))
            if game_step < 120 and action >= 3:
                # Don't shoot to avoid initial death.
                action -= 3

            # Execute action in game, get next state.
            game_state = live_game.perform_action(action)
            next_phi = Phi(phi, game_state)

            # Make transition to capture what we experienced.
            transition = Transition(phi, action, next_phi)
            if DEBUG:
                print_transition(transition)
            phi = next_phi

            # Append transition to replay buffer.
            replay_buffer.add(transition)

            if game_state.game_over:
                time.sleep(5)
                print("Game over, score =", game_state.score)
                break

            game_step += 1

        # Save the replay buffer between every game.
        replay_buffer.save()

        episode += 1

    live_game.kill()

# Analyze the saved replay buffer.
def look_replay():
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    replay_buffer.load()

    for transition in replay_buffer.buffer:
        print_transition(transition)

try:
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command == "play":
        play()
    elif command == "learn":
        learn()
    elif command == "make-replay":
        make_replay()
    elif command == "look-replay":
        look_replay()
    else:
        print("Unknown command")
except KeyboardInterrupt:
    pass

