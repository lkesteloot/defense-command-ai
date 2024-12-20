
import sys
import os
os.environ['KERAS_BACKEND'] = 'torch'
import pickle

import random
import numpy as np
import time
from collections import deque
import game
from boxaverage import BoxAverage
from ml import DQN

# Hyperparameters.
LEARNING_RATE=0.001 # Default is 0.001.
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
DOUBLE_DQN = True # Use Double DQN.
REPLAY_WINS = False
REPLAY_BUFFER_FILENAME = "games.pkl"
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

        if False:
            # Simple direction, left or right.
            return np.array([
                move_recommendation(game_state),
            ])

        if False:
            # One-hot position of player, lowest enemy, and our bullet.
            v = np.zeros(64*3)
            player = get_player(game_state)
            if player:
                x = player.x//2
                if x >= 0 and x < 64:
                    v[x] = 1
            lowest_enemy = get_lowest_enemy(game_state)
            if lowest_enemy:
                x = lowest_enemy.x//2
                if x >= 0 and x < 64:
                    v[64 + x] = 1
            player_bullet = get_player_bullet(game_state)
            if player_bullet:
                x = player_bullet.x//2
                if x >= 0 and x < 64:
                    v[2*64 + x] = 1
            return np.array(v)

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

        if True:
            # Full info.
            v = []
            for game_state in self.game_states:
                for i in range(game.NUM_ENTITIES):
                    entity = game_state.entities[i]
                    if entity is None:
                        v.extend([0, 0, 0])
                    else:
                        v.extend([1,
                                  normalize_x(entity.x),
                                  normalize_y(entity.y)])
                v.extend([
                    game_state.score/10000, # Make it a small number.
                    game_state.ships_left/4, # Normalize.
                    game_state.abms_left/4, # Normalize.
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

    def get_score(self):
        # Max to avoid getting negative score when game restarts.
        previous_game_state = self.previous_phi.game_states[-1]
        next_game_state = self.next_phi.game_states[-1]
        return max(next_game_state.score - previous_game_state.score, 0)

    def compute_reward(self):
        previous_game_state = self.previous_phi.game_states[-1]
        next_game_state = self.next_phi.game_states[-1]

        reward = 0

        if True:
            # Penalize death.
            if False and next_game_state.ships_left < previous_game_state.ships_left:
                reward += -10

            # Penalize being away from center.
            player = get_player(next_game_state)
            if player:
                reward += -((player.x - 64)**2)/4096

        if False:
            # Score.
            # TODO take into account number of gas cans.
            # TODO take into account 16-bit score wrapping.
            reward += self.get_score()

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
        # Deque of transitions.
        self.buffer = deque([], size)

        # Lists are more efficient for sample() but don't work for add().
        self.is_list = False

    def sample(self, count):
        if REPLAY_WINS:
            indices = random.sample(self.wins, min(count, len(self.wins)))
            return [self.buffer[i] for i in indices]
        else:
            return random.sample(self.buffer, min(count, len(self.buffer)))

    def add(self, transition):
        if self.is_list:
            raise Exception("can't add to a list")
        self.buffer.append(transition)

    def load(self):
        # Replay buffer of Transition objects.
        if os.path.exists(REPLAY_BUFFER_FILENAME):
            with open(REPLAY_BUFFER_FILENAME, "rb") as f:
                new_transitions = pickle.load(f)
                self.buffer.extend(new_transitions)
            print(f"Loaded replay buffer: {len(new_transitions):,}")
        else:
            print("Replay buffer missing")

        if REPLAY_WINS:
            # Indices of wins and a few moves just before.
            self.wins = set()
            for i, transition in enumerate(self.buffer):
                if transition.get_score() > 0:
                    self.wins.update(range(max(i - 20, 0), i + 1))
            self.wins = list(self.wins)
            print(f"Wins {len(self.wins):,} ({len(self.wins)/len(self.buffer)*100:.1f}% of total)")

        self.buffer = list(self.buffer)
        self.is_list = True

    def save(self):
        with open(REPLAY_BUFFER_FILENAME, "wb") as f:
            pickle.dump(self.buffer, f)

def dump_player_position_graph(model, filename):
    with open(filename, "w") as f:
        f.write("x [domain]\t" +
                "\t".join(action if action != "" else "None"
                          for action in game.ACTIONS) + "\n")
        for x in range(128):
            game_state = game.GameState()
            game_state.set_entity_state(0, game.TYPE_PLAYER_SHOT, x, 42)
            phi = Phi(None, game_state)
            input_tensor = phi.input_tensor()
            actions = model.predict(input_tensor[np.newaxis,:], verbose=0)[0]
            f.write(str(x) + " " + " ".join(actions.astype(str)) + "\n")

def learn():
    # Parameters of the neural network.
    input_tensor_shape = Phi(None, game.GameState()).input_tensor().shape
    #print("phi", Phi(None, game.GameState()))
    #print("input_tensor", Phi(None, game.GameState()).input_tensor())
    print("input_tensor_shape", input_tensor_shape)
    action_count = len(game.ACTIONS)

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    replay_buffer.load()

    dqn = DQN(input_tensor_shape[0], action_count, replay_buffer, LEARNING_RATE,
              MINIBATCH_SIZE, DOUBLE_DQN, GAMMA)
    #print("Initial weights:")
    #dqn.print_weights()

    if False:
        dqn.load_weights()

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
            predictions_acc = None
            predictions_count = 0
            while True:
                loss, mae, predictions = dqn.update()

                if predictions_acc is None:
                    predictions_acc = predictions
                else:
                    predictions_acc += predictions
                predictions_count += 1

                if learn_step % 1000 == 0:
                    predictions_acc /= predictions_count
                    predictions_file.write(str(learn_step) + " " +
                                           " ".join(predictions_acc.mean(axis=0).astype(str)) + "\n")
                    predictions_file.flush()
                    predictions_acc = None
                    predictions_count = 0
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
                    #dqn.print_weights()
                if learn_step % 1000 == 0:
                    dqn.save_weights()
                if learn_step % 1000 == 0:
                    dump_player_position_graph(dqn.model, "model_player.txt")
                    dump_player_position_graph(dqn.target_model, "target_model_player.txt")
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

def print_actions_graph(actions, action):
    min_q = actions.min()
    max_q = actions.max()
    diff = max_q - min_q
    if diff == 0:
        return

    SIZE = 20

    def make_band(i, ch):
        value = max(int(round((actions[i] - min_q)/diff*SIZE)), 1)
        left = (SIZE - value)//2
        right = SIZE - value - left
        s = " "*left + ch*value + " "*right
        if i == action:
            s = chr(27) + "[1;33m" + s + chr(27) + "[0m"
        return s

    print()
    print(make_band(4, "<") + "  " + make_band(3, "^") + "  " + make_band(5, ">"))
    print(make_band(1, "<") + "  " + make_band(0, ".") + "  " + make_band(2, ">"))
    print()

def play():
    # Parameters of the neural network.
    input_tensor_shape = Phi(None, game.GameState()).input_tensor().shape
    print("input_tensor_shape", input_tensor_shape)
    action_count = len(game.ACTIONS)

    dqn = DQN(input_tensor_shape[0], action_count, REPLAY_BUFFER_SIZE, LEARNING_RATE,
              MINIBATCH_SIZE, DOUBLE_DQN, GAMMA)

    dqn.load_weights()
    #dqn.print_weights()

    live_game = game.LiveGame(False)

    # An episode is a full play of the game.
    episode = 0
    for episode in range(EPISODE_COUNT):
        now = time.perf_counter()

        # Run the game.
        seed = episode
        game_state = live_game.start_new_game(seed)
        game_step = 0

        # Initialize our game state.
        phi = Phi(None, game_state)

        # Each iteration is an action chosen in the game.
        while True:
            input_tensor = phi.input_tensor()
            #print(input_tensor)
            time.sleep(0.040)
            actions = dqn.model.predict(input_tensor[np.newaxis,:], verbose=0)[0]
            action = np.argmax(actions)
            print_actions_graph(actions, action)
            if game_step < 120 and action >= 3:
                # Don't shoot to avoid initial death.
                action -= 3
            #print(input_tensor, actions, action, game.ACTIONS[action])

            # Execute action in game, get next state.
            game_state = live_game.perform_action(action)
            next_phi = Phi(phi, game_state)
            if game_state.game_over:
                print("Game over, score =", game_state.score)
                break

            phi = next_phi
            game_step += 1

        episode += 1

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
                if DEBUG:
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

