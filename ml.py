
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam

MODEL_WEIGHTS_FILENAME = "model.weights.h5"
TARGET_MODEL_WEIGHTS_FILENAME = "target_model.weights.h5"

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
        #middle_layer_size = 32

        # Create neural net.
        self.model = Sequential([
            InputLayer( (input_size,) ),
            # relu = rectified linear unit = max(0, x)
            Dense(middle_layer_size, activation='relu'),
            Dense(middle_layer_size, activation='relu'),
            Dense(middle_layer_size, activation='relu'),
            #Dense(middle_layer_size, activation='relu', kernel_initializer="zeros", bias_initializer="zeros"),
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
        transitions = self.replay_buffer.sample(self.minibatch_size)

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
        targets = rewards + self.gamma*best_futures

        # Update targets for the actions we actually took.
        actions = np.array([transition.action for transition in transitions])
        fixed_predictions = predictions.copy()
        fixed_predictions[np.arange(fixed_predictions.shape[0]),actions] = targets

        # Perform gradient descent.
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
