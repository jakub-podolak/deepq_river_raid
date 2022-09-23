import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.base import Base
from actions import NUM_ACTIONS, LEFT, NONE, RIGHT


class DeepQModel(Base):
    def build_model(self):
        inputs = layers.Input(shape=(80, 80, 3))

        # Convolution layers to process a frame
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        # Flatten layer to proceed to regression
        layer4 = layers.Flatten()(layer3)

        # Decision layers
        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(NUM_ACTIONS, activation="linear")(layer5)

        return keras.Model(inputs=inputs, outputs=action)


    def __init__(self, env, hyperparams):
        self.env = env 

        self.number_of_runs = 1000
        self.max_frames_for_run = 1000 
        self.target_points = 200

        self.gamma = 0.995 # discounting factor
        self.epsilon = 1 # for random-exploration actions
        self.epsilon_min = 0.1
        self.epsilon_max = 1
        self.epsilon_interval = (
            self.epsilon_max - self.epsilon_min
        ) 
        self.epsilon_random_frames = 50000 # for this number of frames - only random and observe
        self.epsilon_random_decrease = 100000 # for this number of frames decrease probability of random action

        self.batch_size = 16  # how many frames to take from memory
        self.max_memory_length = 100000

        # less stable model that predicts makes actions based on predicted Q-values
        self.model_action = self.build_model()

        # more stable model that predicts Q-values to compare with gained reward.
        # Based on that loss can be calculated and model_action can learn. 
        # After some time model_reward is updated to model_action
        self.model_reward = self.build_model()
    

    def learn_from_past(self):
        pass


    def train(self):
        run_number = 0
        total_frames = 0
        # update reward network each 10000 frames
        update_reward_network = 10000
        update_action_network = 16

        history = {
            'action': [],
            'state': [],
            'state_next': [],
            'end': [],
            'reward': []
        }

        rewards_history = []
        mean_reward = 0

        loss_function = keras.losses.Huber()
        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        # run desired number of games or until reached specified number of points
        while run_number < self.number_of_runs:
            run_number += 1

            # reset to new run
            self.env.reset_state()

            state = self.env.get_current_state()
            run_reward = 0

            for frame in range(0, self.max_frames_for_run):

                # exploration - initial frames or by chance
                if total_frames < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    # take random action
                    action = np.random.choice(NUM_ACTIONS)
                else:
                    # predict q-values for each action using model_action
                    state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
                    action_rewards = self.model_action(state_tensor, training=False)
                    action = tf.argmax(action_rewards[0]).numpy()

                # decay probability of random choice - model knows more
                self.epsilon -= self.epsilon_interval / self.epsilon_random_decrease
                self.epsilon = max(self.epsilon, self.epsilon_min)

                # apply action in environment
                next_state, reward, end = self.env.step(action)
                end = 1 if end else 0

                run_reward += reward

                # memorize this frame
                history['action'].append(action)
                history['state'].append(state)
                history['state_next'].append(next_state)
                history['end'].append(end)
                history['reward'].append(reward)

                # move to the next state
                state = next_state

                # update action model only if we have enough data
                if total_frames % update_action_network == 0 and len(history['state']) > self.batch_size:
                    
                    # get moments to evaluate move and reward
                    moment_indices = np.random.choice(range(len(history['state'])), size=self.batch_size)

                    state_sample = np.array([history['state'][i] for i in moment_indices])
                    state_next_sample = np.array([history['state_next'][i] for i in moment_indices])
                    rewards_sample = [history['reward'][i] for i in moment_indices]
                    action_sample = [history['action'][i] for i in moment_indices]
                    done_sample = tf.convert_to_tensor(
                        [float(history['end'][i]) for i in moment_indices]
                    )

                    # predict future rewards from these moments using model_reward (assume it does
                    # best future choices)
                    future_rewards = self.model_reward.predict(state_next_sample)

                    # expected Q-values: observed reward + gamma * future reward
                    expected_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

                    # end frames (game overs) should lead to Q-value -1
                    expected_q_values = expected_q_values * (1 - done_sample) - done_sample

                    # mask to calculate loss only on these selected Q-values
                    mask = tf.one_hot(action_sample, NUM_ACTIONS)

                    # train the action model - compare predicted q_values with those from model_reward
                    with tf.GradientTape() as tape:
                        q_values = self.model_action(state_sample)

                        # apply the masks to the Q-values to get only the Q-value for taken actions
                        q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)

                        # calculate loss between Q-value expected and this predicted my model_action
                        loss = loss_function(expected_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, self.model_action.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model_action.trainable_variables))

                # update reward model to catch up with model_action
                if total_frames % update_reward_network == 0:
                    self.model_reward.set_weights(self.model_action.get_weights())

                total_frames += 1

                # delete oldest memory record if reached limit
                if len(history['reward']) > self.max_memory_length:
                    del history['reward'][0]
                    del history['state'][0]
                    del history['state_next'][0]
                    del history['action'][0]
                    del history['end'][0]

                # if crashed we break this run
                if end:
                    print('Run {} Ended with reward {:.1f}, Total Frames: {}, Mean reward: {:.1f}'\
                        .format(run_number, run_reward, total_frames, mean_reward))
                    break
            
            rewards_history.append(run_reward)
            if len(rewards_history) > 100:
                del rewards_history[0]
            mean_reward = np.mean(rewards_history)

    def evaluate(self, state):
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action_rewards = self.model_action(state_tensor, training=False)
        action = tf.argmax(action_rewards[0]).numpy()
        return action


    def get_name(self):
        return "DeepQ"


    def plot(self):
        pass