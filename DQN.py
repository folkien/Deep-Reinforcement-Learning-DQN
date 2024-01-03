
import time
import numpy as np
import gym
import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
import random
from collections import deque
from keras.callbacks import TensorBoard

class DQN:

    def __init__(self, 
                 gym_game, 
                 epsilon=1, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01, 
                 batch_size=32, 
                 discount_factor=0.9, 
                 num_of_episodes=500,
                 is_render:bool=False
                 ):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.num_of_episodes = num_of_episodes
        self.game = gym_game
        self.is_render = is_render

        self.environment = gym.make(gym_game,
                                    render_mode="human"
                                    # metadata={'render.modes': ['human', 'rgb_array']}
                                    )

        try:
            try:
                shape = self.environment.observation_space.shape
                self.state_size = (shape[0], shape[1], shape[2])
                self.s = shape[0]*shape[1]*shape[2]
                self.state_mode = "observation"
            except:
                self.state_size = self.environment.observation_space.shape[0]
                self.s = self.state_size
                self.state_mode = "information"
            self.state_container = "box"
        except:
            self.state_size = self.environment.observation_space.n
            self.state_mode = "information"
            self.state_container = "discrete"

        try:
            self.action_size = self.environment.action_space.shape[0]
        except:
            try:
                self.action_size = self.environment.action_space.n
            except:
                self.action_size = self.environment.action_space.shape[0]
        self.a = self.action_size

        print("state size is: ",self.state_size)
        print("action size is: ", self.action_size)
        self.memory = deque(maxlen=20000)
        self.model = self.create_model()
        self.tensorboard = TensorBoard('./logs', update_freq=1)
        print(self.model.summary())

    def create_model(self):
        """
        neural network model
        :return:
        """
        try:
            input = Input(shape=(self.state_size))
        except:
            input = Input(shape=(self.state_size,))

        if self.state_mode == "information":
            # if the state is not an image
            out = Dense(24, activation="relu")(input)
            out = Dense(24, activation="relu")(out)
            out = Dense(self.action_size, activation="linear")(out)
        elif self.state_mode == "observation":
            # if the state is an image
            out = Conv2D(128, kernel_size=(5,5), padding="same", activation="relu")(input)
            out = MaxPooling2D()(out)
            out = Conv2D(128, kernel_size=(3,3), padding="same", activation="relu")(out)
            out = MaxPooling2D()(out)
            out = Flatten()(out)
            out = Dense(24, activation="relu")(out)
            out = Dense(24, activation="relu")(out)
            out = Dense(self.action_size, activation="linear")(out)

        model = Model(inputs=input, outputs=out)
        model.compile(optimizer="adam", loss="mse")
        return model

    def state_reshape(self, state):
        shape = state.shape
        if self.state_mode == "observation":
            return np.reshape(state, [1, shape[0], shape[1], shape[2]])
        elif self.state_mode == "information":
            return np.reshape(state, [1, shape[0]])

    def act(self, state):
        """
        act randomly by probability of epsilon or predict the next move by the neural network model
        :param state:
        :return:
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, next_state, action, reward, done):
        """
        remember the experience
        :param state:
        :param next_state:
        :param action:
        :param reward:
        :param done:
        :return:
        """
        self.memory.append((state, next_state, action, reward, done))

    def replay(self):
        """
        experience replay. find the q-value and train the neural network model with state as input and q-values as targets
        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        batch = random.choices(self.memory,k=self.batch_size)
        for state, next_state, action, reward, done in batch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.max(self.model.predict(next_state)[0])
            final_target = self.model.predict(state)
            final_target[0][action] = target
            self.model.fit(state,
                           final_target,
                           verbose=0,
                            callbacks=[self.tensorboard]
                           )



    def play(self):
        """
        play for num_of_episodes. remember the experiences in each episode. replay the experience at the end of the episode
        :return:
        """
        start_time = time.time()
        for episode in range(self.num_of_episodes+1):
            episode_time_start = time.time()
            state_tuple = self.environment.reset()
            state = self.state_reshape(state_tuple[0])
            r = []
            t = 0
            while True:
                # Model : Act
                action = self.act(state)

                # Model : Get the next state
                next_state, reward, done, _result, _dictionary = self.environment.step(action)

                # Mode : Render
                if self.is_render:
                    self.environment.render()

                next_state = self.state_reshape(next_state)
                self.remember(state, next_state, action, reward, done)
                state = next_state
                r.append(reward)
                t += 1

                if done:
                    episode_time = time.time() - episode_time_start
                    r = np.mean(r)
                    print("episode number: ", episode,", reward: ",r , "time score: ", t, "episode time: ", episode_time)
                    self.save_info(episode, r, t)
                    break

            self.replay()

        total_time = time.time() - start_time

        print(f"Training finished.\n{self.num_of_episodes} episodes trained in {total_time} seconds.")
        print(f"Avg episode time: {total_time / self.num_of_episodes} seconds.")

        # Model : Save the model
        self.model.save("./Model/DQN-"+self.game+"-"+str(self.num_of_episodes)+"-episodes-batchsize-"+str(self.batch_size))


    def save_info(self, episode, reward, time):
        file = open("./Plot/DQN-"+self.game+"-"+str(self.num_of_episodes)+"-episodes-batchsize-"+str(self.batch_size), 'a')
        file.write(str(episode)+" "+str(reward)+" "+str(time)+" \n")
        file.close()


# game = "Pong-v0" # bd
game = "CartPole-v1" # bd
dqn = DQN(game, 
          num_of_episodes=500,
          is_render=True
          )
dqn.play()
