
'''
    Agent base class for reinforcement learning training
    and playing.
'''
from dataclasses import dataclass, field
import time
import numpy as np
import gym
import random
from collections import deque, namedtuple
from keras.callbacks import TensorBoard
import tensorflow as tf

# Episode step as namedtuple
EpisodeStep = namedtuple('EpisodeStep',
                         ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class ReinforcementAgent:
    ''' Reinforcement learning agent for training and playing.'''
    # Environment : The environment handle to train and play in.
    env: gym.Env = field(init=True, default=None)

    # Training epochs
    num_epochs: int = field(init=True, default=None)
    # Learning rate - Means how much we value new information compared to previous information.
    alpha: float = 0.9
    # Min alpha - minimum value of learning rate
    min_alpha: float = 0.1
    # Epsilon - Means how much we choose to explore instead of exploit.
    epsilon: float = 1.0
    # Min epsilon
    min_epsilon: float = 0.05
    # Discount factor  - Means how much we value future rewards compared to present rewards.
    gamma: float = 0.99

    # Batch size - How many experiences we use in each training batch
    batch_size: int = field(default=32)
    # Batches number per epoch
    num_batches: int = field(init=True, default=4)
    # Memory size - How many experiences we store in memory
    memory_size: int = field(default=1000)
    # Memory : deque - A list-like data structure that can be efficiently appended to and popped from either side.
    memory: deque = field(init=False, default=None)

    # Tensorboard : Tensorboard callback for logging
    tensorboard: TensorBoard = field(init=False, default=None)

    def __post_init__(self):
        ''' Initialize object.'''
        # Environment : If missing then raise error
        if (self.env is None):
            raise ValueError('Environment is missing.')

        # Memory : Initialize memory
        self.memory = deque(maxlen=self.batch_size * self.num_batches)

        # Tensorboard : Initialize tensorboard
        self.tensorboard = TensorBoard('./logs', update_freq=1)

        # Model : Initialize model
        self.Init()

    @property
    def action_size(self) -> int:
        ''' Get the action size.'''
        if (self.env is None):
            return 0

        if (len(self.env.action_space.shape) > 0):
            return self.env.action_space.shape[0]

        return self.env.action_space.n

    @property
    def state_size(self) -> int:
        ''' Get the state size.'''
        if (self.env is None):
            return 0

        if (len(self.env.observation_space.shape) > 0):
            return self.env.observation_space.shape[0]

        return self.env.observation_space.n

    @property
    def memory_len(self) -> int:
        ''' Get the memory length.'''
        return len(self.memory)

    def IsMemoryFull(self) -> bool:
        ''' Check if memory is full.'''
        return self.memory_len == self.memory.maxlen

    def Init(self):
        '''Initialize the agent and model. '''
        raise NotImplementedError('Init method is not implemented.')

    def ModelPredict(self,
                     state: tuple,
                     verbose: int = 0,
                     use_multiprocessing: bool = False) -> int:
        ''' Predict action from the model in state.'''
        raise NotImplementedError('ModelAction method is not implemented.')

    def ModelFit(self,
                 state: tuple,
                 target: tuple,
                 batch_size: int = 32,
                 verbose: int = 0,
                 use_multiprocessing: bool = False) -> int:
        ''' Fit the model state -> target.'''
        raise NotImplementedError('ModelFit method is not implemented.')

    def Act(self,
            state: tuple,
            force_optimal: bool = False) -> int:
        ''' Act according to epsilon-greedy policy.'''
        # Random : Act randomly by probability of epsilon, if not force optimal
        if (np.random.rand() < self.epsilon) and (not force_optimal):
            return self.env.action_space.sample()

        # Model : Predict
        predictions = self.ModelPredict(state)

        # Return : Return the action with the highest probability
        return np.argmax(predictions[0])

    def Forget(self):
        ''' Forget all the experiences in memory.'''
        self.memory.clear()

    def Remember(self,
                 state: tuple,
                 action: tuple,
                 reward: float,
                 next_state: tuple,
                 done: int):
        ''' Remember the experience in memory.'''
        # Check : If done is True, then do not remember
        if (done is True):
            return

        # Memory : Add experience to memory
        self.memory.append(EpisodeStep(state=state,
                                       action=action,
                                       reward=reward,
                                       next_state=next_state,
                                       done=done
                                       ))

    def Train(self, num_epochs: int) -> list[float]:
        ''' Use experiences in memory to train the agent.'''
        # List of training cumulative rewards
        training_rewards = []
        # Store num of epochs
        self.num_epochs = num_epochs
        # Number of play episodes in one epoch
        num_play_episodes = 10
        # Number of train/replay episodes in one epoch
        num_replay_episodes = 10

        # After 80% of training time alpha should decay to min_alpha
        alpha_decay = (self.min_alpha / self.alpha) ** (1.0 /
                                                        (num_epochs * 0.8))

        # Epsilon decay - used for decreasing epsilon over time.
        #   epsilon' = epsilon * epsilon_decay
        #   - means how much we decay epsilon after each episode. (reduce exploration time)
        #   - calculate based on trainig episodes number
        #   - After 80% of training time epislon should decay to min_epsilon
        epsilon_decay = (self.min_epsilon /
                         self.epsilon) ** (1.0 / (num_epochs * 0.8))

        # Time : Start time
        start_time = time.time()

        # Loop of training : For N episodes
        for epoch in range(num_epochs):
            # Training completness : Calculate
            training_completness = epoch / num_epochs * 100

            # Memory : Clear memory (for epoch and new model)
            self.Forget()

            # Info : Print last episode informations
            print(f'Epoch {epoch}/{num_epochs} {training_completness:2.2f}%,'
                  f'epsilon: {self.epsilon:2.2f}, learning rate: {self.alpha:2.2f}')

            # 1. Memory collecting
            # - Play N episodes and collect experiences in memory
            print(f'Collecting full memory for replay...')
            while (not self.IsMemoryFull()):
                self.Play()

            # 2. Train/Replay
            # - Replay memory stored experiences.
            print(
                f'Replaying {self.memory_len} memories for {num_replay_episodes} replay episodes...')
            self.Replay_vec(epochs=num_replay_episodes)

            # Learning rate : Decay alpha after each episode
            self.alpha = max(self.alpha * alpha_decay, self.min_alpha)
            # Exploration : Decay epsilon after each episode
            self.epsilon = max(self.epsilon * epsilon_decay, self.min_epsilon)

            # 3. Statistics
            # - Model current policy reward.
            reward = self.Play(force_optimal=True)
            print(f'Play avg. reward: {reward}')
            training_rewards.append(reward)

        # Time : End time
        end_time = time.time()

        # Info : Print training time
        print(f'Training time: {end_time - start_time}s')
        print(
            f'Training time per epoch: {(end_time - start_time) / num_epochs}s')

        return training_rewards

    def Replay(self):
        ''' Replay memory stored experiences.'''
        # Check : If memory size is less than minimum size, return
        if (self.memory_len < 1):
            return

        # Batch : Sample from memory (randomly because of correlation between experiences)
        batch: list[EpisodeStep] = random.choices(self.memory,
                                                  k=self.batch_size)

        # Training : Input batch (states)
        input_batch = np.array([step.state for step in batch])
        # Training : Output batch (new policies)
        output_batch = []
        # Batch : Convert to (input, output) pairs.
        for step in batch:
            # Check : Episode ended, do nothing (missing next state)
            if step.done is None:
                continue

            # # Q-Table : Compute the TD error (Bellman equation)
            # q_table[state + (action,)] += alpha * (reward + gamma *
            #                                     q_table[next_state + (best_next_action,)] - q_table[state + (action,)])

            # Policy : Get model policy for current state
            policy = self.ModelPredict(step.state)
            # Policy (next step) : Get model policy for next state
            policy_next = self.ModelPredict(step.next_state)
            # Policy best action (next step) : Get best action from policy
            policy_next_best_action = np.max(policy_next[0])

            # Reward : Update reward
            next_reward = self.alpha * \
                (step.reward + self.gamma * policy_next_best_action)

            # Policy : Update
            policy[0][step.action] = next_reward

            # Output batch : Add policy
            output_batch.append(policy[0])

        # Output batch : Convert to numpy array
        output_batch = np.array(output_batch)

        # Create tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_batch, output_batch))

        # Model : Fit for created batch (input, output)
        self.ModelFit(input_batch=dataset, output_batch=None, batch_size=None)

    def Replay_vec(self, epochs: int = 1, batches: int = 4):
        '''
            Replay memory stored experiences. Code vectorized
            for better performance of training.

            Parameters
            ----------
            epochs : int
                Number of epochs to train.
        '''
        # Check : If memory size is less than minimum size, return
        if len(self.memory) == 0:
            return

        # Batches : Sample from memory (randomly because of correlation between experiences)
        batch = random.sample(self.memory,
                              k=self.batch_size*batches)

        # Batches: Convert to (input, output) pairs numpy arrays in a single operation.
        states, next_states, rewards, actions = zip(*[(step.state, step.next_state, step.reward, step.action)
                                                      for step in batch])
        # Convert to numpy arrays
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        actions = np.array(actions)

        # Przewidywanie dla obecnego i następnego stanu (2x forward pass dla całego batcha)
        current_q = self.model.predict(states)
        next_q = self.model.predict(next_states)

        # Aktualizacja wartości Q
        for i in range(len(batch)):
            current_q[i, actions[i]] = rewards[i] + \
                self.gamma * np.max(next_q[i])

        # Trenowanie modelu za pomocą aktualnych wartości Q
        self.model.fit(states,
                       current_q,
                       batch_size=self.batch_size,
                       epochs=epochs,
                       use_multiprocessing=True,
                       callbacks=[self.tensorboard])

    def Play(self, force_optimal: bool = False) -> float:
        ''' Play single episode of the environment.'''
        # State : Get initial episode state
        state_initial_dict = self.env.reset()
        state = state_initial_dict[0]

        # Cumulative reward : Reset
        cum_reward = 0
        # Episode : Process episode until not done
        done = False
        while not done:
            # Action : Get action
            action = self.Act(state, force_optimal=force_optimal)

            # Episode : Take action and get next state and reward
            next_state, reward, done, truncated,  _ = self.env.step(action)

            # Model : Remember the experience
            self.Remember(state=state,
                          action=action,
                          reward=reward,
                          next_state=next_state,
                          done=done)

            # State : Update state
            state = next_state

            # Cumulative reward : Add reward
            cum_reward += reward

        return cum_reward

    def Save(self, directorypath: str):
        ''' Save model.'''
        raise NotImplementedError('Save method is not implemented.')
