
'''
    Agent base class for reinforcement learning training
    and playing.
'''
from dataclasses import dataclass, field
import numpy as np
import gym
import random
from collections import deque, namedtuple
from keras.callbacks import TensorBoard

# Episode step as namedtuple
EpisodeStep = namedtuple('EpisodeStep',
                         ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class ReinforcementAgent:
    ''' Reinforcement learning agent for training and playing.'''
    # Environment : The environment handle to train and play in.
    env: gym.Env = field(init=True, default=None)

    # Learning rate - Means how much we value new information compared to previous information.
    alpha: float = 0.8
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
    # Memory size - How many experiences we store in memory
    memory_size: int = field(default=10000)
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
        self.memory = deque(maxlen=self.memory_size)

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
                 verbose: int = 0,
                 use_multiprocessing: bool = False) -> int:
        ''' Fit the model state -> target.'''
        raise NotImplementedError('ModelFit method is not implemented.')

    def Act(self, state: tuple) -> int:
        ''' Act according to epsilon-greedy policy.'''
        # Random : Act randomly by probability of epsilon
        if np.random.rand() < self.epsilon:
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
        self.memory.append(EpisodeStep(state=state,
                                       action=action,
                                       reward=reward,
                                       next_state=next_state,
                                       done=done
                                       ))

    def Train(self, num_episodes: int):
        ''' Use experiences in memory to train the agent.'''
        # After 50% of training time alpha should decay to min_alpha
        alpha_decay = (self.min_alpha / self.alpha) ** (1.0 /
                                                        (num_episodes * 0.5))

        # Epsilon decay - used for decreasing epsilon over time.
        #   epsilon' = epsilon * epsilon_decay
        #   - means how much we decay epsilon after each episode. (reduce exploration time)
        #   - calculate based on trainig episodes number
        #   - After 80% of training time epislon should decay to min_epsilon
        epsilon_decay = (self.min_epsilon /
                         self.epsilon) ** (1.0 / (num_episodes * 0.8))

        # Loop of training : For N episodes
        for episode in range(num_episodes):
            # Training completness : Calculate
            training_completness = episode / num_episodes * 100

            # Info : Print episode informations
            print(
                f'Episode {episode}/{num_episodes} {training_completness:2.2f}%, epsilon: {self.epsilon:2.2f}')

            # Episode : Play single episode of the environment
            cum_reward = self.Play()

            # Replay : Replay memory stored experiences
            self.Replay(alpha_decay, epsilon_decay)

            # # Cumulative rewards : Add cumulative reward for this episode
            # training_cum_rewards.append(cum_reward)

    def Replay(self, alpha_decay: float, epsilon_decay: float):
        ''' Replay memory stored experiences.'''
        # Check : If memory size is less than minimum size, return
        if (self.memory_len < 1):
            return

        # Memory : Sample a batch (randomly because of correlation between experiences)
        batch: list[EpisodeStep] = random.choices(self.memory,
                                                  k=self.batch_size)

        # Batch : Replay training
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

            # Model : Fit for (state, new policy).
            self.ModelFit(step.state,
                          policy)

        # Learning rate : Decay alpha after each episode
        self.alpha = max(self.alpha * alpha_decay, self.min_alpha)

        # Exploration : Decay epsilon after each episode
        self.epsilon = max(self.epsilon * epsilon_decay, self.min_epsilon)

    def Play(self) -> float:
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
            action = self.Act(state)

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

    def Save(self, episode: int, reward: float, time: int):
        ''' Save model.'''
        # @TODO: Implement training
