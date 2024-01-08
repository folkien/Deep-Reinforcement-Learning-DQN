from collections import namedtuple
from enum import Enum
import sys
import gym
import numpy as np
import pandas as pd
from helpers.agent_testing import agent_test
from helpers.ploting import plot_checkpoint_rewards, plot_rewards
from helpers.ReinforcementAgentDQN import ReinforcementAgentDQN

# CartPole-v1 state:
CartPoleState = namedtuple(
    'CartPoleState', ['cart_pos', 'cart_vel', 'pole_angle', 'pole_ang_vel'])

# CartPole-v1 action: enum
# - 0 - push cart to the left
# - 1 - push cart to the right


class CartPoleAction(int, Enum):
    Left = 0
    Right = 1


# Number of training episodes
num_episodes = 60
# Number of episodes to test/average model
num_episodes_test = 7
# Validation checkpoints : Episode number when to test
validation_checkpoints = [0,
                          100,
                          250,
                          500,
                          1_000,
                          2_500,
                          5_000,
                          10_000,
                          25_000,
                          50_000,
                          100_000,
                          250_000,
                          500_000]


# Inicjalizacja środowiska
env = gym.make('CartPole-v1')

# Agent : Initialize
agent = ReinforcementAgentDQN(env=env)

# Agent : Train
training_rewards = agent.Train(num_epochs=num_episodes)

# Training rewards : Plot to file
models_directory = 'Model'
filename = f'dqn_rewards_{num_episodes}_episodes.png'
plot_rewards(f'{models_directory}/{filename}',
             training_rewards,
             data_label='Best cum. rewards reached during training')


# Preview : Test model on single episode with human rendering
# --------------------------------------------------------
# Inicjalizacja środowiska CartPole-v1
env_test = gym.make('CartPole-v1',
                    render_mode='human')

# Test : Test model on single episode
agent.env = env_test
reward = agent.Play(force_optimal=True)
print(f'Test play: Reward: {reward}')

# Environment : Close
env_test.close()
sys.exit(0)


# Test : Test model on many episodes and get average reward
# --------------------------------------------------------
print(f'Testing model on environment.')

# Inicjalizacja środowiska CartPole-v1
env_test = gym.make('CartPole-v1')

# Rewards : Get rewards for 10 episodes
rewards = [agent_test(env_test,
                      state_bounds,
                      n_states,
                      choose_action,
                      discretize_state,
                      delay_time=None)
           for index in range(num_episodes_test)]

# Rewards : Plot to file
filename = f'testing_rewards_{episode}_episodes_{epsilon}_epsilon_{alpha}_alpha.png'
plot_rewards(f'{models_directory}/{filename}',
             rewards,
             data_label=f'Best cum. rewards reached during testing {len(rewards)} episodes')

# Info : Print average reward and standard deviation
print(f'Testing : Average reward: {np.mean(rewards)}')
print(f'Testing : Standard deviation: {np.std(rewards)}')


# Results dataframe : Load if exists
tests_directory = 'tests'
try:
    results_df = pd.read_csv(
        f'{tests_directory}/results.csv', sep=';', index_col=False)
except:
    results_df = pd.DataFrame(
        columns=['episodes', 'reward_mean', 'reward_std'])


# Dataframe : Create current results dataframe
results_current_df = pd.DataFrame(columns=['episodes', 'reward_mean', 'reward_std'],
                                  data=[[num_episodes, np.mean(rewards), np.std(rewards)]])

# Results dataframe : Concatenate
results_df = pd.concat([results_df, results_current_df], axis=0)

# Results dataframe : Save
results_df.to_csv(f'{tests_directory}/results.csv', index=False, sep=';')


# Environment : Close
env_test.close()
