from collections import namedtuple
from enum import Enum
from time import sleep, time
import gym
import numpy as np
import pandas as pd
from helpers.agent_testing import agent_test
import math

# CartPole-v1 state:
CartPoleState = namedtuple('CartPoleState', ['cart_pos', 'cart_vel', 'pole_angle', 'pole_ang_vel'])

# CartPole-v1 action: enum
# - 0 - push cart to the left
# - 1 - push cart to the right
class CartPoleAction(int, Enum):
    Left = 0
    Right = 1

# Inicjalizacja środowiska
env = gym.make('CartPole-v1')

# Podział stanu na przedziały (dyskretyzacja na 20 przedziałów)
n_states = [50, 50, 50, 50] # Liczba przedziałów dla każdego elementu stanu

# Ograniczenia dla stanu
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

# Inicjalizacja tabeli Q.
# - Mamy 4 elementy stanu (obserwacji) zdykretyzowane na 20 przedziałów,
# - Mamy 2 akcje (0 lub 1),
# Pary (stan, akcja) są indeksowane w tabeli Q.
# - czyli mamy 20*20*20*20*2=320000 pól w tabeli Q. WOW!
# - Wartości w tabeli Q są inicjalizowane na 0 (nagród nie znamy).
q_table_shape = n_states + [env.action_space.n]
q_table = np.zeros(q_table_shape)

# Number of training episodes
num_episodes = 5000

# Learning rate - used for updating Q-values.
# -------------------------------
#  - Means how much we value new information compared to previous information.
alpha = 0.8
# Min alpha - minimum value of learning rate
min_alpha = 0.1
# Alpha decay - used for decreasing alpha over time.
#   alpha' = alpha * alpha_decay
#   - means how much we decay alpha after each episode. (reduce learning rate over time)
#   - calculate based on trainig episodes number
#   - After 50% of training time alpha should decay to min_alpha
alpha_decay = (min_alpha / alpha) ** (1.0 / (num_episodes * 0.5))



# Discount factor - used for updating Q-values
# - Means how much we value future rewards compared to present rewards.
gamma = 0.99

# Epsilon - used for exploration.
# -------------------------------
# - Means how much we choose to explore instead of exploit.
epsilon = 1.0
# Min epsilon
# - Means minimum value % of exploration.
min_epsilon = 0.05
# Epsilon decay - used for decreasing epsilon over time.
#   epsilon' = epsilon * epsilon_decay
#   - means how much we decay epsilon after each episode. (reduce exploration time)
#   - calculate based on trainig episodes number
#   - After 80% of training time epislon should decay to min_epsilon
epsilon_decay = (min_epsilon / epsilon) ** (1.0 / (num_episodes * 0.8))


def discretize_state(state : np.ndarray, 
                     state_bounds : list[tuple], 
                     n_states : list):
    ''' Discretize a state vector to return a tuple of integers '''
    discretized = []

    # State  : For each state variable
    for i in range(len(state)):
        # State : Get the bounds for this variable
        scaling = (state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])

        # State : Get the value for this variable
        new_state = int(round((n_states[i] - 1) * scaling))
        new_state = min(n_states[i] - 1, max(0, new_state))

        # State : Add to the list of discretized values
        discretized.append(new_state)

    return CartPoleState(*discretized)

def choose_action(state : tuple,
                  force_optimal : bool = False
                  ) -> CartPoleAction:
    ''' Choose an action for the current state according to the Q policy'''
    # Exploration : If random number < epsilon and not force optimal
    if (np.random.random() < epsilon) and (not force_optimal):
        return CartPoleAction(env.action_space.sample())
    
    # Q Policy : Best q value for this state (argmax returns the index of the max value)
    return CartPoleAction(np.argmax(q_table[state]))

# Funkcja aktualizacji tabeli Q
def update_q_table(state : tuple, action : CartPoleAction, reward : float, next_state : tuple):
    ''' Update the Q table with the new Q value'''
    # Action : as int
    action = int(action)

    # Q-Table : Get the best q value for the next state
    best_next_action = np.argmax(q_table[next_state])


    # Q-Table : Compute the TD error (Bellman equation)
    q_table[state + (action,)] += alpha * (reward + gamma * q_table[next_state + (best_next_action,)] - q_table[state + (action,)])

# Loop of training : For N episodes
for episode in range(num_episodes):
    # Training completness : Calculate
    training_completness = episode / num_episodes * 100

    # Info : Print episode informations
    print(f"Episode {episode} started {training_completness:2.2f}%. Epsilon: {epsilon}")

    # State : Get initial episode state
    state_initial_dict = env.reset()

    # State : Discretize the state to values according to n_states discretization
    state = discretize_state(state_initial_dict[0], state_bounds, n_states)

    # Episode : Process episode until not done
    done = False
    while not done:
        # Action : Choose action according to epsilon-greedy policy
        action = choose_action(state)

        # Episode : Take action and get next state and reward
        next_state, reward, done, truncated,  _ = env.step(action)

        # State : Discretize the next state to values according to n_states discretization
        next_state = discretize_state(next_state, state_bounds, n_states)

        # Q-Table : Update Q-table
        update_q_table(state, action, reward, next_state)

        # State : Update state
        state = next_state

    # Learning rate : Decay alpha after each episode
    alpha = max(alpha * alpha_decay, min_alpha)

    # Exploration : Decay epsilon after each episode
    epsilon = max(epsilon * epsilon_decay, min_epsilon) 


# Qtable filename with number of training episodes, epsilon and alpha
models_directory = "Model"
filename = f"q_table_{episode}_episodes_{epsilon}_epsilon_{alpha}_alpha.npy"

# Q-Table : Save Q-table as numpy array pickle
np.save(f"{models_directory}/{filename}", q_table)


# Environment : Close
env.close()

# Test : Test model on many episodes and get average reward
# --------------------------------------------------------
print(f"Testing model on environment.")

# Inicjalizacja środowiska CartPole-v1
env_test = gym.make("CartPole-v1")

# Rewards : Get rewards for 10 episodes
rewards = [ agent_test(env_test, 
                    state_bounds, 
                    n_states, 
                    choose_action, 
                    discretize_state, 
                    delay_time=None)
            for index in range(30) ]

# Info : Print average reward and standard deviation
print(f"Testing : Average reward: {np.mean(rewards)}")
print(f"Testing : Standard deviation: {np.std(rewards)}")


# Results dataframe : Load if exists
tests_directory = "tests"
try:
    results_df = pd.read_csv(f"{tests_directory}/results.csv", sep=";", index_col=False)
except:
    results_df = pd.DataFrame(columns=['episodes', 'reward_mean', 'reward_std'])
    

# Dataframe : Create current results dataframe    
results_current_df = pd.DataFrame(columns=['episodes', 'reward_mean', 'reward_std'],
                                    data=[[num_episodes, np.mean(rewards), np.std(rewards)]])

# Results dataframe : Concatenate
results_df = pd.concat([results_df, results_current_df], axis=0)

# Results dataframe : Save
results_df.to_csv(f"{tests_directory}/results.csv", index=False, sep=";")


# Environment : Close
env_test.close()

# Preview : Test model on single episode with human rendering
# --------------------------------------------------------
# Inicjalizacja środowiska CartPole-v1
env_test = gym.make("CartPole-v1", 
                    render_mode="human")

# Test : Test model on single episode
agent_test(env_test, 
           state_bounds, 
           n_states, 
           choose_action, 
           discretize_state, 
           delay_time=0.1)

# Environment : Close
env_test.close()




