from collections import namedtuple
import gym
import numpy as np
import math

# CartPole-v1 state:
CartPoleState = namedtuple('CartPoleState', ['cart_pos', 'cart_vel', 'pole_angle', 'pole_ang_vel'])

# Inicjalizacja środowiska
env = gym.make('CartPole-v1')

# Podział stanu na przedziały (dyskretyzacja na 20 przedziałów)
n_states = [20, 20, 20, 20] # Liczba przedziałów dla każdego elementu stanu

# Ograniczenia dla stanu
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

# Inicjalizacja tabeli Q.
# - Mamy 4 elementy stanu (obserwacji) zdykretyzowane na 20 przedziałów,
# - Mamy 2 akcje (0 lub 1),
# Pary (stan, akcja) są indeksowane w tabeli Q.
# - czyli mamy 20*20*20*20*2=320000 pól w tabeli Q. WOW!
q_table_shape = n_states + [env.action_space.n]
q_table = np.zeros(q_table_shape)

# Learning rate - used for updating Q-values.
#  - Means how much we value new information compared to previous information.
alpha = 0.1
# Discount factor - used for updating Q-values
# - Means how much we value future rewards compared to present rewards.
gamma = 0.99

# Epsilon - used for exploration.
# - Means how much we choose to explore instead of exploit.
epsilon = 1.0
# Epsilon decay - used for decreasing epsilon over time.
# - Means how much we decay epsilon after each episode. (reduce exploration time)
epsilon_decay = 0.995
# Min epsilon
# - Means minimum value % of exploration.
min_epsilon = 0.01

# Funkcja dyskretyzująca stan
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

def choose_action(state : tuple) -> int:
    ''' Choose an action for the current state according to the Q policy'''
    # Exploration : If random number < epsilon
    if np.random.random() < epsilon:
        return env.action_space.sample() # Losowa akcja (eksploracja)
    
    # Q Policy : Best q value for this state
    return np.argmax(q_table[state]) # Najlepsza znana akcja (eksploatacja)

# Funkcja aktualizacji tabeli Q
def update_q_table(state : tuple, action : int, reward : float, next_state : tuple):
    ''' Update the Q table with the new Q value'''
    # Q-Table : Get the best q value for the next state
    best_next_action = np.argmax(q_table[next_state])

    # Q-Table : Compute the TD error (Bellman equation)
    q_table[state + (action,)] += alpha * (reward + gamma * q_table[next_state + (best_next_action,)] - q_table[state + (action,)])

# Loop of training : For N episodes
for episode in range(1000):
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

    epsilon = max(epsilon * epsilon_decay, min_epsilon) # Zmniejszanie eksploracji

env.close()


