import gym
import numpy as np
import math

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

# Parametry uczenia
alpha = 0.1
# gamma = 0.9
gamma = 0.99

# Parametry eksploracji
epsilon = 1.0
epsilon_decay = 0.995
# Min epsilon
min_epsilon = 0.01

# Funkcja dyskretyzująca stan
def discretize_state(state : tuple, state_bounds : list, n_states : list):
    discretized = []
    for i in range(len(state)):
        scaling = (state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])
        new_state = int(round((n_states[i] - 1) * scaling))
        new_state = min(n_states[i] - 1, max(0, new_state))
        discretized.append(new_state)

    return tuple(discretized)

# Funkcja wyboru akcji
def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample() # Losowa akcja (eksploracja)
    else:
        return np.argmax(q_table[state]) # Najlepsza znana akcja (eksploatacja)

# Funkcja aktualizacji tabeli Q
def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
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
        action = choose_action(state)
        next_state, reward, done, truncated,  _ = env.step(action)
        next_state = discretize_state(next_state, state_bounds, n_states)
        update_q_table(state, action, reward, next_state)
        state = next_state

    epsilon = max(epsilon * epsilon_decay, min_epsilon) # Zmniejszanie eksploracji

env.close()


