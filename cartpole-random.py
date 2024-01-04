import gym

# Inicjalizacja środowiska CartPole-v1
env = gym.make("CartPole-v1")

# Przykładowa interakcja ze środowiskiem
state = env.reset()
done = False

cum_reward = 0
while not done:
    # Wybór losowej akcji
    action = env.action_space.sample()
    
    # Wykonanie akcji w środowisku
    next_state, reward, done, truncated, info = env.step(action)

    # Dodanie nagrody do sumy nagród
    cum_reward += reward
    
    # Aktualizacja stanu
    state = next_state

    # Wyświetlenie informacji o stanie
    print(f"Stan: {state}, Nagroda: {reward}/{cum_reward}, Czy zakończony: {done}")

# Zamknięcie środowiska
env.close()
