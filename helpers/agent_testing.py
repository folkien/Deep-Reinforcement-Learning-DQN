from time import sleep
import gym




def agent_test(env_test : gym.Env, 
               state_bounds : list,
               n_states : int,
               choose_action : callable,
               discretize_state : callable,
               delay_time : float = None,
               ) -> float:
    ''' Test model on single environment episode.'''
    # State : Get initial episode state
    state_initial_dict = env_test.reset()

    # State : Discretize the state to values according to n_states discretization
    state = discretize_state(state_initial_dict[0], state_bounds, n_states)

    # Episode : Process episode until not done
    done = False
    cum_reward = 0
    while not done:
        # Action : Choose action according to epsilon-greedy policy
        action = choose_action(state, force_optimal=True)

        # Episode : Take action and get next state and reward
        next_state, reward, done, truncated,  _ = env_test.step(action)
        cum_reward += reward

        # State : Discretize the next state to values according to n_states discretization
        next_state = discretize_state(next_state, state_bounds, n_states)

        # Informations : Print
        print(f"EnvTest : State {state}, action {action}, total reward {cum_reward}.")
        
        # Time : Sleep
        if (delay_time is not None):
            sleep(delay_time)

        # State : Update state
        state = next_state

    return cum_reward


