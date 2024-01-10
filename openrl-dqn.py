# train_ppo.py
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.modules.common import DQNNet as DQNNet
from openrl.runners.common import PPOAgent as Agent
from openrl.runners.common import DQNAgent as DQNAgent

# Tensorboard : Add graph


# Create an environment and set the environment parallelism to 9.
env = make('CartPole-v1', env_num=9)

# Create neural network.
net = DQNNet(env)

# Initialize the agent.
agent = DQNAgent(net,
                 run_dir='./logs',
                 use_tensorboard=True)
# Start training and set the total number of steps to 20,000 for the running environment.
agent.train(total_time_steps=50_000)

# Create an environment for test, set the parallelism of the environment to 9, and set the rendering mode to group_human.
env = make('CartPole-v1', env_num=9, render_mode='group_human')
agent.set_env(env)  # The agent requires an interactive environment.
# Initialize the environment to obtain initial observations and environmental information.
obs, info = env.reset()
while True:
    # The agent predicts the next action based on environmental observations.
    action, _ = agent.act(obs)
    # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
    obs, r, done, info = env.step(action)
    if any(done):
        break
env.close()  # Close test environment
