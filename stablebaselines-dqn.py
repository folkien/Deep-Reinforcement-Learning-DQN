from dataclasses import dataclass, field
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th

env = gym.make('Taxi-v3')


class EmbeddingExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, 10)
        self.mlp_extractor = nn.Sequential(
            # 500 is the size of the state space, 10 is the embedding size
            nn.Embedding(500, 10),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp_extractor(observations)


policy_kwargs = dict(
    features_extractor_class=EmbeddingExtractor,
)

# PPO
# - Jaki mode użyje PPO? Jak rozłozy observation_space na inputy?
# - Jakie są akcje?
model = PPO('MlpPolicy',
            env,
            verbose=1,
            tensorboard_log='./logs',
            policy_kwargs=policy_kwargs,
            )


# Model : Train
model.learn(total_timesteps=20_000,
            progress_bar=True, )


# Create an environment for test, set the parallelism of the environment to 9, and set the rendering mode to group_human.
env = gym.make('Taxi-v3',
               render_mode='human')
model.set_env(env)  # The agent requires an interactive environment.
# Initialize the environment to obtain initial observations and environmental information.
obs, info = env.reset()
while True:

    # Action : Predict with action_mask and get action
    action, _states = model.predict([obs], deterministic=True)

    # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
    obs, r, done, trunacted, info = env.step(action[0])
    if (done):
        break

env.close()  # Close test environment
