
'''
    Agent base class for reinforcement learning training
    and playing.
'''
from dataclasses import dataclass, field
import time
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
import random
from helpers.ReinforcementAgent import ReinforcementAgent


@dataclass
class ReinforcementAgentDQN(ReinforcementAgent):
    ''' Reinforcement learning agent for training and playing.'''
    # Neural network model
    model: keras.Model = field(init=False, default=None)

    def ___post_init__(self):
        ''' Initialize object.'''

    def ModelPredict(self,
                     state: tuple,
                     verbose: int = 0,
                     use_multiprocessing: bool = False) -> int:
        ''' Predict an action from a state.'''
        return self.model.predict(state,
                                  verbose=verbose,
                                  use_multiprocessing=use_multiprocessing)

    def ModelFit(self,
                 state: tuple,
                 target: tuple,
                 verbose: int = 0,
                 use_multiprocessing: bool = False) -> int:
        ''' Fit the model state -> target.'''
        return self.model.fit(state,
                              target,
                              use_multiprocessing=True,
                              callbacks=[self.tensorboard])

    def Init(self):
        ''' Initialize the agent. '''
        # Model : Initialize model
        # State -> 4x4x -> Action
        out = Dense(4, activation='relu')(out)
        out = Dense(4, activation='relu')(out)
        out = Dense(self.action_size, activation='linear')(out)

        model = Model(inputs=input, outputs=out)
        model.compile(optimizer='adam', loss='mse')
        return model
