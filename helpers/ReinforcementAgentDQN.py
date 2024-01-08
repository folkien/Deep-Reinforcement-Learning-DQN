
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
from keras.utils import plot_model
from helpers.ReinforcementAgent import ReinforcementAgent


@dataclass
class ReinforcementAgentDQN(ReinforcementAgent):
    ''' Reinforcement learning agent for training and playing.'''
    # Neural network model
    model: keras.Model = field(init=False, default=None)

    def __post_init__(self):
        ''' Initialize object.'''
        # Parent : Initialize parent
        super().__post_init__()

    def state_reshape(self, state: np.ndarray):
        ''' Reshape current state to add
            batch size dimension as first. '''
        return np.reshape(state, [1, state.shape[0]])

    def ModelPredict(self,
                     state: tuple,
                     verbose: int = 0,
                     use_multiprocessing: bool = False) -> int:
        ''' Predict an action from a state.'''
        return self.model.predict(self.state_reshape(state),
                                  verbose=verbose,
                                  use_multiprocessing=use_multiprocessing)

    def ModelFit(self,
                 state: tuple,
                 target: tuple,
                 verbose: int = 0,
                 use_multiprocessing: bool = False) -> int:
        ''' Fit the model state -> target.'''
        return self.model.fit(self.state_reshape(state),
                              self.state_reshape(target),
                              use_multiprocessing=True,
                              callbacks=[self.tensorboard])

    def Init(self):
        ''' Initialize the agent. '''
        # Input : Create input layer
        state_input = Input(shape=(self.state_size,))

        # Layers : Create layers
        out = Dense(4, activation='relu')(state_input)
        out = Dense(4, activation='relu')(out)
        out = Dense(self.action_size, activation='linear')(out)

        # Model : Create model and compile
        model = Model(inputs=state_input, outputs=out)
        model.compile(optimizer='adam', loss='mse')
        self.model = model

        # Model : Plot model to file (clasname.png inside /Model directory)
        plot_model(model,
                   to_file=f'Model/{self.__class__.__name__}.png',
                   show_shapes=True,
                   show_layer_names=True)
