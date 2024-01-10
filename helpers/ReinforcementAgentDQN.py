
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
        '''
            Predict an action from a state.

            Parameters
            ----------
            state : tuple
                Current state.
            verbose : int
                Verbosity mode.
            use_multiprocessing : bool
                Use multiprocessing.

            Returns
            -------
            int
                Predicted action.
        '''
        return self.model.predict(self.state_reshape(state),
                                  verbose=verbose,
                                  use_multiprocessing=use_multiprocessing)

    def ModelFit(self,
                 input_batch: tuple,
                 output_batch: tuple,
                 batch_size: int = 32,
                 verbose: int = 0,
                 use_multiprocessing: bool = False) -> int:
        ''' Fit the model state -> target.'''
        return self.model.fit(input_batch,
                              output_batch,
                              batch_size=batch_size,
                              epochs=10,
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

    def Save(self, directorypath: str = 'Model'):
        ''' Save model.'''
        if (self.model is None):
            raise Exception('Model not initialized.')

        # Model : Save the model
        self.model.save(
            f'{directorypath}/DQN_{self.num_epochs}_{self.batch_size}')
