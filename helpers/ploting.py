'''
    Module for ploting data.
'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy
import matplotlib
import pickle
import logging
import matplotlib as mpl

matplotlib.use('Agg')
logging.getLogger('matplotlib.font_manager').disabled = True
# Fix for matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def PlotConfigure():
    ''' Configure ploting options.'''
    matplotlib.use('Agg')


def plot_rewards(filepath: str,
                 rewards : list,
                 data_label : str = 'Rewards',
                 xlabel : str = 'Episodes',
                 ylabel : str = 'Rewards',
                 ):
    '''
        Cretes rewards/iteration plot and saves it to file.

        Parameters
        ----------
        filepath : str
            Filepath to save plot.

        rewards : list
            List of rewards values for every training episode, where
            list index position is episode number.        
    '''
    # Check : rewards is not None
    if (rewards is None):
        logging.error('RewardsIterationPlot: rewards is None')
        return

    # Figure : Create
    figure_handle = plt.figure(figsize=(16.0, 9.0))

    # Continuous line plot
    plt.plot(rewards, label=data_label)

    # Grid, formatting and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    # Figure : Save to file
    figure_handle.savefig(filepath)


