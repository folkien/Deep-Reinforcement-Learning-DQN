'''
    Module for ploting data.
'''
import matplotlib.pyplot as plt
import matplotlib
import logging

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


def plot_checkpoint_rewards(filepath: str,
                    checkpoint_rewards : list[tuple],
                    data_label : str = 'Rewards',
                    xlabel : str = 'Episodes',
                    ylabel : str = 'Rewards',
                    ):
    '''
        Creates (episode_number, reward) plot and saves it to file.

        Parameters
        ----------
        filepath : str
            Filepath to save plot.

        rewards : list[tuple]
            List of (episode_number, reward) tuples.
    '''
    # Check : rewards is not None
    if (checkpoint_rewards is None):
        logging.error('rewards is None')
        return

    # Figure : Create
    figure_handle = plt.figure(figsize=(16.0, 9.0))

    # Separated tuples to x,y lists mapping zip
    x, y = zip(*checkpoint_rewards)

    # Plot x,y 
    plt.plot(x, y, label=data_label)

    # Grid, formatting and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    # Figure : Save to file
    figure_handle.savefig(filepath)


