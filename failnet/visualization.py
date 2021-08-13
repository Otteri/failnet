import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def init_plot() -> bool:
    """
    Initializes a plot: sets a good configuration
    values for our use case.
    """
    # Set some good default settings
    plt.figure(figsize=(15,7))
    plt.xlabel("X", fontsize=18, labelpad=5)
    plt.ylabel("Y", fontsize=18, labelpad=5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    return True

def draw_signal(signal, color=None, linestyle=None, label=None) -> bool:
    """
    Plots signals with random colors.

    Args:
        signals (array): data to be plotted.
        color (str): color for the signal in the plot.
        linestyle (str): linestyle for the signal in the plot.
        label (str): label for the signal. Shown in legend.

    Returns:
        True when signal has been drawn to the figure
    """
    if color is None:
        color = np.random.rand(3,)
    if label is None:
        label = "signal"
    if linestyle is None:
        linestyle = '-'

    x = np.arange(len(signal)) # signals can have different length
    plt.plot(x, signal, linestyle=linestyle, color=color, label=label)
    return True

def create_plot(filename, signal1, signal2, signal3):
    """
    A helper function for creating plots.

    Args:
        filename (str): Used for naming the plot file.
        signal 1,2,3 (1d array): Data for plotting.
    """
    assert len(signal1) == len(signal2), "Signal 1 & 2 lengths must be equal"
    assert len(signal2) == len(signal3), "Signal 2 & 3 lengths must be equal"

    init_plot()
    draw_signal(signal1, linestyle='-',  color='b', label='input')
    draw_signal(signal2, linestyle='--', color='r', label='filtered input')
    draw_signal(signal3, linestyle='-',  color='g', label='learning outcome')
    plt.legend()
    plt.savefig(filename)
