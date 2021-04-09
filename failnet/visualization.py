import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def init_plot():
    # Set some good default settings
    plt.figure(figsize=(15,7))
    plt.xlabel("X", fontsize=18, labelpad=5)
    plt.ylabel("Y", fontsize=18, labelpad=5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

def draw_signal(signal, color=None, linestyle=None, label=None) -> None:
    """
    Plots signals with random colors.

    Args:
        signals (array): data to be plotted.
        color (str): color for the signal in the plot.
        linestyle (str): linestyle for the signal in the plot.
        label (str): label for the signal. Shown in legend.
    """
    if color is None:
        color = np.random.rand(3,)
    if label is None:
        label = "signal"
    if linestyle is None:
        linestyle = '-'

    x = np.arange(len(signal)) # signals can have different length
    plt.plot(x, signal, linestyle=linestyle, color=color, label=label)
