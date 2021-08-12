import torch
import numpy as np

def filter_signal(signal, n=3) -> torch.tensor:
    """
    Preprocesses input data by average filtering filtering it.
    This reduces noise levels, which can boost learning.
    Padding can be added, so that data dimensions stay the same.
    Padding is added to the beginning of the signal and it is the
    first value in the array. 

    Args:
        signal (1d array): Data to be processed.
        n (int, optional): Moving average filter window size.

    Returs:
        filtered_signal: signal that has been filtered
    """
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    padding = torch.empty(n-1).fill_(signal[0])
    padded_signal = torch.cat((padding, signal), 0)
    filtered_signal = moving_average(padded_signal, n)
    return filtered_signal
