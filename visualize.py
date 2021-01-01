import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model import Channel

def plot(input_data, filtered_input, learned, iteration, invert=False):
    plt.figure(figsize=(15,7))
    plt.xlabel("X", fontsize=18, labelpad=5)
    plt.ylabel("Y", fontsize=18, labelpad=5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    x = np.arange(input_data.size(2)) # input_length
    plt.plot(x, input_data[0, Channel.SIG1, :], "-", color='b', label="input")
    plt.plot(x, filtered_input[0, Channel.SIG1, :], '--', color='r', label="filtered input (f)")
    
    if invert: # (mirrors the signal horizontally)
        plt.plot(x, -learned[0, :], '-', color='green', label="learning outcome (l)")
        plt.plot(x, input_data[0, Channel.SIG1, :] - learned[0, :], '-', color='black', label="f * l")
    else:
        plt.plot(x, learned[0, :], '-', color='green', label="learning outcome (l)")
    
    plt.legend()

    plt.savefig(f"predictions/prediction_{iteration+1}.svg")
    plt.close()
