from pathlib import Path
from enum import IntEnum
import argparse
import torch
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config as cfg
import pulsegen
from model import Model, Sequence

# This model tries to learn a periodical signal from provided input data.
# After learning, the model can predict future values of similar signal.
# Generate_data.py can be used to generate input data for the model.
#
# number of recordings x rotations x signal length
# Saved data: [angle, signal1]
# Data referes to the three dimensional block of data, wherease signal is just a 1d vector

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=15, help="steps to run")
parser.add_argument("--show_input", default=False, action="store_true", help="Visualize input training data")
parser.add_argument("--invert", default=False, action="store_true", help="Invert learning outcome")
parser.add_argument("--use_sim", action='store_true', default=None, help="Use simulator for data generation")
parser.add_argument("--use_speed", action='store_true', default=False, help="When using sim, learns from 0: torque, 1: speed")
args = parser.parse_args()

if args.use_sim:
    from runsim import recordRotations

ave_n = 5 # Moving avg filtering window

def configure(args):
    # Configure
    if not args.show_input:
        matplotlib.use("Agg")

    # Create a directory for weights and plots
    data_directory = "predictions"
    Path(data_directory).mkdir(exist_ok=True)

# Define enum, so it is easy to update meaning of data indices
class Channel(IntEnum): # Channels in data block
    ANGLE = 0
    SIGNAL1 = 1 # torque / speed

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot(input_data, filtered_input, learned, iteration, invert=False):
    plt.figure(figsize=(15,7))
    plt.xlabel("X", fontsize=18, labelpad=5)
    plt.ylabel("Y", fontsize=18, labelpad=5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    x = np.arange(input_data.size(2)) #input_length
    plt.plot(x, input_data[0, 1, :], "-", color='b', label="input")
    plt.plot(x, filtered_input[0, 1, :], '--', color='r', label="filtered input")
    
    if invert: # compensate (mirrors signal)
        plt.plot(x, -learned[0, :], '-', color='green', label="learning output")    
        plt.plot(x, input_data[0, 1, :] - learned[0, :], '-', color='black', label="sum")
    else:
        plt.plot(x, learned[0, :], '-', color='green', label="learning output")
    
    plt.legend()

    plt.savefig(f"predictions/prediction_{iteration+1}.svg")
    plt.close()

# Avg-filter input signal, since it can be quite noisy and we don't want to try learn white noise.
# Add padding by copying first few values in the beginning, so the data vector length does not change.
# input data: [B, 2, N]
def preprocessBatch(input_data, n=1):
    filtered_data = input_data.clone()
    for i in range(0, input_data.size(1)):
        padded_input_data = torch.cat((input_data[i, Channel.SIGNAL1, 0:(n-1)], input_data[i, Channel.SIGNAL1, :]))
        filtered_data[i, Channel.SIGNAL1, :] = moving_average(padded_input_data, n=n)
    return filtered_data

# Gather a data batch with N-rows
# batch_num x signals_num x signal_len
def getDataBatch(env):
    data = np.empty((cfg.repetitions, 2, cfg.signal_length), 'float64')

    if args.use_sim:
        data = recordRotations(rotations=cfg.repetitions, data_length=cfg.signal_length, use_speed=args.use_speed)
    else:    
        data = env.recordRotations(rotations=cfg.repetitions, viz=args.show_input)

    # Shift datavectors. If input: x[k], then target: x[k+1]
    input_data = torch.from_numpy(data[..., :-1])
    target_data = torch.from_numpy(data[..., 1:])

    return input_data, target_data


def main(args):

    env = gym.make("FourierSeries-v0", config_path="config.py")

    data = {
        "train_input"  : [],
        "train_target" : [],
        "test_input"   : [],
        "test_target"  : []
    }

    # Realize requested settings
    configure(args)

    # Create a new model
    model = Model(device=cfg.device)

    # Start training
    for i in range(args.steps):
        print("STEP:", i)

        # 1) Get data
        data["train_input"], data["train_target"] = getDataBatch(env) # Use different data for \
        data["test_input"], data["test_target"] = getDataBatch(env)   # training and testing...
        unfiltered_test_input = data["test_input"] # for visualization

        # 2) Preprocess data: filter it
        for batch_name, batch_data in data.items():
            data[batch_name] = preprocessBatch(batch_data, n=ave_n)

        # 3) Train the model with collected data
        model.train(data["train_input"], data["train_target"])

        # 4) Check how the model is performing
        y = model.predict(data["test_input"], data["test_target"])

        # 5) Visualize performance
        plot(unfiltered_test_input, data["test_input"], y[:, 1, :], i, args.invert)

    # Save outcome
    torch.save(model.seq.state_dict(), "predictions/compensator.mdl")
    np.savetxt("predictions/compensation-pattern.csv", np.array([y[0, 0, :], y[0, 1, :]]), delimiter=",")

if __name__ == "__main__":
    main(args)
