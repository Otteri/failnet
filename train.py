from pathlib import Path
import argparse
import torch
import numpy as np
import gym
import pulsegen
import config as cfg
from visualize import plot
from model import Model, Sequence, Channel

DATA_DIR = "predictions" # Location where weights and plots are saved to

# Brief:
# Functions in here can be used for training the model defined in model.py

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15, help="steps to run")
    parser.add_argument("--show_input", default=False, action="store_true", help="Visualizes input data used for training")
    parser.add_argument("--make_plots", default=False, action="store_true", help="Visualizes learning process during training")
    parser.add_argument("--invert", default=False, action="store_true", help="Invert learning outcome")
    args = parser.parse_args()
    return args

# Avg-filter input signal, since it can be quite noisy and we don't want to try learn noise.
# Add padding by copying first few values in the beginning, so the data vector length does not change.
# input data: [B, 2, N]
# n: filtering window size
def preprocessBatch(input_data, n=3):

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    filtered_data = input_data.clone()
    for i in range(0, input_data.size(1)):
        padded_input_data = torch.cat((input_data[i, Channel.SIG1, 0:(n-1)], input_data[i, Channel.SIG1, :]))
        filtered_data[i, Channel.SIG1, :] = moving_average(padded_input_data, n)
    
    return filtered_data

# Gather a data batch with N-rows
# batch_num x signals_num x signal_len
def getDataBatch(env):
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
            data[batch_name] = preprocessBatch(batch_data, n=5)

        # 3) Train the model with collected data
        model.train(data["train_input"], data["train_target"])

        # 4) Check how the model is performing
        y = model.predict(data["test_input"], data["test_target"])

        # 5) Visualize performance
        if args.make_plots:
            plot(unfiltered_test_input, data["test_input"], y[:, 1, :], i, args.invert)

    # Save outcome
    torch.save(model.seq.state_dict(), f"{DATA_DIR}/weights.mdl")

if __name__ == "__main__":

    # Create a directory for weights and plots
    Path(DATA_DIR).mkdir(exist_ok=True)

    args = parseArgs()

    main(args)
