import torch
import argparse
import numpy as np
import gym
import pulsegen
import config as cfg
from pathlib import Path
from visualize import plot
from model import Model, Sequence, Channel

# Brief:
# The model tries to learn provided periodical signal from input data.
# Learned signal pattern can be leveraged for doing something useful.
# Expects data to have form: [B, S, L], where B is batch size,
# S is number of signals and L is the signal length.

def parseArgs():
    """
    Parses provided comman line arguments.

    Returns:
        argparse: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, help="steps to run")
    parser.add_argument("--show_input", default=False, action="store_true", help="Visualizes input data used for training")
    parser.add_argument("--make_plots", default=False, action="store_true", help="Visualizes learning process during training")
    parser.add_argument("--invert", default=False, action="store_true", help="Invert learning outcome")
    args = parser.parse_args()
    return args

def filterSignal(signal, n=3):
    """
    Preprocesses input data by average filtering filtering it.
    This reduces noise levels, which can boost learning.
    Padding can be added, so that data dimensions stay the same.

    Args:
        signal (1d array): Data to be processed.
        n (int, optional): Moving average filter window size.
    """
    # TODO: Unit test that shape stays same -- that padding works
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    padded_signal = torch.cat((signal[0:(n-1)], signal[:]))
    filtered_signal = moving_average(padded_signal, n)
    return filtered_signal

def getDataBatch(env):
    """
    Collects data which can be used for training.
    Data is a 3d array: [B, S, L], where B is batch
    S is recorded signal and L is recorded data value index.

    Args:
        env (pulsegen): Gym envinment used for data generation.

    Returns:
        input_data: training input data.
        target_data: predictions made by the model, should match to this.
    """
    data = env.recordRotations(rotations=cfg.repetitions, viz=args.show_input)

    # Shift datavectors. If input: x[k], then target: x[k+n]
    n = cfg.predict_n
    input_data = torch.from_numpy(data[..., :-n])
    target_data = torch.from_numpy(data[..., n:])
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
        unfiltered_test_input = data["test_input"].clone() # for visualization

        # 2) Preprocess data: filter it
        for batch_name, batch_data in data.items():
            for batch in batch_data:
                signal = batch[Channel.SIG1, :]
                batch[Channel.SIG1, :] = filterSignal(signal)

        # 3) Train the model with collected data
        model.train(data["train_input"], data["train_target"])

        # 4) Check how the model is performing
        y = model.predict(data["test_input"], data["test_target"])

        # 5) Visualize performance
        if args.make_plots:
            plot(unfiltered_test_input, data["test_input"], y[:, Channel.SIG1, :], i, args.invert)

    # Save outcome
    torch.save(model.seq.state_dict(), f"{cfg.data_dir}/weights.mdl")

if __name__ == "__main__":

    # Create a directory for weights and plots
    Path(cfg.data_dir).mkdir(exist_ok=True)

    # Read command line arguments
    args = parseArgs()

    # Run training loop
    main(args)
