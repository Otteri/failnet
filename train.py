import torch
import argparse
import numpy as np
import gym
import pulsegen
import config as cfg
import matplotlib.pyplot as plt
from enum import IntEnum
from pathlib import Path
from visualization import draw_signal, init_plot
from model import Model, Batch

# This enum allows to reference feature vectors in an abstract manner.
class Channel(IntEnum):
    SIG1 = 0 # First measurement channel.

# Brief:
# The model tries to learn provided periodical signal from input data.
# Learned signal pattern can be leveraged for doing something useful.
# Expects data to have form: [B, S, L], where B is batch size,
# S is number of signals and L is the signal length.

def parse_args() -> argparse:
    """
    Parses provided comman line arguments.

    Returns:
        argparse: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, help="steps to run")
    parser.add_argument("--show_input", default=False, action="store_true", help="Visualizes input data used for training")
    parser.add_argument("--make_plots", default=False, action="store_true", help="Visualizes learning process during training")
    args = parser.parse_args()
    return args

def filter_signal(signal, n=3) -> torch.tensor:
    """
    Preprocesses input data by average filtering filtering it.
    This reduces noise levels, which can boost learning.
    Padding can be added, so that data dimensions stay the same.

    Args:
        signal (1d array): Data to be processed.
        n (int, optional): Moving average filter window size.

    Returs:
        filtered_signal: signal that has been filtered
    """
    # TODO: Unit test that shape stays same -- that padding works
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    padded_signal = torch.cat((signal[0:(n-1)], signal[:]))
    filtered_signal = moving_average(padded_signal, n)
    return filtered_signal

def get_data_batch(env) -> (torch.tensor, torch.tensor):
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
    n = cfg.predict_n
    input_data = Batch(cfg.repetitions, 1, cfg.signal_length-n)
    target_data = Batch(cfg.repetitions, 1, cfg.signal_length-n)
    for i in range(0, cfg.repetitions):
        signal = env.recordRotation(viz=args.show_input)
        input_data[i, Channel.SIG1] = signal[:-n]
        target_data[i, Channel.SIG1] = signal[n:]
    return input_data, target_data

def create_plot(iteration, signal1, signal2, signal3):
    """
    A helper function for creating plots.

    Args:
        iteration (int): Used for naming save file.
        signal 1,2,3 (3d array): Data for plotting.
    """
    init_plot()
    draw_signal(signal1[0, Channel.SIG1], linestyle='-',  color='b', label='input')
    draw_signal(signal2[0, Channel.SIG1], linestyle='--', color='r', label='filtered input')
    draw_signal(signal3[0, Channel.SIG1], linestyle='-',  color='g', label='learning outcome')
    plt.legend()
    plt.savefig(f"predictions/prediction_{iteration+1}.svg")

def main(args):

    env = gym.make("FourierSeries-v0", config_path="config.py")

    # Create a new model
    model = Model(
        training      = True,
        device        = cfg.device,
        signal_length = cfg.signal_length,
        hidden        = cfg.hidden,
        predict_n     = cfg.predict_n,
        lr            = cfg.learning_rate,
        max_iter      = cfg.max_iter,
        history_size  = cfg.history_size
    )

    # Start training
    for i in range(args.steps):
        print("STEP:", i)

        # 1) Get data
        train_input, train_target = get_data_batch(env)   # Use different data for \
        test_input, test_target = get_data_batch(env)     # training and testing...
        unfiltered_test_input = test_input.data.clone() # for visualization

        # 2) Preprocess all data: filter first channel signal
        for data in [train_input, train_target, test_input, test_target]:
            for batch in data:
                signal = batch[Channel.SIG1]
                batch[Channel.SIG1] = filter_signal(signal, n=3)

        # 3) Train the model with collected data
        model.train(train_input, train_target)

        # 4) Check how the model is performing
        y = model.predict(test_input, test_target)

        # 5) Visualize performance
        if args.make_plots:
            create_plot(i, unfiltered_test_input, test_input, y)

    # Save outcome
    model.save_model("failnet.pt")

if __name__ == "__main__":

    # Create a directory for weights and plots
    Path(cfg.data_dir).mkdir(exist_ok=True)

    # Read command line arguments
    args = parse_args()

    # Run training loop
    main(args)
