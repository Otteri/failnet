import gym
import torch
import numpy as np
import pulsegen
from enum import IntEnum
from pathlib import Path
from failnet.visualization import create_plot
from failnet.model import Model, Batch
from failnet.filters import filter_signal
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, date

# This enum allows to reference feature vectors in an abstract manner.
class Channel(IntEnum):
    SIG1 = 0 # First measurement channel.

# Brief:
# The model tries to learn provided periodical signal from input data.
# Learned signal pattern can be leveraged for doing something useful.
# Expects data to have form: [B, S, L], where B is batch size,
# S is number of signals and L is the signal length.

def get_data_batch(env, cfg, show_input=False) -> (torch.tensor, torch.tensor):
    """
    Collects data which can be used for training.
    Data is a 3d array: [B, S, L], where B is batch
    S is recorded signal and L is recorded data value index.
    Calculation signal length is signal_length - predict_n, i.e.
    intersection between input and target batches:
    input:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target:    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] -> size 9

    Args:
        env (pulsegen): Gym envinment used for data generation.
        cfg: configurations, same file as used creating the `env`
        show_input (bool): Optional flag, requires matplotlib

    Returns:
        input_data: training input data.
        target_data: predictions made by the model, should match to this.
    """
    n = cfg.predict_n
    input_data = Batch(cfg.repetitions, 1, cfg.signal_length-n)
    target_data = Batch(cfg.repetitions, 1, cfg.signal_length-n)
    for i in range(0, cfg.repetitions):
        signal = env.record_rotation(viz=show_input)
        input_data[i, Channel.SIG1] = signal[:-n]
        target_data[i, Channel.SIG1] = signal[n:]
    return input_data, target_data

def train(args, cfg):

    env = gym.make("PeriodicalSignal-v0", config_path=args.config_path)
    today = date.today().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H-%M-%S")
    if args.tensorboard:
        writer = SummaryWriter(f"{cfg.data_dir}/runs/{today}/{time}")

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
        print("STEP:", i+1)

        # 1) Get data
        train_input, train_target = get_data_batch(env, cfg, args.show_input)   # Use different data for \
        test_input, test_target = get_data_batch(env, cfg, args.show_input)     # training and testing...
        unfiltered_test_input = test_input.data.clone() # for visualization

        # 2) Preprocess all data: filter signal in first channel
        for data in [train_input, train_target, test_input, test_target]:
            for batch in data:
                signal = batch[Channel.SIG1]
                batch[Channel.SIG1] = filter_signal(signal, n=3)

        # 3) Train the model with collected data
        loss = model.train(train_input, train_target)

        # Record some data for TensorBoard
        if args.tensorboard:
            writer.add_scalar("Training loss", loss, i)
            writer.add_histogram("Linear/gradient", model.seq.linear.weight.grad, i)
            writer.add_histogram("Linear/weight", model.seq.linear.weight, i)
            writer.add_histogram("Linear/bias", model.seq.linear.bias, i)
            writer.add_histogram("Conv/gradient", model.seq.conv.weight.grad, i)
            writer.add_histogram("Conv/weight", model.seq.conv.weight, i)
            writer.add_histogram("Conv/bias", model.seq.conv.bias, i)
            writer.flush()

        # 4) Check how the model is performing
        y = model.predict(test_input, test_target)

        # 5) Visualize prediction performance
        if args.make_plots:
            create_plot(
                f"{cfg.data_dir}/predictions/prediction_{i+1}.svg",
                unfiltered_test_input[0, Channel.SIG1],
                test_input[0, Channel.SIG1],
                y[0, Channel.SIG1]
            )

    # Save outcome
    model.save_model("data/failnet.pt")

    # Write TensorBoard metrics
    if args.tensorboard:
        writer.add_hparams(
            {"lr": cfg.learning_rate,
            "steps": args.steps,
            "hidden": cfg.hidden,
            "max_iter": cfg.max_iter,
            "history_size": cfg.history_size},
            {"final loss": loss},
            run_name="." # write to same directory
        )
        writer.flush()
        writer.close()
