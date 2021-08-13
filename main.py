import argparse
import os
from pathlib import Path
from failnet.train import train

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
    parser.add_argument("--console_only", default=False, action="store_true", help="Use only console window")
    parser.add_argument("--config_path", type=str, default="failnet/config.py", help="Path to a custom config")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Read command line arguments
    args = parse_args()

    # Add directory including the config to search path
    import sys
    abs_path = os.path.dirname(os.path.realpath(args.config_path))
    print(f"[INFO] Loading config '{abs_path}'...")
    sys.path.append(abs_path)

    # Load configuration values
    import config as cfg
    args.config_path = os.path.abspath(args.config_path)

    # Create a directory for weights and plots
    Path(cfg.data_dir).mkdir(exist_ok=True)

    if args.console_only:
        # Use different matplotlib backend. This allows us to save plots
        # without them being rendered. Without GUI, the process just gets
        # stuck if we try to visualize without this arg being set to true.
        matplotlib.use('Agg')

    # Run training loop
    train(args, cfg)
