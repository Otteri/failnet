import argparse
import os
import sys
import matplotlib
from pathlib import Path

def parse_args() -> argparse:
    """
    Parses provided comman line arguments.

    Returns:
        argparse: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", default=False, action="store_true", help="Train model")
    parser.add_argument("--steps", type=int, default=10, help="steps to run")
    parser.add_argument("--show_input", default=False, action="store_true", help="Visualizes input data used for training")
    parser.add_argument("--make_plots", default=False, action="store_true", help="Visualizes learning process during training")
    parser.add_argument("--console_only", default=False, action="store_true", help="Use only console window")
    parser.add_argument("--config_path", type=str, default="failnet/config.py", help="Path to a custom config")
    parser.add_argument("--make_onnx", default=False, action="store_true", help="Generate ONNX model from trained model")
    parser.add_argument("--run_onnx", default=False, action="store_true", help="Try to run generated ONNX model")
    args = parser.parse_args()
    return args

# Only import modules if these are needed.
# This makes it possible to run code parts with missing dependencies
def main(args, cfg):

    if not args.no_train:
        from failnet.train import train
        train(args, cfg)

    if args.make_onnx:
        from failnet.pytorch2onnx import generate_onnx
        generate_onnx(cfg)

    if args.run_onnx:
        from failnet.run_onnx import run_onnx
        run_onnx(args, cfg)

if __name__ == "__main__":
    # Read command line arguments
    args = parse_args()
    
    # Add directory including the config to search path
    config_dir = os.path.dirname(os.path.realpath(args.config_path))
    sys.path.append(config_dir)

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

    main(args, cfg)
