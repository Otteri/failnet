import torch
import gym
import pulsegen
import numpy as np
from failnet.model import Model

def generate_onnx(cfg):
    # Some input data for tracing
    n = cfg.predict_n
    dummy_data = np.zeros((1, 1, cfg.signal_length))
    dummy_input = torch.from_numpy(dummy_data[..., :-n]).to(cfg.device)

    # Create a new model
    nn_model = Model(
        load_path     = "failnet.pt",
        training      = False,
        device        = cfg.device,
        signal_length = cfg.signal_length,
        hidden        = cfg.hidden,
        predict_n     = cfg.predict_n,
        lr            = cfg.learning_rate,
        max_iter      = cfg.max_iter,
        history_size  = cfg.history_size
    ).seq

    # Only for readability
    input_names = [ "input_batch" ]
    output_names = [ "outpu_batch" ]

    output_file = "failnet.onnx"
    torch.onnx.export(nn_model,
                    dummy_input,
                    output_file,
                    verbose=False,
                    input_names=input_names,
                    output_names=output_names,
                    export_params=True
                    )

    print(f"Done, '{output_file}' has been generated!")
