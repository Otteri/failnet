import torch
import gym
import pulsegen
import config as cfg
import numpy as np
from model import Model, Sequence, Channel

# This script can be used for generatinx onnx model

# Some input data for tracing
n = cfg.predict_n
dummy_data = np.zeros((1, 1, cfg.signal_length))
dummy_input = torch.from_numpy(dummy_data[..., :-n]).to(cfg.device)

# Create a new model
nn_model = Model(training=False, device=cfg.device, load_path="failnet.pt").seq

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
