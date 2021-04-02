import numpy as np
import config as cfg
import onnxruntime as ort
from detectors import compareSingleValue
from visualization import plot_signal

# Env related imports
import gym
import pulsegen

# Some test input data
# n = cfg.predict_n
# dummy_data = np.zeros((1, 1, cfg.signal_length))
# dummy_input = dummy_data[..., :-n]

n = cfg.predict_n
env = gym.make("FourierSeries-v0", config_path="config.py")
data = env.recordRotations(rotations=1, viz=False)
input_data = data[..., :-n]

# Create NN
ort_session = ort.InferenceSession('failnet.onnx')

# Run through NN
model_predictions = ort_session.run(None, {'input_batch': input_data})

# Visualize result
model_predictions = np.array(model_predictions) # convert to numpy
prediction = model_predictions[0, 0, :]
plot_signal(prediction)
