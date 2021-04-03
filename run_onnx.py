import numpy as np
import config as cfg
import onnxruntime as ort
from detectors import compareSingleValue
from visualization import plot_signals

# Env related imports
import gym
import pulsegen

# Some test data that the model has been trying to learn
n = cfg.predict_n
env = gym.make("FourierSeries-v0", config_path="config.py")
data = env.recordRotations(rotations=1, viz=False)
input_data = data[..., :-n]
target_data = data[..., n:]

# Create NN
ort_session = ort.InferenceSession('failnet.onnx')

# Run through NN
model_predictions = ort_session.run(None, {'input_batch': input_data})

# Visualize result
model_predictions = np.array(model_predictions) # convert to numpy
prediction = model_predictions[0, 0, :]
actual = target_data[0, 0, :]
plot_signals([prediction, actual])

# Check if failures
is_failure = compareSingleValue(actual, prediction, epsilon=2.0)
print("Failure(s): ", is_failure)
