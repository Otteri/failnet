# FailNet - Simple and Robust Failure Detector

FailNet is a failure detector that can be easily integrated to any system. It works in a black-box manner, which allows it to be used easily with anykind of sequential data. Therefore, it doesn't matter much what kind of system you have. The detector aims to be generic, simple, lightweight, robust and reliable. All these properties can be achieved with a lightweight convolutional neural network which is trained to predict patterns and inform user if there is too great difference between predictions and real values.

## Installation
It is recommended to first create a virtual environment, then install requirements:
```bash
$ pip install -r requirements.txt
```
This installs all required python packages and data generation environemnt. (Installation has been found to work with python 3.8.5 at least. Shold work with other versions as well).

## Configuration
Model properties can be configured using `config.py`. Configuration file also allows to adjust generated data properties when using `pulsegen` environment.

## Usage
Now you should be able to train the model. Training takes couple of minutes with default configuration.
```bash
$ python train.py --make_plots
```
After training, you may inspect generated plots in `predictions` directory. You can also find the model weights from the same directory.

## ONNX
Pytorch model can be converted to ONNX model with `pytorch2onnx.py` script. ONNX model can run with `run_onnx.py`.

## Simulation environment
The detector can be tested and developed using simulation environment called
[pulsegen](https://github.com/Otteri/gym-envs). Pulsegen was used in the run examlpe. Another option is to feed recorded data for the model and use this for training.

## Testing & Linting
It is assumed that you are in the root of the repository and you have installed pytest and flake8. Then you can call:
```bash
pytest tests/ # run tests
flake8 .      # run linter
```
