# Minseq
A tiny convolutional neural network for sequential data. Learns to predict patterns and is capable of predicting future values, if data patterns are repeatitive from nature.

Uses [pulsegen](https://github.com/Otteri/gym-envs) for generating training data by default. However, data can be from anywhere as long as it follows model requirements. Model itself is quite felxible, but for efficient computation, data is processed in blocks. Hence, data should be provided as 3D arrays: [B, S, L], where first dimension represents batches, second signals (channels) and third data length. Length and batch sizes can be freely configured.

Example call for training the model:  
`$ python train.py --steps 15 --make_plots`

## Configuration
Model properties can be configured using `config.py`. Configuration file also allows to adjust generated data properties when using `pulsegen` environment.

## Installation
`$ pip install -r requirements.txt`  
This should install all required python packages and data generation environemnt.
