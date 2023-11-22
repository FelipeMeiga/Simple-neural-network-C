# Neural Network in C

## Overview
This repository contains a C implementation of a simple feedforward neural network. The network is designed with one hidden layer and is capable of performing basic binary classification tasks. It features ReLU and sigmoid activation functions for nonlinear transformations.

## Features
- **Neuron Structure**: Custom struct definition for neurons.
- **Layered Architecture**: Separate structs for input, hidden, and output layers.
- **Activation Functions**: ReLU and Sigmoid functions for activation.
- **Forward and Backward Propagation**: Implemented for training the network.
- **Dynamic Memory Allocation**: For neuron weights and biases.
- **Loss Calculation**: MSE (Mean Squared Error) used for loss calculation.

## Compilation and Execution
To compile and run this project, use the following command in a terminal:
```bash
gcc main.c -o neural_network -lm
./neural_network
