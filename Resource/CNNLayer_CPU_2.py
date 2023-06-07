import numpy as np

from CNNModel import *
import numpy as np
from numba import cuda, types as numba_types


# @title Convolution Layer
class Convolution(Layer):
    def __init__(self, n_filters=32, filter_size=3, stride=1, activation=None, input_shape=(1, 28, 28)):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.activation = activation
        self.use_device = False
        self.bias = np.zeros((n_filters, 1))
        self.init_weight()

    def get_out_shape(self):
        output_width = (self.input_shape[2] -
                        self.filter_size) // self.stride + 1
        output_height = (
            self.input_shape[1] - self.filter_size) // self.stride + 1

        return (self.n_filters, output_height, output_width)

    def init_weight(self):
        np.random.seed(10)
        self.weights = np.random.randn(
            self.n_filters, self.input_shape[0], self.filter_size, self.filter_size)/(self.filter_size**2)

    def forward(self, inputs):
        self.inputs = inputs
        n_batchs, n_chanels, in_height, in_width = inputs.shape
        assert self.input_shape == inputs.shape[1:], "Input shape incorrect"
        output_height, output_width = self.get_out_shape()[1:]
        outputs = np.zeros(
            (n_batchs, self.n_filters, output_height, output_width))
        for row in range(output_height):
            for col in range(output_width):
                for f_idx in range(self.n_filters):
                    row_start = row * self.stride
                    row_end = row_start + self.filter_size
                    col_start = col * self.stride
                    col_end = col_start + self.filter_size
                    outputs[:, f_idx, row, col] = np.sum(
                        self.weights[f_idx]*inputs[:, :, row_start:row_end, col_start:col_end], axis=(1, 2, 3))

        if(self.activation == "relu"):
            outputs = np.maximum(0, outputs)

        return outputs

    def backward(self, output_gradient, learning_rate):
        n_batchs, input_channels, input_height, input_width = self.inputs.shape
        _, n_filters,  output_height, output_width = output_gradient.shape

        filter_gradient = np.zeros(self.weights.shape)
        input_gradient = np.zeros(self.inputs.shape)
        # for i_batch in range(n_batchs):
        for row in range(output_height):
            for col in range(output_width):
                for fillterIdx in range(n_filters):
                    row_start = row * self.stride
                    row_end = row_start + self.filter_size
                    col_start = col * self.stride
                    col_end = col_start + self.filter_size
                    out_grad_val = output_gradient[:, fillterIdx,
                                                   row, col, np.newaxis, np.newaxis, np.newaxis]
                    filter_gradient[fillterIdx] += np.sum(
                        self.inputs[:, :, row_start:row_end, col_start:col_end] * out_grad_val, axis=0)
                    input_gradient[:, :, row_start:row_end,
                                   col_start:col_end] += self.weights[fillterIdx] * out_grad_val

        if(self.activation == "relu"):
            input_gradient[self.inputs <= 0] = 0

        self.weights -= learning_rate * filter_gradient/n_batchs
        return input_gradient


# @title Maxpooling Layer

class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2, input_shape=(1, 28, 28)):
        self.pool_size = pool_size
        self.stride = stride
        self.use_device = False
        self.inputs = None
        self.inputs_device = None
        self.input_shape = input_shape

    def get_out_shape(self):
        output_height = (
            self.input_shape[1] - self.pool_size) // self.stride + 1
        output_width = (self.input_shape[2] -
                        self.pool_size) // self.stride + 1
        return (self.input_shape[0], output_height, output_width)

    def forward(self, inputs):
        # Save input
        batch_size, num_channels, input_height, input_width = inputs.shape
        assert self.input_shape == inputs.shape[1:], "Input shape incorrect"
        self.inputs = inputs
        (_, output_height, output_width) = self.get_out_shape()

        outputs = np.zeros(
            (batch_size, num_channels, output_height, output_width))
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size
                outputs[:, :, h, w] = np.max(
                    inputs[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return outputs

    def backward(self, output_gradient, learning_rate):
        batch_size, num_channels, output_height, output_width = output_gradient.shape
        input_gradient = np.zeros(self.inputs.shape)
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size
                input_slice = self.inputs[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(
                    input_slice, axis=(2, 3), keepdims=True)
                max_mask = (input_slice == max_vals)
                input_gradient[:, :, h_start:h_end, w_start:w_end] += max_mask * \
                    output_gradient[:, :,  h, w,  np.newaxis, np.newaxis]
        return input_gradient

    def init_weight(self):
        pass

# @title Dense Layer


class Dense(Layer):
    def __init__(self, num_outputs, activation=None, input_shape=100):
        self.num_outputs = num_outputs
        self.biases = np.zeros((1, num_outputs))
        self.activation = activation
        self.use_device = False
        self.inputs = None
        self.input_shape = input_shape
        self.init_weight()

    def init_weight(self):
        self.weights = np.random.randn(
            self.input_shape, self.num_outputs) / self.num_outputs

    def get_out_shape(self):
        return self.num_outputs

    def forward(self, inputs):
        self.inputs = inputs
        assert self.input_shape == inputs.shape[-1], "Input shape incorrect"
        outputs = np.dot(inputs, self.weights) + self.biases
        if self.activation == "softmax":
            outputs = self.softmax(outputs)
        return outputs

    def softmax(self, x):
        e_x = np.exp(x-np.max(x, axis=1, keepdims=True))
        return e_x/e_x.sum(axis=1, keepdims=True)

    def backward(self, output_gradient, learning_rate):
        input_grad = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.inputs.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        return input_grad


