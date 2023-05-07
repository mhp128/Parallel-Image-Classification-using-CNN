import numpy as np
from numba import cuda
import time
from kernels import *


class Layer():

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

    def get_out_shape(self):
        pass

    def init_weight(self):
        pass


class Convolution(Layer):
    def __init__(self, n_filters=32, filter_size=3, stride=1, activation=None, input_shape=(28, 28, 1)):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.activation = activation
        self.use_device = False
        self.bias = np.zeros((n_filters, 1))
        self.init_weight()

    def get_out_shape(self):
        output_width = (self.input_shape[1] -
                        self.filter_size) // self.stride + 1
        output_height = (
            self.input_shape[0] - self.filter_size) // self.stride + 1
        return (output_height, output_width,  self.n_filters)

    def init_weight(self):
        self.weights = np.random.randn(
            self.n_filters, self.filter_size, self.filter_size, self.input_shape[-1])/(self.filter_size**2)

    def forward(self, inputs):

        self.inputs = inputs
        n_batchs, in_height, in_width, n_chanels = inputs.shape
        assert self.input_shape == inputs.shape[1:], "Input shape incorrect"
        output_height, output_width = self.get_out_shape()[:-1]
        outputs = None
        ## ===========USING CPU===========##
        if(self.use_device == False):
            outputs = np.zeros(
                (n_batchs, output_height, output_width,  self.n_filters))
            _weight = self.weights.transpose(0, 3, 1, 2)
            _inputs = inputs.transpose(0, 3, 1, 2)
            for i_idx in range(n_batchs):
                for row in range(output_height):
                    for col in range(output_width):
                        for f_idx in range(self.n_filters):
                            row_start = row * self.stride
                            row_end = row_start + self.filter_size
                            col_start = col * self.stride
                            col_end = col_start + self.filter_size
                            outputs[i_idx, row, col, f_idx] = np.sum(
                                _weight[f_idx] * _inputs[i_idx, :, row_start:row_end, col_start:col_end])

            if(self.activation == "relu"):
                outputs = np.maximum(0, outputs)
        ## ===========USING DEVICE===========##
        else:
            block_size = (8, 8, 8)
            grid_size = ((n_batchs-1)//block_size[0]+1, (output_height-1) //
                         block_size[1]+1, (output_width-1)//block_size[2]+1)
            d_outputs = cuda.device_array(
                (n_batchs, output_height, output_width,  self.n_filters))
            self.d_weights = cuda.to_device(self.weights)
            self.d_inputs = cuda.to_device(self.inputs)
            conv_forward_kernel[grid_size, block_size](
                self.d_inputs, self.d_weights, 1, d_outputs, int(self.activation == "relu"))
            cuda.synchronize()
            outputs = d_outputs.copy_to_host()
        ## ===========END USING DEVICE===========##

        return outputs

    def backward(self, output_gradient, learning_rate):
        n_batchs, input_height, input_width, input_channels = self.inputs.shape
        _,  output_height, output_width, n_filters = output_gradient.shape

        filter_gradient = None
        input_gradient = None

        ## ===========USING CPU===========##
        if(self.use_device == False):
            filter_gradient = np.zeros(self.weights.shape)
            input_gradient = np.zeros(self.inputs.shape)
            for i_batch in range(n_batchs):
                for row in range(output_height):
                    for col in range(output_width):
                        for fillterIdx in range(n_filters):
                            row_start = row * self.stride
                            row_end = row_start + self.filter_size
                            col_start = col * self.stride
                            col_end = col_start + self.filter_size
                            for i_chanel in range(self.weights.shape[-1]):
                                out_grad_val = output_gradient[i_batch,
                                                               row, col, fillterIdx]
                                filter_gradient[fillterIdx, :, :, i_chanel] += self.inputs[i_batch,
                                                                                           row_start:row_end, col_start:col_end, i_chanel] * out_grad_val
                                input_gradient[i_batch,  row_start:row_end, col_start:col_end,
                                               i_chanel] += self.weights[fillterIdx, :, :, i_chanel] * out_grad_val
            if(self.activation == "relu"):
                input_gradient[self.inputs <= 0] = 0

           ## ===========USING DEVICE===========##
        else:
            block_size = (4, 4, 4)
            grid_size = ((n_batchs-1)//block_size[0]+1, (output_height-1) //
                         block_size[1]+1, (output_width-1)//block_size[2]+1)
            d_filter_grad = cuda.device_array(self.weights.shape)
            d_input_grad = cuda.device_array(self.inputs.shape)
            d_output_grad = cuda.to_device(output_gradient)
            # call kernel
            conv_backward_kernel[grid_size, block_size](
                self.d_inputs, self.d_weights, 1, d_input_grad, d_output_grad, d_filter_grad, int(self.activation == "relu"))
            cuda.synchronize()
            input_gradient = d_input_grad.copy_to_host()
            filter_gradient = d_filter_grad.copy_to_host()
        ## ===========END USING DEVICE===========##
        self.weights -= learning_rate * filter_gradient/n_batchs

        return input_gradient


class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2, input_shape=(28, 28, 1)):
        self.pool_size = pool_size
        self.stride = stride
        self.use_device = False
        self.inputs = None
        self.inputs_device = None
        self.input_shape = input_shape

    def get_out_shape(self):
        output_height = (
            self.input_shape[0] - self.pool_size) // self.stride + 1
        output_width = (self.input_shape[1] -
                        self.pool_size) // self.stride + 1
        return (output_height, output_width, self.input_shape[-1])

    def forward(self, inputs):
        # Save input
        batch_size, input_height, input_width, num_channels = inputs.shape
        assert self.input_shape == inputs.shape[1:], "Input shape incorrect"
        self.inputs = inputs
        (output_height, output_width, _) = self.get_out_shape()
        outputs = None
        ## ===========USING CPU===========##
        if(self.use_device == False):
            outputs = np.zeros(
                (batch_size, output_height, output_width, num_channels))
            for c in range(num_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        outputs[:, h, w, c] = np.max(
                            inputs[:, h_start:h_end, w_start:w_end, c], axis=(1, 2))
        ## ===========USING DEVICE===========##
        else:
            d_outputs = cuda.device_array(
                (batch_size, output_height, output_width, num_channels))
            block_size = (8, 4, 4)
            grid_size = ((batch_size-1)//block_size[0]+1, (output_height-1) //
                         block_size[1]+1, (output_width-1)//block_size[2]+1)
            self.d_inputs = cuda.to_device(inputs)
            maxPool2D_forward_kernel[grid_size, block_size](
                self.d_inputs, d_outputs, self.stride, self.pool_size)
            outputs = d_outputs.copy_to_host()
        ## ===========END USING DEVICE===========##

        return outputs

    def backward(self, output_gradient, learning_rate):
        batch_size, output_height, output_width, num_channels = output_gradient.shape
        input_gradient = None
        ## =========== USING HOST===========##
        if(self.use_device == False):
            input_gradient = np.zeros(self.inputs.shape)
            for c in range(num_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        input_slice = self.inputs[:,
                                                  h_start:h_end, w_start:w_end, c]
                        max_vals = np.max(
                            input_slice, axis=(1, 2), keepdims=True)
                        max_mask = (input_slice == max_vals)
                        input_gradient[:, h_start:h_end, w_start:w_end, c] += max_mask * \
                            output_gradient[:,  h, w, c,
                                            np.newaxis, np.newaxis]
        ## =========== USING DEVICE===========##
        else:
            d_input_grad = cuda.device_array(self.inputs.shape)
            d_output_grad = cuda.to_device(output_gradient)
            block_size = (8, 4, 4)
            grid_size = ((batch_size-1)//block_size[0]+1, (output_height-1) //
                         block_size[1]+1, (output_width-1)//block_size[2]+1)
            maxPool2D_backward_kernel[grid_size, block_size](
                self.d_inputs, d_input_grad, d_output_grad, self.stride, self.pool_size)
            input_gradient = d_input_grad.copy_to_host()

        return input_gradient

    def init_weight(self):
        pass


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
        block_size = (8, 4)
        grid_size = ((inputs.shape[0]-1)//block_size[0]+1,
                     (self.num_outputs-1)//block_size[1]+1)
        if(grid_size[0]*grid_size[1] < 128):
            self.use_device = False
        outputs = None
        if(self.use_device == False):
            outputs = np.dot(inputs, self.weights) + self.biases
        else:
            self.d_weights = cuda.to_device(self.weights)
            self.d_biases = cuda.to_device(self.biases)
            d_outputs = cuda.device_array((inputs.shape[0], self.num_outputs))
            self.d_inputs = cuda.to_device(inputs)
            start = time.time()
            dense_forward_kernel[grid_size, block_size](
                self.d_inputs, self.d_weights, self.d_biases, d_outputs)
            outputs = d_outputs.copy_to_host()

        # if(self.activation=="relu"):
        #   outputs = np.maximum(0,outputs)
        if self.activation == "softmax":
            outputs = self.softmax(outputs)
        return outputs

    def softmax(self, x):
        e_x = np.exp(x-np.max(x, axis=1, keepdims=True))
        return e_x/e_x.sum(axis=1, keepdims=True)

    def backward(self, output_gradient, learning_rate):
        # start = time.time()
        # input_grad=None
        # if(self.use_device==False):

        input_grad = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.inputs.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_grad


class Flatten(Layer):
    def __init__(self, input_shape=(28, 28, 1)):
        self.input_shape = input_shape
        pass

    def get_out_shape(self):
        t = 1
        for i in self.input_shape:
            t *= i
        return t

    def forward(self, inputs):
        self.inputs = inputs
        assert self.input_shape == inputs.shape[1:], "Input shape incorrect"
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, output_gradient, learning_rate):
        shape = self.inputs.shape
        return output_gradient.reshape(shape)

    def init_weight(self):
        pass
