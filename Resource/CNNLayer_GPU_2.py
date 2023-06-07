# @title Convolution Layer
import numpy as np
from CNNModel import *
from numba import cuda, types as numba_types
# @title Convolution Layer
import numpy as np
from numba import cuda

#------------Convolution kernel--------------#

# @title Convolution Layer
import numpy as np
from numba import cuda, types as numba_types

#------------Convolution kernel--------------#


@cuda.jit
def conv_forward_kernel(d_inputs, weights, stride, outputs, activation):
    shared_input = cuda.shared.array((8, 6, 6), numba_types.float32)
    i_batch, row, col = cuda.grid(3)
    n_batch, n_filters, output_height, output_width = outputs.shape
    if(i_batch >= n_batch):
        return
    filter_size = weights.shape[-1]
    n_chanels = d_inputs.shape[1]
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    by = cuda.blockIdx.y*cuda.blockDim.y
    bz = cuda.blockIdx.z*cuda.blockDim.z
    for fillterIdx in range(n_filters):
        sum_val = 0
        for chanel_idx in range(n_chanels):
            for i in range(4):
                idx = (ty*4+tz)+i*16
                y, z = idx//6, idx % 6
                if(y >= 6):
                    break
                shared_input[tx, y, z] = d_inputs[i_batch,
                                                  chanel_idx, by+y, bz+z]
            cuda.syncthreads()
            for fillterRow in range(filter_size):
                for fillterCol in range(filter_size):
                    sum_val += shared_input[tx, fillterRow+ty, fillterCol+tz] * \
                        weights[fillterIdx, chanel_idx, fillterRow, fillterCol]
            cuda.syncthreads()

        if not (row >= output_height or col >= output_width):
            if(activation == 1 and sum_val < 0):
                sum_val = 0
            outputs[i_batch, fillterIdx, row, col] = sum_val


@cuda.jit
def conv_backward_kernel(d_inputs, weights, stride, input_gradient, output_gradient, filter_gradient, activation):
    shared_input = cuda.shared.array((8, 6, 6), numba_types.float32)
    n_chanels, filter_size = weights.shape[1:-1]
    n_batch, n_filters, output_height, output_width = output_gradient.shape
    i_batch, row, col = cuda.grid(3)
    # if(row >= output_height or col >= output_width or i_batch >= n_batch):
    #     return
    if(i_batch >= n_batch):
        return
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    by = cuda.blockIdx.y*cuda.blockDim.y
    bz = cuda.blockIdx.z*cuda.blockDim.z
    for fillterIdx in range(n_filters):
        out_value = output_gradient[i_batch, fillterIdx, row, col]
        for i_chanel in range(n_chanels):
            for i in range(3):
                idx = (ty*4+tz)+i*16
                y, z = idx//6, idx % 6
                if(y >= 6):
                    break
                shared_input[tx, y, z] = d_inputs[i_batch,
                                                  i_chanel, by+y, bz+z]
            cuda.syncthreads()
            if not (row >= output_height or col >= output_width):
                for fillterRow in range(filter_size):
                    for fillterCol in range(filter_size):
                        iR = row*stride + fillterRow
                        iC = col*stride + fillterCol
                        in_val = d_inputs[i_batch, i_chanel, iR, iC]
                        cuda.atomic.add(
                            filter_gradient, (fillterIdx, i_chanel, fillterRow, fillterCol), in_val * out_value)
                        if(not (in_val <= 0 and activation == 1)):
                            cuda.atomic.add(input_gradient, (i_batch, i_chanel, iR, iC),
                                            weights[fillterIdx, i_chanel, fillterRow, fillterCol] * out_value)
            cuda.syncthreads()


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
        output_width = (self.input_shape[2] -
                        self.filter_size) // self.stride + 1
        output_height = (
            self.input_shape[1] - self.filter_size) // self.stride + 1

        return (self.n_filters, output_height, output_width)

    def init_weight(self):
        self.weights = np.random.randn(
            self.n_filters, self.input_shape[0], self.filter_size, self.filter_size)/(self.filter_size**2)

    def forward(self, inputs):

        self.inputs = inputs
        n_batchs, n_chanels, in_height, in_width = inputs.shape
        assert self.input_shape == inputs.shape[1:], "Input shape incorrect"
        output_height, output_width = self.get_out_shape()[1:]
        block_size = (8, 4, 4)
        grid_size = ((n_batchs-1)//block_size[0]+1, (output_height-1) //
                     block_size[1]+1, (output_width-1)//block_size[2]+1)
        d_outputs = cuda.device_array(
            (n_batchs, self.n_filters, output_height, output_width))
        self.d_weights = cuda.to_device(self.weights)
        self.d_inputs = cuda.to_device(self.inputs)
        conv_forward_kernel[grid_size, block_size](
            self.d_inputs, self.d_weights, 1, d_outputs, int(self.activation == "relu"))
        outputs = d_outputs.copy_to_host()

        return outputs

    def backward(self, output_gradient, learning_rate):
        n_batchs, input_channels, input_height, input_width = self.inputs.shape
        _, n_filters,  output_height, output_width = output_gradient.shape
        block_size = (8, 4, 4)
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


# @title Maxpooling Layer
@cuda.jit
def maxPool2D_forward_kernel(d_inputs, outputs, stride, pool_size):
    share_size = 8
    shared_input = cuda.shared.array(
        (8, share_size, share_size), numba_types.float32)
    n_batchs, n_chanels, in_height, in_width = d_inputs.shape
    n_batchs, n_chanels, output_height, output_width = outputs.shape
    i_batch, out_h, out_w = cuda.grid(3)
    # Max pool over input
    if(i_batch >= n_batchs):
        return
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    by = cuda.blockIdx.y*cuda.blockDim.y * stride
    bz = cuda.blockIdx.z*cuda.blockDim.z * stride

    for i_chanel in range(n_chanels):
        for i in range(4):
            idx = (ty*4+tz)+i*16
            y, z = idx//share_size, idx % share_size
            if(y >= share_size):
                break
            shared_input[tx, y, z] = d_inputs[i_batch, i_chanel, by+y, bz+z]
        cuda.syncthreads()
        max_value = -np.inf
        if not (out_h >= output_height or out_w >= output_width):
            for h_pool in range(pool_size):
                for w_pool in range(pool_size):
                    max_value = max(
                        # max_value, d_inputs[i_batch, i_chanel,out_h*stride+h_pool, w_pool+out_w*stride])
                        max_value, shared_input[tx, ty*stride+h_pool, w_pool+tz*stride])
            outputs[i_batch, i_chanel, out_h, out_w] = max_value
        cuda.syncthreads()


@cuda.jit
def maxPool2D_backward_kernel(d_inputs, inputs_grad, outputs_grad, stride, pool_size):
    share_size = 8
    shared_input = cuda.shared.array(
        (8, share_size, share_size), numba_types.float32)
    n_batchs, n_chanels, in_height, in_width = d_inputs.shape
    n_batchs, n_chanels, output_height, output_width,  = outputs_grad.shape
    i_batch, out_h, out_w = cuda.grid(3)
    # Max pool over input
    if(i_batch >= n_batchs):
        return
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    by = cuda.blockIdx.y*cuda.blockDim.y * stride
    bz = cuda.blockIdx.z*cuda.blockDim.z * stride
    for i_chanel in range(n_chanels):
        for i in range(4):
            idx = (ty*4+tz)+i*16
            y, z = idx//share_size, idx % share_size
            if(y >= share_size):
                break
            shared_input[tx, y, z] = d_inputs[i_batch, i_chanel, by+y, bz+z]
        cuda.syncthreads()

        max_value = -np.inf
        if not (out_h >= output_height or out_w >= output_width):
            for h_pool in range(pool_size):
                for w_pool in range(pool_size):
                    max_value = max(
                        max_value,  shared_input[tx, ty*stride+h_pool, w_pool+tz*stride])

            for h_pool in range(pool_size):
                for w_pool in range(pool_size):
                    if(shared_input[tx, ty*stride+h_pool, w_pool+tz*stride] == max_value):
                        inputs_grad[i_batch, i_chanel, out_h*stride+h_pool, w_pool +
                                    out_w*stride] += outputs_grad[i_batch, i_chanel,  out_h, out_w]
        cuda.syncthreads()
#------------Linear kernel--------------#


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
        d_outputs = cuda.device_array(
            (batch_size, num_channels, output_height, output_width))
        block_size = (8, 4, 4)
        grid_size = ((batch_size-1)//block_size[0]+1, (output_height-1) //
                     block_size[1]+1, (output_width-1)//block_size[2]+1)
        self.d_inputs = cuda.to_device(inputs)
        maxPool2D_forward_kernel[grid_size, block_size](
            self.d_inputs, d_outputs, self.stride, self.pool_size)
        outputs = d_outputs.copy_to_host()

        return outputs

    def backward(self, output_gradient, learning_rate):
        batch_size, num_channels, output_height, output_width = output_gradient.shape
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


# @title Dense Layer

@cuda.jit
def dense_forward_kernel(inputs, weights, bias, outputs):
    row, col = cuda.grid(2)
    height = weights.shape[0]
    if(row >= outputs.shape[0] or col >= outputs.shape[1]):
        return
    sum = 0
    for i in range(inputs.shape[1]):
        sum += inputs[row, i] * weights[i, col]
    outputs[row, col] = sum + bias[0, col]


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
        # if(grid_size[0]*grid_size[1] < 128):
        #     self.use_device = False
        # outputs = None
        # if(self.use_device == False):
        #     outputs = np.dot(inputs, self.weights) + self.biases
        # else:
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
