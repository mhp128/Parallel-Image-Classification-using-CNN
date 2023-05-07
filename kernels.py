import numpy as np
from numba import cuda

#------------Convolution kernel--------------#


@cuda.jit
def conv_forward_kernel(input, weights, stride, output, activation):
    n_chanels = input.shape[-1]
    n_filters, filter_size = weights.shape[:2]
    n_batch, output_height, output_width, _ = output.shape
    i_batch, row, col = cuda.grid(3)
    if(row >= output_height or col >= output_width or i_batch >= n_batch):
        return

    for fillterIdx in range(n_filters):
        sum = 0
        for chanel_idx in range(n_chanels):
            for fillterRow in range(filter_size):
                for fillterCol in range(filter_size):
                    iR = row*stride + fillterRow
                    iC = col*stride + fillterCol
                    sum += input[i_batch, iR, iC, chanel_idx] * \
                        weights[fillterIdx, fillterRow, fillterCol, chanel_idx]
        if(activation == 1 and sum < 0):
            sum = 0
        output[i_batch, row, col, fillterIdx] = sum


@cuda.jit
def conv_backward_kernel(input, weights, stride, input_gradient, output_gradient, filter_gradient, activation):
    filter_size = weights.shape[1]
    n_chanels = input.shape[-1]
    n_batch, output_height, output_width, n_filters = output_gradient.shape
    i_batch, row, col = cuda.grid(3)
    if(row >= output_height or col >= output_width or i_batch >= n_batch):
        return

    for fillterIdx in range(n_filters):
        for fillterRow in range(filter_size):
            for fillterCol in range(filter_size):
                for i_chanel in range(n_chanels):
                    iR = row*stride + fillterRow
                    iC = col*stride + fillterCol
                    out_value = output_gradient[i_batch, row, col, fillterIdx]
                    in_val = input[i_batch, iR, iC, i_chanel]
                    cuda.atomic.add(
                        filter_gradient, (fillterIdx, fillterRow, fillterCol, i_chanel), in_val * out_value)
                    if(not (in_val <= 0 and activation == 1)):
                        cuda.atomic.add(input_gradient, (i_batch, iR, iC, i_chanel),
                                        weights[fillterIdx, fillterRow, fillterCol, i_chanel] * out_value)

#------------MaxPool2D kernel--------------#


@cuda.jit
def maxPool2D_forward_kernel(inputs, outputs, stride, pool_size):
    n_batchs, in_height, in_width, n_chanels = inputs.shape
    n_batchs, output_height, output_width, _ = outputs.shape
    ibatch, out_h, out_w = cuda.grid(3)
    # Max pool over input
    if(ibatch >= n_batchs or out_h >= output_height or out_w >= output_width):
        return

    for i_chanel in range(n_chanels):
        max_value = -np.inf
        for h_pool in range(pool_size):
            for w_pool in range(pool_size):
                max_value = max(
                    max_value, inputs[ibatch, out_h*stride+h_pool, w_pool+out_w*stride, i_chanel])
        outputs[ibatch, out_h, out_w, i_chanel] = max_value


@cuda.jit
def maxPool2D_backward_kernel(inputs, inputs_grad, outputs_grad, stride, pool_size):
    n_batchs, in_height, in_width, n_chanels = inputs.shape
    n_batchs, output_height, output_width, _ = outputs_grad.shape
    ibatch, out_h, out_w = cuda.grid(3)
    # Max pool over input
    if(ibatch >= n_batchs or out_h >= output_height or out_w >= output_width):
        return
    for i_chanel in range(n_chanels):
        max_value = -np.inf
        for h_pool in range(pool_size):
            for w_pool in range(pool_size):
                max_value = max(
                    max_value, inputs[ibatch, out_h*stride+h_pool, w_pool+out_w*stride, i_chanel])

        for h_pool in range(pool_size):
            for w_pool in range(pool_size):
                if(inputs[ibatch, out_h*stride+h_pool, w_pool+out_w*stride, i_chanel] == max_value):
                    inputs_grad[ibatch, out_h*stride+h_pool, w_pool+out_w*stride,
                                i_chanel] += outputs_grad[ibatch,  out_h, out_w, i_chanel]

#------------Linear kernel--------------#


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
