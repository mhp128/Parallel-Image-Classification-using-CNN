from layers import Layer
import numpy as np


class CNNModel:
    def __init__(self, layers: list[Layer] = []):
        pre_layer = layers[0]
        pre_layer.init_weight()
        for layer in layers[1:]:
            layer.input_shape = pre_layer.get_out_shape()
            layer.init_weight()
            pre_layer = layer
        self.layers: list[Layer] = layers

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, out_grad, learning_rate):
        for layer in reversed(self.layers):
            out_grad = layer.backward(out_grad, learning_rate)
        return out_grad

    def fit(self, X_train, Y_train, epochs=1, batch_size=32, learning_rate=0.001):
        num_batch = (len(X_train)-1)//batch_size+1
        for i_epoch in range(epochs):
            print(f"\nEpoch {i_epoch+1}/{epochs}:")
            train_loss = 0
            acc = 0
            progress = '.'*30
            for i in range(num_batch-1):

                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size
                batch_X = X_train[batch_start: batch_end]
                batch_Y = Y_train[batch_start: batch_end]
                predictions = self.forward(batch_X)
                out_grad = 2.0 * (predictions - batch_Y)
                self.backward(out_grad, learning_rate)

                # print result
                acc_batch = np.mean(
                    np.argmax(predictions, axis=1) == np.argmax(batch_Y, axis=1))
                acc += acc_batch
                loss = np.sum((predictions - batch_Y) ** 2)
                train_loss += loss
                i_str = int(i/num_batch*30)
                progress = progress[:i_str] + ">" + progress[i_str+1:]
                print(
                    f"\r {i}/{num_batch} [{progress}] accuaray: {acc_batch:.5f}, train loss = {loss/len(batch_Y):.5f}", end='')
                progress = progress[:i_str] + "=" + progress[i_str+1:]

            train_loss /= len(X_train)

            print(
                f"\r {num_batch}/{num_batch} [{progress}] accuaray: {acc/num_batch:.5f}, train loss = {train_loss:.5f}", end='')

    def predict(self, X):
        return self.forward(X)

    def use_device(self, value):
        for layer in self.layers:
            output = layer.use_device = value
