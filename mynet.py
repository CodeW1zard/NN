import numpy as np
import random as rd
import time

# explanation about SGD alglorithm:
# http://www.cnblogs.com/maybe2030/p/5089753.html#_label2


class Network (object):
    def __init__(self, size):
        self.size = size                                                                # length: L+1
        self.weights = [np.random.randn(x, y)/np.sqrt(y) for x, y in zip(size[1:], size[:-1])]     # length: L
        self.biases = [np.random.randn(x, 1) for x in size[1:]]                         # length: L
        self.num_layers = len(size)-1                                                   # L
        self.accuracy = [0]
        self.velocity = [np.zeros(w.shape) for w in self.weights]

    def SGD(self, data, epochs, mini_batch_size, eta, lmbda=0.0, mu=0.0, test_data=None,\
            learning_rate_schedule=False):
        start = time.time()
        data = list(data)
        n = len(data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        epoch = 0
        while self.learning_schedule(learning_rate_schedule, epoch, epochs):
            mid = time.time()
            rd.shuffle(data)
            mini_batches = [data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n, mu)
            if test_data:
                count = self.evaluate(test_data)
                self.accuracy.append(count/n_test)
                print('Epoch {}:'.format(epoch + 1))
                print(' accuracy:   {}/{}'.format(count, n_test))
            else:
                print("Epoch {0} complete".format(epoch))
            print('  time:   {}'.format(str(time.time()-mid)))
            epoch += 1
        self.accuracy = self.accuracy[1:]
        print('Total time: {}'.format(str(time.time()-start)))

    def update_mini_batch(self, mini_batch, eta, lmbda, n, mu):
        batch_size = len(mini_batch)
        inputs = np.asarray([x.ravel() for x, t in mini_batch]).transpose()
        labels = np.asarray([t.ravel() for x, t in mini_batch]).transpose()

        nabla_biases, nabla_weights = self.backprop(inputs, labels)
        self.velocity = [mu*v - (eta/batch_size)*nw for v, nw in zip(self.velocity, nabla_weights)]
        self.biases = [(1-eta*(lmbda/n))*b - (eta / batch_size) * nb for b, nb in zip(self.biases, nabla_biases)]
        self.weights = [(1-eta*(lmbda/n))*w + v for w, v in zip(self.weights, self.velocity)]

    def backprop(self, inputs, labels):
        nabla_b = [0 for b in self.biases]
        nabla_w = [0 for w in self.weights]

        # feedforward
        activations, zs = self.feedforward(inputs)

        # backward pass
        delta = self.cost_derivative(zs[-1], labels)
        nabla_b[-1] = delta.sum(axis=1).reshape([len(delta), 1])
        nabla_w[-1] = delta.dot(zs[-2].transpose())
        for i in range(2, self.num_layers+1):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sigmoid_prime(activations[-i])
            nabla_b[-i] = delta.sum(axis=1).reshape(len(delta), 1)
            nabla_w[-i] = delta.dot(zs[-i-1].transpose())
        return nabla_b, nabla_w

    def feedforward(self, inputs):
        # inputs: 784×10 ndarray
        activations = []
        z = inputs
        zs = [inputs]
        for w, b in zip(self.weights, self.biases):
            activation = np.dot(w, z) + b
            activations.append(activation)
            z = sigmoid(activation)
            zs.append(z)
        return activations, zs

    def evaluate(self, test_data):
        test_data = list(test_data)
        count = 0
        for x, y in test_data:
            z = self.feedforward(x)[1][-1]
            count += (np.argmax(z) == y)
        return count

    def cost_derivative(self, y, labels):
        # return y-labels
        return y-labels

    def learning_schedule(self, threshold, epoch, epochs):
        if threshold == 0:
            return epoch < epochs
        elif threshold > 0:
            if epoch < threshold:
                return 1
            else:
                return max(self.accuracy[-threshold:]) >= max(self.accuracy[-2*threshold:-threshold])
        else:
            print('threshold should not be negative')
            return 0

def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))       # equal to sigmoid function after some simple transform


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))





