"""
NN578_network.py
==============

nt: Modified from the NNDL book code "network.py".

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import json
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, stopaccuracy=1.0):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        training_results = []
        test_results = []
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # random.shuffle(training_data) #4/2019 nt: supressed for now
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            results = self.evaluate(training_data)
            training_results += results
            # print("results: {}".format(results))
            pdata = results + [j, n]
            # print("everything: {}".format(pdata))
            print(
                "[Epoch {5}]: Training: MSE={2}, CE={3}, LL={4}, Correct={0}/{6}, Acc: {1}".format(
                    *pdata
                )
            )
            if test_data:
                results = self.evaluate(test_data)
                pdata = results + [j, n]
                test_results += results
                print(
                    "[            Test:     MSE={2}, CE={3}, LL={4}, Correct={0}/{6}, Acc: {1}".format(
                        *pdata
                    )
                )

            ## Early stopping -- comparing against stopaccuracy
            if train_results[1] >= stopaccuracy:
                break

        return [training_results, test_results]

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    # to make this pre-allocate the activations, calculate dimensions each time?
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward pass
        activations = [np.zeros((nodes, 1)) for nodes in self.sizes]
        activations[0] = x
        zs = []  # list to store all the z vectors, layer by layer
        for layer in range(self.num_layers - 1):
            # for b, w in zip(self.biases, self.weights):
            z = np.dot(self.weights[layer], activations[layer]) + self.biases[layer]
            zs.append(z)
            activations[layer + 1] = sigmoid(z)
            # print("act{} size{}".format(layer + 1, np.shape(activations[layer + 1])))

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result."""
        # nt: Changed so the target (y) is a one-hot vector -- a vector of
        #  0's with exactly one 1 at the index where the targt is true.

        num = len(test_data)

        # tuples of tds and ods
        tds_ods = [(y, self.feedforward(x)) for (x, y) in test_data]

        cat_results = [(np.argmax(td), np.argmax(od)) for (td, od) in tds_ods]
        correct = sum(int(ti == oi) for (ti, oi) in cat_results)

        # For each output node
        # 1/2N * sum_d(td - od)^2
        # vec length = sqrt(sum(td - od)^2)
        # vec length ^ 2 = (td - od)^2
        # sum over the output nodes and the training examples
        MSE = sum(sum((td - od) ** 2)[0] for (td, od) in tds_ods) / (2 * num)

        # assuming element-wise multiplication
        CE = (
            -sum(
                sum(np.nan_to_num(td * np.log(od) + (1 - td) * np.log(1 - od)))[0]
                for (td, od) in tds_ods
            )
            / num
        )

        # - ln a_y^L where y here is the index for target, or argmax(td)
        LL = sum([np.nan_to_num(-np.log(od[np.argmax(td)])) for (td, od) in tds_ods])[0]

        return [correct, correct / num, MSE, CE, LL]

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y


# Miscellaneous functions


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


# Saving a Network to a json file
def save_network(net, filename):
    """Save the neural network to the file ``filename``."""
    data = {
        "sizes": net.sizes,
        "weights": [w.tolist() for w in net.weights],
        "biases": [b.tolist() for b in net.biases]  # ,
        # "cost": str(net.cost.__name__)
    }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()


# Loading a Network from a json file
def load_network(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network. """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    # cost = getattr(sys.modules[__name__], data["cost"])
    # net = Network(data["sizes"], cost=cost)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


# Miscellaneous functions
def vectorize_target(n, target):
    """Return an array of shape (n,1) with a 1.0 in the target position
    and zeroes elsewhere.  The parameter target is assumed to be
    an array of size 1, and the 0th item is the target position (1). """
    e = np.zeros((n, 1))
    e[int(target[0])] = 1.0
    return e


#######################################################
#### ADDITION to load a saved network

# Function to load the train-test (separate) data files.
# Note the target (y) is assumed to be already in the one-hot-vector notation.


def my_load_csv(fname, no_trainfeatures, no_testfeatures):
    ret = np.genfromtxt(fname, delimiter=",")
    data = np.array(
        [(entry[:no_trainfeatures], entry[no_trainfeatures:]) for entry in ret]
    )
    temp_inputs = [np.reshape(x, (no_trainfeatures, 1)) for x in data[:, 0]]
    temp_results = [np.reshape(y, (no_testfeatures, 1)) for y in data[:, 1]]
    dataset = list(zip(temp_inputs, temp_results))
    return dataset
