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

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, stopaccuracy = 1.0):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        training, test = [], []
        n_train = len(training_data)
        for j in range(epochs):
            # random.shuffle(training_data) #4/2019 nt: supressed for now
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n_train, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            #evaluate()
            result_train = self.evaluate(training_data)
            training.append(result_train) 
            accu_train = result_train[1]
            print(("[Epoch {0}] Training: MSE={1}, CE={2}, LL={3}, Correct:{4}/{5}, Acc:{6}".format(
                j, result_train[2], result_train[3], result_train[4], result_train[0], n_train, result_train[1])))
            
            if test_data:
                result_test = self.evaluate(test_data)
                test.append(result_test) 
                n_test = len(test_data)
                print ("          Test:     MSE={1}, CE={2}, LL={3}, Correct:{4}/{5}, Acc:{6}".format(
                    j,result_test[2], result_test[3], result_test[4], result_test[0], n_test, result_test[1]))
            else:
                test.append([])
                  
            if (accu_train >= stopaccuracy):
                break
        
        return [training, test]


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

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
                  
        # forward pass          
        #initailize structure of activations to numpy zeros array
        activations=[np.zeros((i,1)) for i in range(0,len(self.sizes))]
        activation = x
        activations[0] = x  
        #index for layer
        index=1 

        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations[index] = activation
            index = index +1

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

        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) 
                        for (x, y) in test_data]
        
        length = len(test_data)
        count = sum(int(x == y) for (x, y) in test_results)
        accuarcy = count/length
        sum_MSE, sum_CE, sum_LL = 0,0,0
        
        #Iterate over each data in dataset
        for (x, y) in test_data:
            a = self.feedforward(x)
            max_a = a[np.argmax(y)][0]
            sum_MSE += (0.5 * np.linalg.norm(a - y)**2)
            sum_CE += np.sum(np.nan_to_num(-y * np.log(a)-(1 - y) * np.log (1 - a)))
            sum_LL += np.sum(np.nan_to_num(-np.log(max_a)))
        MSE = sum_MSE/length
        CE = sum_CE/length
        LL = sum_LL/length
        
        results_lst = [count, accuarcy, MSE, CE, LL]
        
        return results_lst

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
