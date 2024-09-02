import numpy as np

class MLP:
    """
    A Multilayer Perceptron class
    """
    def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):
        """
        Constructor for the Multilayer Perceptron class. Takes number of inputs,
        a variable number of hidden layers, and a number of outputs.

        Arguments
            num_inputs (int): Number of inputs
            num_hidden (list): A list of hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # create random connection weights for the layers
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1])
            self.weights.append(w)

    def forward_propagate(self, inputs):
        """
        Computes forward propagation of the network based on inputs signals

        Arguments
            inputs (ndarray): Input signals
        Returns
            activations (ndarray): Output values
        """
        activations = inputs

        for w in self.weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            # calculate the activations
            activations =self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

if __name__ == '__main__':

    # create an MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward prop
    outputs = mlp.forward_propagate(inputs)

    # print the results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))
