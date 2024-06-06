import random
from typing import List
from autodiff.scalar import Scalar

class Module:
    def zero_grad(self) -> None:
        """
        Reset the gradients of all parameters to zero.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Scalar]:
        """
        Return a list of parameters of the module.
        """
        return []

class Neuron(Module):
    def __init__(self, num_inputs: int, use_relu=True):
        """
        Initialize the Neuron with the given number of inputs.

        :param num_inputs: Number of inputs that the neuron will receive
        :param use_relu: Whether to use ReLU activation function or no activation function
        """
        # We randomly initialize the weights of the neuron `self.w`
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(num_inputs)]
        # We initialize the bias `self.b` to 0
        self.b = Scalar(0)
        self.use_relu = use_relu

    def __call__(self, x: List[Scalar]) -> Scalar:
        """
        Forward pass through the neuron. Return a Scalar value, representing the output of the neuron.
        Apply the ReLU activation function if `self.use_relu` is True. Otherwise, use no activation function.
        Hint: Given a Scalar object `s`, you can compute the ReLU of `s` by calling `s.relu()`.

        :param x: List of Scalar values, representing the inputs to the neuron
        """
        z = sum(w * xi for w, xi in zip(self.w, x)) + self.b
        
        # Apply the activation function
        if self.use_relu:
            return z.relu()
        else:
            return z

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.use_relu else 'Linear'}Neuron({len(self.w)})"

class FeedForwardLayer(Module):
    def __init__(self, num_inputs: int, num_outputs: int, use_relu: bool):
        """
        Initialize the FeedForwardLayer with the given number of inputs and outputs.

        :param num_inputs: Number of inputs that each neuron in that layer will receive
        :param num_outputs: Number of neurons in that layer
        """
        self.neurons = [Neuron(num_inputs, use_relu) for _ in range(num_outputs)]

    def __call__(self, x: List[Scalar]) -> List[Scalar]:
        """
        Forward pass through the layer. Return a list of Scalars, where each Scalar is the output of a neuron.

        :param x: List of Scalar values, representing the input features
        """
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"FeedForwardLayer of [{', '.join(str(n) for n in self.neurons)}]"

class MultiLayerPerceptron(Module):
    def __init__(self, num_inputs: int, num_hidden: List[int], num_outputs: int):
        """
        Initialize the MultiLayerPerceptron with the given architecture.
        Note that num_inputs and num_outputs are integers, while num_hidden is a list of integers.

        :param num_inputs: Number of input features
        :param num_hidden: List of integers, where each integer represents the number of neurons in that hidden layer
        :param num_outputs: Number of output neurons
        """
        self.layers = [
            FeedForwardLayer(in_params, layer_neurons, use_relu=True)
            for in_params, layer_neurons in zip([num_inputs] + num_hidden, num_hidden)
        ]
        self.layers.append(FeedForwardLayer(num_hidden[-1], num_outputs, use_relu=False))

    def __call__(self, x: List[Scalar]) -> List[Scalar]:
        """
        Forward pass through the network.
        Call the first layer with the input x.
        Call each layer after that with the output of the previous layer.

        :param x: List of Scalar values, representing the input features
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MultiLayerPerceptron of [{', '.join(str(layer) for layer in self.layers)}]"
