from torch import nn


class FullyConnectedNetwork(nn.Module):
    """Builds a fully connected estimate network with the specified layer sizes"""

    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNetwork, self).__init__()

        layers = []

        # add input and hidden layers
        size_in = input_size
        for size_out in hidden_sizes:
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())

            size_in = size_out  # layer output size is size_in for the next layer

        layers.append(nn.Linear(size_in, output_size))  # add the output layer

        self.model = nn.Sequential(*layers)  # create a sequential model from the list of layers

    def forward(self, x):
        return self.model.forward(x)  # delegate to the underlying sequential model
