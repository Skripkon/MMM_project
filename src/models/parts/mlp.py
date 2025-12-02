from torch import nn


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) model.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        """
        Args:
            input_dim (int): dimension of the input features.
            hidden_dims (list[int]): list of hidden layer dimensions.
            output_dim (int): dimension of the output features.
            activation (nn.Module): activation function to use.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): input tensor of shape (B, input_dim).
        Returns:
            output (Tensor): output tensor of shape (B, output_dim).
        """
        return self.network(x)
