import torch
from torch.nn import Sequential, Linear

from reclib.modules import CrossNetwork
from reclib.modules import FeedForward
from reclib.modules.embedders import LinearEmbedder, Embedding


class DeepCrossNetwork(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Parameters
    ----------
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = Embedding(field_dims, embed_dim)
        self.linear = LinearEmbedder(field_dims, 1)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.cn_output = torch.nn.Linear(self.embed_output_dim, 1)
        if isinstance(mlp_dims, int):
            mlp_layers = 1
        else:
            mlp_layers = len(mlp_dims)

        self.mlp = Sequential(FeedForward(num_layers=mlp_layers,
                                          input_dim=self.embed_output_dim,
                                          hidden_dims=mlp_dims,
                                          batch_norm=True,
                                          activations=torch.nn.ReLU(),
                                          dropouts=dropout),
                              Linear(mlp_dims[-1], 1))

    def forward(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        cross_term = self.cn(embed_x)
        x = self.linear(x) + self.cn_output(cross_term) + self.mlp(embed_x)
        return torch.sigmoid(x.squeeze(1))
