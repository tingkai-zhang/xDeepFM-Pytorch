import torch
from torch.nn import Sequential, Linear

from reclib.modules import FactorizationMachine
from reclib.modules import FeedForward
from reclib.modules.embedders import Embedding
from reclib.modules.embedders import LinearEmbedder


class NeuralFactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Parameters
    ----------
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = Embedding(field_dims, embed_dim)
        self.linear = LinearEmbedder(field_dims, 1)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )

        self.mlp = Sequential(FeedForward(num_layers=1,
                                          input_dim=embed_dim,
                                          hidden_dims=mlp_dims,
                                          batch_norm=True,
                                          activations=torch.nn.ReLU(),
                                          dropouts=dropouts[1]),
                              Linear(mlp_dims[-1], 1))

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.LongTensor
            Shape of ``(batch_size, num_fields)``

        Returns
        -------
        label_logits:
            A tensor of shape ``(batch_size, num_labels)`` representing un-normalised log
            probabilities of the label.
        """
        # ``(batch_size, embed_dim)``
        cross_term = self.fm(self.embedding(x))
        tmp = self.linear(x) + self.mlp(cross_term)
        label_logits = torch.sigmoid(tmp.squeeze(1))
        return label_logits
